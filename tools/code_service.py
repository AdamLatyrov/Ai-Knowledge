from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
import sqlite3
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable

import sqlite_vec
from fastembed import TextEmbedding
from tree_sitter_language_pack import detect_language_from_path, get_parser

try:
    from tools.serena_provider import (
        DEFAULT_SERENA_COMMAND,
        SerenaProvider,
        SerenaProviderError,
        command_available,
    )
except ModuleNotFoundError:  # direct `python tools/code_service.py` execution
    from serena_provider import (  # type: ignore[no-redef]
        DEFAULT_SERENA_COMMAND,
        SerenaProvider,
        SerenaProviderError,
        command_available,
    )


ROOT = Path(__file__).resolve().parent.parent
CONFIG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
VECTOR = CONFIG["vector"]
CODE_CONFIG = CONFIG.get("code", {})
DEFAULT_DB_PATH = ROOT / CODE_CONFIG.get("database", "code-index.sqlite3")
CODE_PROVIDER = str(CODE_CONFIG.get("provider", "serena")).strip().lower()
SERENA_COMMAND = tuple(CODE_CONFIG.get("serena", {}).get("command", DEFAULT_SERENA_COMMAND))
SERENA_TIMEOUT_SECONDS = float(CODE_CONFIG.get("serena", {}).get("timeout_seconds", 180))
MODEL_NAME = CODE_CONFIG.get("model", VECTOR["model"])
DIMENSIONS = int(CODE_CONFIG.get("dimensions", VECTOR["dimensions"]))
MAX_FILE_BYTES = int(CODE_CONFIG.get("max_file_bytes", 500_000))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


SCHEMA = """
CREATE TABLE IF NOT EXISTS code_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
) STRICT;
CREATE TABLE IF NOT EXISTS repositories (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE COLLATE NOCASE,
    root_path TEXT NOT NULL UNIQUE,
    manifest_hash TEXT,
    registered_at TEXT NOT NULL,
    indexed_at TEXT
) STRICT;
CREATE TABLE IF NOT EXISTS code_files (
    id INTEGER PRIMARY KEY,
    repository_id INTEGER NOT NULL,
    path TEXT NOT NULL,
    language TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    modified_ns INTEGER NOT NULL,
    indexed_at TEXT NOT NULL,
    UNIQUE(repository_id, path),
    FOREIGN KEY(repository_id) REFERENCES repositories(id) ON DELETE CASCADE
) STRICT;
CREATE TABLE IF NOT EXISTS code_chunks (
    id INTEGER PRIMARY KEY,
    file_id INTEGER NOT NULL,
    chunk_index INTEGER NOT NULL,
    symbol TEXT NOT NULL,
    symbol_kind TEXT NOT NULL,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    token_estimate INTEGER NOT NULL,
    FOREIGN KEY(file_id) REFERENCES code_files(id) ON DELETE CASCADE
) STRICT;
CREATE INDEX IF NOT EXISTS idx_code_chunks_file ON code_chunks(file_id);
CREATE INDEX IF NOT EXISTS idx_code_chunks_symbol ON code_chunks(symbol);
CREATE VIRTUAL TABLE IF NOT EXISTS code_chunks_fts USING fts5(
    path,
    symbol,
    content,
    tokenize='unicode61 remove_diacritics 2'
);
"""


EXTENSIONS = {
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".java",
    ".ps1",
    ".cs",
    ".go",
    ".rs",
    ".c",
    ".h",
    ".cpp",
    ".hpp",
    ".sql",
    ".svelte",
    ".vue",
}
EXCLUDED_DIRS = {
    ".git",
    ".idea",
    ".vscode",
    ".venv",
    ".venv_docs",
    "node_modules",
    "build",
    "dist",
    "bin",
    "obj",
    "coverage",
    ".next",
    ".svelte-kit",
    "vendor",
}
SYMBOL_KINDS = {
    "function_definition": "function",
    "class_definition": "class",
    "function_declaration": "function",
    "generator_function_declaration": "function",
    "class_declaration": "class",
    "method_definition": "method",
    "interface_declaration": "interface",
    "type_alias_declaration": "type",
    "enum_declaration": "enum",
    "method_declaration": "method",
    "constructor_declaration": "constructor",
    "function_statement": "function",
    "function_item": "function",
    "impl_item": "implementation",
    "trait_item": "trait",
    "struct_item": "struct",
    "enum_item": "enum",
    "function_declaration": "function",
    "procedure_declaration": "procedure",
}


class CodeServiceError(ValueError):
    pass


class ConfirmationRequired(CodeServiceError):
    pass


def now_iso() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def resolved_serena_command() -> tuple[str, ...]:
    if not SERENA_COMMAND:
        return SERENA_COMMAND
    executable = Path(SERENA_COMMAND[0])
    if not executable.is_absolute():
        executable = ROOT / executable
    return (str(executable), *SERENA_COMMAND[1:])


def connect(path: str | Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    connection = sqlite3.connect(Path(path))
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.enable_load_extension(True)
    sqlite_vec.load(connection)
    connection.enable_load_extension(False)
    connection.executescript(SCHEMA)
    connection.execute(
        f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_code_chunks USING vec0(embedding float[{DIMENSIONS}])"
    )
    return connection


@lru_cache(maxsize=1)
def embedding_model() -> TextEmbedding:
    cache_dir = ROOT / VECTOR.get("cache_dir", "models")
    cache_dir.mkdir(parents=True, exist_ok=True)
    return TextEmbedding(model_name=MODEL_NAME, cache_dir=str(cache_dir))


def default_embeddings(texts: list[str]) -> list[Any]:
    return list(embedding_model().embed(texts, batch_size=32))


def token_estimate(text: str) -> int:
    return max(1, math.ceil(len(text) / 3.2))


def iter_code_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
            continue
        try:
            relative = path.relative_to(root)
        except ValueError:
            continue
        if any(part.lower() in EXCLUDED_DIRS for part in relative.parts[:-1]):
            continue
        try:
            if path.stat().st_size <= MAX_FILE_BYTES:
                yield path
        except OSError:
            continue


def manifest(root: Path) -> tuple[str, list[Path]]:
    files = sorted(iter_code_files(root), key=lambda item: item.as_posix().lower())
    digest = hashlib.sha256()
    for path in files:
        stat = path.stat()
        relative = path.relative_to(root).as_posix()
        digest.update(f"{relative}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode("utf-8"))
    return digest.hexdigest(), files


def project_status(connection: sqlite3.Connection, name: str) -> dict[str, Any]:
    row = connection.execute(
        "SELECT * FROM repositories WHERE name = ? COLLATE NOCASE", (name.strip(),)
    ).fetchone()
    if row is None:
        return {
            "project": name,
            "state": "unregistered",
            "provider": CODE_PROVIDER,
            "should_offer_index": True,
            "should_offer_reindex": False,
            "message": (
                "No Serena project is registered. Offer to prepare it; do not build it silently."
                if CODE_PROVIDER == "serena"
                else "No code index is registered. Offer to create one; do not build it silently."
            ),
        }
    root = Path(row["root_path"])
    base = {
        "project": row["name"],
        "root_path": row["root_path"],
        "registered_at": row["registered_at"],
        "indexed_at": row["indexed_at"],
        "provider": CODE_PROVIDER,
        "should_offer_index": False,
        "should_offer_reindex": False,
    }
    if not root.is_dir():
        return {
            **base,
            "state": "missing_root",
            "message": "Registered project path is unavailable.",
        }
    if CODE_PROVIDER == "serena":
        if not command_available(resolved_serena_command()):
            return {
                **base,
                "state": "unavailable",
                "message": (
                    "Serena runtime is not available. Install it or configure code.serena.command; "
                    "the legacy SQLite index remains available for rollback."
                ),
            }
        serena_project = root / ".serena" / "project.yml"
        if not serena_project.is_file():
            return {
                **base,
                "state": "needs_index",
                "should_offer_reindex": True,
                "message": "Project is registered but Serena has not prepared its symbol cache.",
            }
        return {
            **base,
            "state": "ready",
            # Serena owns the active cache and does not expose a cheap, stable
            # SQLite-style file/symbol count through this adapter. Do not
            # present the legacy manifest count as if it were Serena's index.
            "files": None,
            "symbols": None,
            "message": "Serena project is ready for semantic code retrieval.",
        }
    if not row["indexed_at"]:
        return {
            **base,
            "state": "needs_index",
            "should_offer_reindex": True,
            "message": "Project is registered but has not been indexed.",
        }
    current_manifest, files = manifest(root)
    if current_manifest != row["manifest_hash"]:
        return {
            **base,
            "state": "stale",
            "files": len(files),
            "should_offer_reindex": True,
            "message": "Source files changed after the last index build.",
        }
    counts = connection.execute(
        """
        SELECT COUNT(DISTINCT f.id) AS files, COUNT(c.id) AS symbols
        FROM code_files f
        LEFT JOIN code_chunks c ON c.file_id = f.id
        WHERE f.repository_id = ?
        """,
        (row["id"],),
    ).fetchone()
    return {
        **base,
        "state": "ready",
        "files": counts["files"],
        "symbols": counts["symbols"],
        "message": "Code index is current and ready for selective retrieval.",
    }


def register_project(
    connection: sqlite3.Connection,
    name: str,
    root_path: str | Path,
    *,
    confirmed: bool,
) -> dict[str, Any]:
    if not confirmed:
        raise ConfirmationRequired("Project registration requires explicit confirmation")
    normalized_name = re.sub(r"\s+", " ", name).strip()
    if not normalized_name or len(normalized_name) > 120:
        raise CodeServiceError("project name must contain 1 to 120 characters")
    root = Path(root_path).resolve()
    if not root.is_dir():
        raise CodeServiceError("project path is not an existing directory")
    created_at = now_iso()
    with connection:
        connection.execute(
            """
            INSERT INTO repositories (name, root_path, registered_at)
            VALUES (?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET root_path = excluded.root_path,
                manifest_hash = NULL, indexed_at = NULL
            """,
            (normalized_name, str(root), created_at),
        )
    return project_status(connection, normalized_name)


def _node_text(source: bytes, node: Any) -> str:
    return source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _symbol_name(source: bytes, node: Any) -> str:
    name_node = node.child_by_field_name("name")
    if name_node is not None:
        return _node_text(source, name_node).strip()
    for child in node.named_children:
        if child.type in {"identifier", "type_identifier", "property_identifier", "name"}:
            return _node_text(source, child).strip()
    return node.type


def extract_symbols(path: Path, source: bytes, language: str) -> list[dict[str, Any]]:
    try:
        parser = get_parser(language)
        tree = parser.parse(source)
    except Exception as error:
        raise CodeServiceError(f"Tree-sitter could not parse {path}: {error}") from error
    symbols: list[dict[str, Any]] = []
    stack = [tree.root_node]
    while stack:
        node = stack.pop()
        kind = SYMBOL_KINDS.get(node.type)
        if kind:
            content = _node_text(source, node).strip()
            if content:
                symbols.append(
                    {
                        "symbol": _symbol_name(source, node),
                        "symbol_kind": kind,
                        "start_line": node.start_point.row + 1,
                        "end_line": node.end_point.row + 1,
                        "content": content,
                    }
                )
        stack.extend(reversed(node.named_children))
    if not symbols:
        text = source.decode("utf-8", errors="replace").strip()
        if text:
            symbols.append(
                {
                    "symbol": path.name,
                    "symbol_kind": "file",
                    "start_line": 1,
                    "end_line": text.count("\n") + 1,
                    "content": text,
                }
            )
    return symbols


def split_identifier(value: str) -> str:
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", value)
    return re.sub(r"[_\-.]+", " ", value)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def build_project(
    connection: sqlite3.Connection,
    name: str,
    *,
    confirmed: bool,
    embedding_function: Callable[[list[str]], list[Any]] = default_embeddings,
) -> dict[str, Any]:
    if not confirmed:
        raise ConfirmationRequired("Code indexing requires explicit confirmation")
    repository = connection.execute(
        "SELECT * FROM repositories WHERE name = ? COLLATE NOCASE", (name,)
    ).fetchone()
    if repository is None:
        raise CodeServiceError("project is not registered")
    if CODE_PROVIDER == "serena":
        return build_serena_project(connection, repository)
    root = Path(repository["root_path"])
    if not root.is_dir():
        raise CodeServiceError("registered project path is unavailable")
    manifest_hash, files = manifest(root)

    cached_vectors: dict[str, bytes] = {}
    existing = connection.execute(
        """
        SELECT c.content_hash, v.embedding
        FROM code_chunks c
        JOIN code_files f ON f.id = c.file_id
        JOIN vec_code_chunks v ON v.rowid = c.id
        WHERE f.repository_id = ?
        """,
        (repository["id"],),
    ).fetchall()
    for row in existing:
        cached_vectors[row["content_hash"]] = row["embedding"]

    prepared: list[dict[str, Any]] = []
    indexed_at = now_iso()
    for path in files:
        source = path.read_bytes()
        try:
            language = str(detect_language_from_path(str(path)))
        except Exception:
            language = path.suffix.lower().lstrip(".") or "text"
        symbols = extract_symbols(path, source, language)
        prepared.append(
            {
                "path": path,
                "relative": path.relative_to(root).as_posix(),
                "language": language,
                "sha256": _sha256_bytes(source),
                "size_bytes": len(source),
                "modified_ns": path.stat().st_mtime_ns,
                "symbols": symbols,
            }
        )

    missing: dict[str, str] = {}
    for file_item in prepared:
        for symbol in file_item["symbols"]:
            content_hash = _sha256_bytes(
                f"{file_item['relative']}\n{symbol['symbol']}\n{symbol['content']}".encode("utf-8")
            )
            symbol["content_hash"] = content_hash
            if content_hash not in cached_vectors:
                missing[content_hash] = (
                    f"Path: {file_item['relative']}\n"
                    f"Symbol: {split_identifier(symbol['symbol'])}\n"
                    f"{symbol['content']}"
                )
    vectors = embedding_function(list(missing.values())) if missing else []
    if vectors and len(vectors[0]) != DIMENSIONS:
        raise CodeServiceError(
            f"embedding dimension {len(vectors[0])} does not match {DIMENSIONS}"
        )
    for content_hash, vector in zip(missing, vectors):
        cached_vectors[content_hash] = sqlite_vec.serialize_float32(vector)

    old_chunk_ids = [
        row[0]
        for row in connection.execute(
            """
            SELECT c.id FROM code_chunks c
            JOIN code_files f ON f.id = c.file_id
            WHERE f.repository_id = ?
            """,
            (repository["id"],),
        ).fetchall()
    ]
    with connection:
        for chunk_id in old_chunk_ids:
            connection.execute("DELETE FROM code_chunks_fts WHERE rowid = ?", (chunk_id,))
            connection.execute("DELETE FROM vec_code_chunks WHERE rowid = ?", (chunk_id,))
        connection.execute(
            "DELETE FROM code_files WHERE repository_id = ?", (repository["id"],)
        )
        symbol_total = 0
        for file_item in prepared:
            cursor = connection.execute(
                """
                INSERT INTO code_files (
                    repository_id, path, language, sha256, size_bytes,
                    modified_ns, indexed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    repository["id"],
                    file_item["relative"],
                    file_item["language"],
                    file_item["sha256"],
                    file_item["size_bytes"],
                    file_item["modified_ns"],
                    indexed_at,
                ),
            )
            file_id = cursor.lastrowid
            for index, symbol in enumerate(file_item["symbols"]):
                cursor = connection.execute(
                    """
                    INSERT INTO code_chunks (
                        file_id, chunk_index, symbol, symbol_kind, start_line,
                        end_line, content, content_hash, token_estimate
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        file_id,
                        index,
                        symbol["symbol"],
                        symbol["symbol_kind"],
                        symbol["start_line"],
                        symbol["end_line"],
                        symbol["content"],
                        symbol["content_hash"],
                        token_estimate(symbol["content"]),
                    ),
                )
                chunk_id = cursor.lastrowid
                connection.execute(
                    "INSERT INTO code_chunks_fts(rowid, path, symbol, content) VALUES (?, ?, ?, ?)",
                    (
                        chunk_id,
                        split_identifier(file_item["relative"]),
                        split_identifier(symbol["symbol"]),
                        symbol["content"],
                    ),
                )
                connection.execute(
                    "INSERT INTO vec_code_chunks(rowid, embedding) VALUES (?, ?)",
                    (chunk_id, cached_vectors[symbol["content_hash"]]),
                )
                symbol_total += 1
        connection.execute(
            """
            UPDATE repositories
            SET manifest_hash = ?, indexed_at = ?
            WHERE id = ?
            """,
            (manifest_hash, indexed_at, repository["id"]),
        )
        connection.execute(
            "INSERT OR REPLACE INTO code_meta(key, value) VALUES ('vector_model', ?)",
            (MODEL_NAME,),
        )
    return {
        "project": repository["name"],
        "state": "ready",
        "files": len(prepared),
        "symbols": symbol_total,
        "embeddings_computed": len(missing),
        "embeddings_reused": symbol_total - len(missing),
        "indexed_at": indexed_at,
    }


def build_serena_project(connection: sqlite3.Connection, repository: sqlite3.Row) -> dict[str, Any]:
    """Explicitly prepare Serena's project configuration and symbol cache."""
    root = Path(repository["root_path"])
    if not root.is_dir():
        raise CodeServiceError("registered project path is unavailable")
    serena_command = resolved_serena_command()
    if not command_available(serena_command):
        raise CodeServiceError(
            "Serena runtime is unavailable. Install it or configure code.serena.command."
        )
    manifest_hash, files = manifest(root)
    try:
        project_file = root / ".serena" / "project.yml"
        if project_file.is_file():
            serena_args = ["project", "index", str(root)]
        else:
            detected_languages: list[str] = []
            if any(path.suffix.lower() == ".py" for path in files):
                detected_languages.append("python")
            if any(path.suffix.lower() in {".js", ".jsx", ".ts", ".tsx"} for path in files):
                detected_languages.append("typescript")
            if not detected_languages:
                detected_languages.append("python")
            serena_args = ["project", "create", str(root), "--name", str(repository["name"]), "--index"]
            for language in detected_languages:
                serena_args.extend(["--language", language])
            serena_args.extend(["--log-level", "ERROR", "--timeout", "10"])
        completed = subprocess.run(
            [*serena_command, *serena_args],
            cwd=str(root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=SERENA_TIMEOUT_SECONDS * 2,
            check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            env={
                **os.environ,
                "PATH": str(Path(serena_command[0]).parent) + os.pathsep + os.environ.get("PATH", ""),
            },
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise CodeServiceError(f"Serena project preparation failed: {error}") from error
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "unknown Serena error").strip()
        raise CodeServiceError(f"Serena project preparation failed: {detail[-2000:]}")
    indexed_at = now_iso()
    # Keep the registry as a small compatibility catalog. Serena owns the actual cache.
    with connection:
        connection.execute(
            "UPDATE repositories SET manifest_hash = ?, indexed_at = ? WHERE id = ?",
            (manifest_hash, indexed_at, repository["id"]),
        )
    return {
        "project": repository["name"],
        "provider": "serena",
        "state": "ready",
        "files": None,
        "symbols": None,
        "source_files": len(files),
        "indexed_at": indexed_at,
        "serena_project": str(root / ".serena" / "project.yml"),
        "output": (completed.stdout or "").strip()[-2000:],
    }


def fts_query(value: str) -> str:
    terms = re.findall(r"[\wА-Яа-яЁё-]+", split_identifier(value), flags=re.UNICODE)
    stop_words = {
        "a", "an", "and", "is", "of", "the", "to", "where", "with",
        "а", "в", "где", "для", "и", "как", "на", "по", "это",
    }
    terms = [term for term in terms if term.lower() not in stop_words]
    return " OR ".join(f'"{term}"' for term in terms)


def _search_rows(
    connection: sqlite3.Connection,
    repository_id: int,
    query: str,
    limit: int,
    embedding_function: Callable[[list[str]], list[Any]],
) -> list[dict[str, Any]]:
    candidate_limit = max(10, limit * 4)
    rankings: dict[int, float] = {}
    rows_by_id: dict[int, sqlite3.Row] = {}
    match = fts_query(query)
    lexical = []
    if match:
        lexical = connection.execute(
            """
            SELECT c.*, f.path, f.language, bm25(code_chunks_fts) AS distance
            FROM code_chunks_fts
            JOIN code_chunks c ON c.id = code_chunks_fts.rowid
            JOIN code_files f ON f.id = c.file_id
            WHERE code_chunks_fts MATCH ? AND f.repository_id = ?
            ORDER BY distance LIMIT ?
            """,
            (match, repository_id, candidate_limit),
        ).fetchall()
    query_vector = embedding_function([query])[0]
    vector_candidates = connection.execute(
        """
        SELECT c.*, f.path, f.language, v.distance
        FROM vec_code_chunks v
        JOIN code_chunks c ON c.id = v.rowid
        JOIN code_files f ON f.id = c.file_id
        WHERE v.embedding MATCH ? AND k = ?
        """,
        (sqlite_vec.serialize_float32(query_vector), candidate_limit * 3),
    ).fetchall()
    vector = [row for row in vector_candidates if connection.execute(
        "SELECT repository_id FROM code_files WHERE id = ?", (row["file_id"],)
    ).fetchone()[0] == repository_id][:candidate_limit]
    for rank, row in enumerate(lexical, start=1):
        rows_by_id[row["id"]] = row
        rankings[row["id"]] = rankings.get(row["id"], 0.0) + 1.0 / (60 + rank)
    for rank, row in enumerate(vector, start=1):
        rows_by_id[row["id"]] = row
        rankings[row["id"]] = rankings.get(row["id"], 0.0) + 1.0 / (60 + rank)
    ordered = sorted(rankings, key=rankings.get, reverse=True)[:limit]
    return [
        {
            "path": rows_by_id[item]["path"],
            "language": rows_by_id[item]["language"],
            "symbol": rows_by_id[item]["symbol"],
            "symbol_kind": rows_by_id[item]["symbol_kind"],
            "start_line": rows_by_id[item]["start_line"],
            "end_line": rows_by_id[item]["end_line"],
            "content": rows_by_id[item]["content"],
            "token_estimate": rows_by_id[item]["token_estimate"],
            "score": rankings[item],
        }
        for item in ordered
    ]


def code_context(
    connection: sqlite3.Connection,
    project: str,
    query: str,
    *,
    max_tokens: int = 5000,
    limit: int = 16,
    embedding_function: Callable[[list[str]], list[Any]] = default_embeddings,
) -> dict[str, Any]:
    status = project_status(connection, project)
    if status["state"] != "ready":
        return {**status, "query": query, "estimated_tokens": 0, "items": []}
    if CODE_PROVIDER == "serena":
        root = Path(status["root_path"])
        try:
            return SerenaProvider(
                command=resolved_serena_command(),
                timeout=SERENA_TIMEOUT_SECONDS,
            ).context(root, query, max_tokens=max_tokens, limit=limit) | {
                "project": status["project"],
                "root_path": status["root_path"],
                "provider": "serena",
                "state": "ready",
                "indexed_at": status.get("indexed_at"),
                "message": status["message"],
            }
        except SerenaProviderError as error:
            return {
                **status,
                "state": "serena_error",
                "message": "Serena could not produce code context; inspect the Serena diagnostics.",
                "error": str(error),
                "query": query,
                "estimated_tokens": 0,
                "items": [],
            }
    repository = connection.execute(
        "SELECT id FROM repositories WHERE name = ? COLLATE NOCASE", (project,)
    ).fetchone()
    candidates = _search_rows(
        connection, repository["id"], query, limit, embedding_function
    )
    selected: list[dict[str, Any]] = []
    used = 0
    seen: set[tuple[str, int, int]] = set()
    for row in candidates:
        key = (row["path"], row["start_line"], row["end_line"])
        if key in seen or used + row["token_estimate"] > max_tokens:
            continue
        seen.add(key)
        selected.append(row)
        used += row["token_estimate"]
    return {
        **status,
        "query": query,
        "method": "Tree-sitter symbols + FTS5 + sqlite-vec + reciprocal-rank fusion",
        "max_tokens": max_tokens,
        "estimated_tokens": used,
        "items": selected,
    }


def command_status(args: argparse.Namespace) -> int:
    connection = connect(args.database)
    print(json.dumps(project_status(connection, args.project), ensure_ascii=False, indent=2))
    connection.close()
    return 0


def command_register(args: argparse.Namespace) -> int:
    connection = connect(args.database)
    result = register_project(
        connection, args.project, args.path, confirmed=args.confirm
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    connection.close()
    return 0


def command_build(args: argparse.Namespace) -> int:
    connection = connect(args.database)
    result = build_project(connection, args.project, confirmed=args.confirm)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    connection.close()
    return 0


def command_context(args: argparse.Namespace) -> int:
    connection = connect(args.database)
    result = code_context(
        connection,
        args.project,
        args.query,
        max_tokens=args.budget,
        limit=args.limit,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    connection.close()
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Selective local code retrieval")
    result.add_argument("--database", type=Path, default=DEFAULT_DB_PATH)
    sub = result.add_subparsers(dest="command", required=True)

    status = sub.add_parser("status", help="Check whether a project index is ready")
    status.add_argument("--project", required=True)
    status.set_defaults(func=command_status)

    register = sub.add_parser("register", help="Register a project after confirmation")
    register.add_argument("--project", required=True)
    register.add_argument("--path", required=True)
    register.add_argument("--confirm", action="store_true")
    register.set_defaults(func=command_register)

    build = sub.add_parser("build", help="Build a registered project index")
    build.add_argument("--project", required=True)
    build.add_argument("--confirm", action="store_true")
    build.set_defaults(func=command_build)

    context = sub.add_parser("context", help="Build bounded code context")
    context.add_argument("--project", required=True)
    context.add_argument("--query", required=True)
    context.add_argument("--budget", type=int, default=5000)
    context.add_argument("--limit", type=int, default=16)
    context.set_defaults(func=command_context)
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        return args.func(args)
    except CodeServiceError as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
