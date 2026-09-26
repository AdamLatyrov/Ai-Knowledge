from __future__ import annotations

import argparse
import datetime as dt
import glob
import hashlib
import json
import os
import re
import sqlite3
import sys
from pathlib import Path

import sqlite_vec

from memory_store import (
    ensure_schema,
    external_projection_documents,
    fact_documents,
    handoff_documents,
    knowledge_item_documents,
)
from java_prep import preparation_documents


ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.json"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,
    collection TEXT NOT NULL,
    scope TEXT,
    doc_type TEXT NOT NULL,
    mode TEXT NOT NULL,
    title TEXT NOT NULL,
    modified_at TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    sha256 TEXT NOT NULL,
    content TEXT NOT NULL DEFAULT '',
    redaction_count INTEGER NOT NULL DEFAULT 0,
    indexed_at TEXT NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
    path UNINDEXED,
    title,
    content,
    tokenize='unicode61 remove_diacritics 2'
);
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,
    source TEXT NOT NULL,
    session_id TEXT,
    started_at TEXT,
    cwd TEXT,
    size_bytes INTEGER NOT NULL,
    modified_at TEXT NOT NULL,
    archived INTEGER NOT NULL DEFAULT 0
);
"""


SECRET_PATTERNS = [
    re.compile(
        r"(?im)(password|passwd|пароль|token|api[ _-]?key|secret)"
        r"(\s*[:=]\s*)([^\s,;`]+)"
    ),
    re.compile(
        r"(?is)-----BEGIN [A-Z ]*PRIVATE KEY-----.*?"
        r"-----END [A-Z ]*PRIVATE KEY-----"
    ),
    re.compile(r"(?i)(https?://[^\s:/]+:)([^@\s]+)(@)"),
]


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def resolve_root(value: str) -> Path:
    expanded = os.path.expandvars(value.replace("%USERPROFILE%", str(Path.home())))
    path = Path(expanded)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def database_path(config: dict) -> Path:
    override = os.environ.get("AI_KNOWLEDGE_DATABASE")
    return Path(override).resolve() if override else (ROOT / config["database"]).resolve()


def connect(config: dict) -> sqlite3.Connection:
    connection = sqlite3.connect(database_path(config))
    connection.row_factory = sqlite3.Row
    connection.enable_load_extension(True)
    sqlite_vec.load(connection)
    connection.enable_load_extension(False)
    connection.executescript(SCHEMA)
    ensure_schema(connection)
    return connection


def read_text(path: Path) -> str:
    data = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "cp1251"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def read_docx(path: Path) -> str:
    """Extract plain text from .docx (paragraphs and table cells)."""
    import xml.etree.ElementTree as ET
    import zipfile

    w = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    with zipfile.ZipFile(path) as archive:
        xml_data = archive.read("word/document.xml")
    root = ET.fromstring(xml_data)
    body = root.find(f"{w}body")
    lines = []
    if body is not None:
        for paragraph in body.iter(f"{w}p"):
            text = "".join(node.text or "" for node in paragraph.iter(f"{w}t"))
            if text.strip():
                lines.append(text.strip())
    return "\n".join(lines)


def redact(text: str) -> tuple[str, int]:
    count = 0

    def replace_key_value(match: re.Match) -> str:
        nonlocal count
        count += 1
        if len(match.groups()) >= 3:
            return f"{match.group(1)}{match.group(2)}[REDACTED]"
        return "[REDACTED PRIVATE KEY]"

    for index, pattern in enumerate(SECRET_PATTERNS):
        if index == 1:
            text, hits = pattern.subn("[REDACTED PRIVATE KEY]", text)
            count += hits
        elif index == 2:
            def replace_url(match: re.Match) -> str:
                nonlocal count
                count += 1
                return f"{match.group(1)}[REDACTED]{match.group(3)}"
            text = pattern.sub(replace_url, text)
        else:
            text = pattern.sub(replace_key_value, text)
    return text, count


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def excluded(path: Path, fragments: list[str]) -> bool:
    normalized = str(path).replace("/", "\\").lower()
    return any(fragment.lower() in normalized for fragment in fragments)


def classify(
    path: Path,
    collection_root: Path,
    scope_override: str | None = None,
    doc_type_override: str | None = None,
) -> tuple[str, str]:
    name = path.name.lower()
    if name == "agents.md":
        doc_type = "rules"
    elif name == "project_memory.md":
        doc_type = "project_current"
    elif name == "project_history.md":
        doc_type = "project_history"
    elif name == "memory_index.md":
        doc_type = "memory_index"
    elif name == "daily_plan.md":
        doc_type = "daily_plan"
    elif name == "current-state.md":
        doc_type = "project_current"
    elif "handoff" in name:
        doc_type = "handoff"
    elif "report" in name or "аудит" in name:
        doc_type = "report"
    else:
        doc_type = "note"

    if doc_type_override:
        doc_type = doc_type_override

    try:
        relative = path.relative_to(collection_root)
    except ValueError:
        relative = path
    parts = list(relative.parts)
    scope = scope_override or "global"
    if scope_override:
        return doc_type, scope
    if "_agent" in parts:
        marker = parts.index("_agent")
        if marker > 0:
            scope = "/".join(parts[:marker])
    elif len(parts) > 1:
        scope = parts[0]
    return doc_type, scope


def title_for(path: Path, text: str) -> str:
    match = re.search(r"(?m)^#\s+(.+?)\s*$", text)
    return match.group(1).strip() if match else path.stem


def iter_collection_files(item: dict, fragments: list[str]):
    root = resolve_root(item["root"])
    seen: set[Path] = set()
    for pattern in item["patterns"]:
        for raw in glob.glob(str(root / pattern), recursive=True):
            path = Path(raw).resolve()
            if path.is_file() and path not in seen and not excluded(path, fragments):
                seen.add(path)
                yield root, path


def sync_documents(connection: sqlite3.Connection, config: dict) -> dict:
    now = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    max_bytes = int(config.get("max_content_bytes", 500000))
    fragments = config.get("exclude_path_fragments", [])
    counts = {
        "content": 0,
        "metadata": 0,
        "database": 0,
        "redactions": 0,
        "skipped_large": 0,
    }
    tables = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
        ).fetchall()
    }
    if "vec_chunks" in tables:
        connection.execute("DELETE FROM vec_chunks")
    if "chunks_fts" in tables:
        connection.execute("DELETE FROM chunks_fts")
    if "memory_chunks" in tables:
        connection.execute("DELETE FROM memory_chunks")
    connection.execute("DELETE FROM documents_fts")
    connection.execute("DELETE FROM documents")

    for item in config["collections"]:
        mode = item["mode"]
        collection_max = int(item.get("max_content_bytes", max_bytes))
        for collection_root, path in iter_collection_files(item, fragments):
            stat = path.stat()
            content = ""
            redactions = 0
            effective_mode = mode
            if mode == "content" and stat.st_size <= collection_max:
                if path.suffix.lower() == ".docx":
                    content, redactions = redact(read_docx(path))
                else:
                    content, redactions = redact(read_text(path))
                counts["content"] += 1
                counts["redactions"] += redactions
            elif mode == "content":
                effective_mode = "metadata"
                counts["metadata"] += 1
                counts["skipped_large"] += 1
            else:
                counts["metadata"] += 1

            doc_type, scope = classify(
                path,
                collection_root,
                scope_override=item.get("scope"),
                doc_type_override=item.get("doc_type"),
            )
            title = title_for(path, content)
            cursor = connection.execute(
                """
                INSERT INTO documents (
                    path, collection, scope, doc_type, mode, title, modified_at,
                    size_bytes, sha256, content, redaction_count, indexed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(path), item["name"], scope, doc_type, effective_mode,
                    title, dt.datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(timespec="seconds"),
                    stat.st_size, sha256(path), content, redactions, now,
                ),
            )
            if content:
                connection.execute(
                    "INSERT INTO documents_fts(rowid, path, title, content) VALUES (?, ?, ?, ?)",
                    (cursor.lastrowid, str(path), title, content),
                )

    database_documents = (
        fact_documents(connection)
        + handoff_documents(connection)
        + external_projection_documents(connection)
        + knowledge_item_documents(connection)
        + preparation_documents(connection)
    )
    for document in database_documents:
        content = document["content"]
        encoded = content.encode("utf-8")
        cursor = connection.execute(
            """
            INSERT INTO documents (
                path, collection, scope, doc_type, mode, title, modified_at,
                size_bytes, sha256, content, redaction_count, indexed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
            """,
            (
                document["path"],
                document["collection"],
                document["scope"],
                document["doc_type"],
                document["mode"],
                document["title"],
                document["modified_at"],
                len(encoded),
                hashlib.sha256(encoded).hexdigest(),
                content,
                now,
            ),
        )
        connection.execute(
            "INSERT INTO documents_fts(rowid, path, title, content) VALUES (?, ?, ?, ?)",
            (cursor.lastrowid, document["path"], document["title"], content),
        )
        counts["database"] += 1
    return counts


def session_metadata(path: Path) -> tuple[str | None, str | None, str | None]:
    try:
        with path.open("r", encoding="utf-8") as source:
            for _ in range(20):
                line = source.readline()
                if not line:
                    break
                item = json.loads(line)
                if item.get("type") == "session_meta":
                    payload = item.get("payload", {})
                    return (
                        payload.get("session_id") or payload.get("id"),
                        payload.get("timestamp") or item.get("timestamp"),
                        payload.get("cwd"),
                    )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        pass
    return None, None, None


def sync_sessions(connection: sqlite3.Connection, config: dict) -> int:
    connection.execute("DELETE FROM sessions")
    count = 0
    for item in config.get("session_collections", []):
        root = resolve_root(item["root"])
        for pattern in item["patterns"]:
            for raw in glob.glob(str(root / pattern), recursive=True):
                path = Path(raw).resolve()
                if not path.is_file():
                    continue
                stat = path.stat()
                session_id, started_at, cwd = session_metadata(path)
                connection.execute(
                    """
                    INSERT OR REPLACE INTO sessions (
                        path, source, session_id, started_at, cwd, size_bytes,
                        modified_at, archived
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(path), item["name"], session_id, started_at, cwd,
                        stat.st_size,
                        dt.datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(timespec="seconds"),
                        1 if item.get("archived") else 0,
                    ),
                )
                count += 1
    return count


def command_sync(_: argparse.Namespace) -> int:
    config = load_config()
    connection = connect(config)
    with connection:
        counts = sync_documents(connection, config)
        sessions = sync_sessions(connection, config)
        connection.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES ('last_sync', ?)",
            (dt.datetime.now().astimezone().isoformat(timespec="seconds"),),
        )
    document_total = connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    connection.close()
    print(f"Documents: {document_total}")
    print(f"Content indexed: {counts['content']}")
    print(f"Database records indexed: {counts['database']}")
    print(f"Metadata only: {counts['metadata']}")
    print(f"Large files skipped: {counts['skipped_large']}")
    print(f"Redactions applied: {counts['redactions']}")
    print(f"Codex sessions catalogued: {sessions}")
    return 0


def fts_query(value: str) -> str:
    terms = re.findall(r"[\wА-Яа-яЁё-]+", value, flags=re.UNICODE)
    return " AND ".join(f'"{term.replace(chr(34), chr(34) * 2)}"' for term in terms)


def search_rows(connection: sqlite3.Connection, query: str, limit: int, project: str | None = None):
    match = fts_query(query)
    if not match:
        return []
    sql = """
        SELECT d.path, d.title, d.scope, d.doc_type, d.modified_at,
               snippet(documents_fts, 2, '[', ']', ' … ', 24) AS snippet,
               bm25(documents_fts) AS rank
        FROM documents_fts
        JOIN documents d ON d.id = documents_fts.rowid
        WHERE documents_fts MATCH ?
    """
    params: list[object] = [match]
    if project:
        sql += " AND lower(d.scope) LIKE ?"
        params.append(f"%{project.lower()}%")
    sql += " ORDER BY rank LIMIT ?"
    params.append(limit)
    return connection.execute(sql, params).fetchall()


def command_search(args: argparse.Namespace) -> int:
    config = load_config()
    connection = connect(config)
    rows = search_rows(connection, args.query, args.limit, args.project)
    for index, row in enumerate(rows, start=1):
        print(f"{index}. {row['title']} [{row['scope']}] ({row['doc_type']})")
        print(f"   {row['path']}")
        print(f"   {row['snippet'].replace(chr(10), ' ')}")
    if not rows:
        print("No results.")
    connection.close()
    return 0


def read_local(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8").strip()


def command_context(args: argparse.Namespace) -> int:
    config = load_config()
    connection = connect(config)
    output = Path(args.output).resolve() if args.output else None
    sections = [
        "# Пакет контекста для AI-агента",
        "",
        f"Generated: {dt.datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"Project: {args.project or 'global'}",
        f"Task/query: {args.query or 'general orientation'}",
        "",
        "Этот пакет — краткий вход. При необходимости используйте поиск по базе, а не загружайте всю историю.",
        "",
        read_local("profile.md"),
        "",
        read_local("workspace.md"),
        "",
        read_local("policies.md"),
    ]

    if args.project:
        rows = connection.execute(
            """
            SELECT title, path, content FROM documents
            WHERE doc_type = 'project_current' AND lower(scope) LIKE ? AND content <> ''
            ORDER BY modified_at DESC LIMIT 2
            """,
            (f"%{args.project.lower()}%",),
        ).fetchall()
        for row in rows:
            sections.extend([
                "",
                f"## Канонический статус: {row['title']}",
                "",
                f"Source: `{row['path']}`",
                "",
                row["content"][:12000],
            ])

    if args.query:
        rows = search_rows(connection, args.query, args.limit, args.project)
        sections.extend(["", "## Релевантные находки", ""])
        for row in rows:
            sections.extend([
                f"### {row['title']}",
                f"Source: `{row['path']}`",
                row["snippet"].replace("\n", " "),
                "",
            ])

    rendered = "\n".join(sections).rstrip() + "\n"
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
        print(output)
    else:
        print(rendered, end="")
    connection.close()
    return 0


def command_status(_: argparse.Namespace) -> int:
    config = load_config()
    path = database_path(config)
    connection = connect(config)
    documents = connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    content = connection.execute("SELECT COUNT(*) FROM documents WHERE content <> ''").fetchone()[0]
    sessions = connection.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
    redactions = connection.execute("SELECT COALESCE(SUM(redaction_count), 0) FROM documents").fetchone()[0]
    last_sync = connection.execute("SELECT value FROM meta WHERE key='last_sync'").fetchone()
    print(f"Database: {path}")
    print(f"Size: {(path.stat().st_size / 1024 / 1024) if path.exists() else 0:.2f} MB")
    print(f"Documents: {documents} ({content} with searchable content)")
    print(f"Codex sessions: {sessions} (metadata only)")
    print(f"Redactions: {redactions}")
    print(f"Last sync: {last_sync[0] if last_sync else 'never'}")
    connection.close()
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Local AI knowledge index")
    sub = result.add_subparsers(dest="command", required=True)

    sync = sub.add_parser("sync", help="Rebuild document and session indexes")
    sync.set_defaults(func=command_sync)

    search = sub.add_parser("search", help="Search indexed knowledge")
    search.add_argument("query")
    search.add_argument("--project")
    search.add_argument("--limit", type=int, default=8)
    search.set_defaults(func=command_search)

    context = sub.add_parser("context", help="Build a compact agent context packet")
    context.add_argument("--project")
    context.add_argument("--query")
    context.add_argument("--limit", type=int, default=6)
    context.add_argument("--output")
    context.set_defaults(func=command_context)

    status = sub.add_parser("status", help="Show index status")
    status.set_defaults(func=command_status)
    return result


def main() -> int:
    args = parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
