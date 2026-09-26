from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import html
import json
import os
import re
import sqlite3
from pathlib import Path
from typing import Any, Callable, Iterable


ROOT = Path(__file__).resolve().parent.parent
CONFIG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
VECTOR = CONFIG["vector"]
DEFAULT_DB_PATH = Path(os.environ.get("AI_KNOWLEDGE_DATABASE", ROOT / CONFIG["database"])).expanduser().resolve()
CORPUS_ROOT = (ROOT / CONFIG.get("corpus", {}).get("root", "corpus_sources")).resolve()
DEFAULT_CHUNK_CHARS = int(VECTOR.get("chunk_chars", 1400))
DEFAULT_OVERLAP_CUES = 1
VECTOR_DIMENSIONS = int(VECTOR["dimensions"])

TIME_RE = re.compile(
    r"(?P<start>\d{1,2}:\d{2}(?::\d{2})?[.,]\d{3})\s+-->\s+"
    r"(?P<end>\d{1,2}:\d{2}(?::\d{2})?[.,]\d{3})"
)
TAG_RE = re.compile(r"<[^>]+>")
WORD_RE = re.compile(r"[\w-]{2,}", re.UNICODE)


SCHEMA = """
CREATE TABLE IF NOT EXISTS corpus_sources (
    source_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    source_type TEXT NOT NULL,
    source_uri TEXT NOT NULL DEFAULT '',
    local_path TEXT NOT NULL DEFAULT '',
    transcript_path TEXT NOT NULL DEFAULT '',
    video_path TEXT NOT NULL DEFAULT '',
    language TEXT NOT NULL DEFAULT '',
    scope TEXT NOT NULL DEFAULT 'ResearchCorpus',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    transcript_sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS corpus_chunks (
    id INTEGER PRIMARY KEY,
    source_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    start_ms INTEGER NOT NULL,
    end_ms INTEGER NOT NULL,
    heading TEXT NOT NULL DEFAULT '',
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    token_estimate INTEGER NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    UNIQUE(source_id, chunk_index),
    FOREIGN KEY(source_id) REFERENCES corpus_sources(source_id)
);
CREATE INDEX IF NOT EXISTS idx_corpus_chunks_source
    ON corpus_chunks(source_id, chunk_index);
CREATE VIRTUAL TABLE IF NOT EXISTS corpus_chunks_fts USING fts5(
    source_id UNINDEXED,
    heading,
    content,
    tokenize='unicode61 remove_diacritics 2'
);
"""


def now_iso() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def allowed_corpus_path(value: str, *, allow_empty: bool = False) -> Path | None:
    raw = str(value or "").strip()
    if not raw and allow_empty:
        return None
    if not raw:
        raise ValueError("corpus path is required")
    path = Path(raw).expanduser().resolve()
    try:
        path.relative_to(CORPUS_ROOT)
    except ValueError as error:
        raise ValueError(f"path must be inside the corpus root: {CORPUS_ROOT}") from error
    return path


def normalize_text(value: str) -> str:
    value = html.unescape(TAG_RE.sub("", str(value or "")))
    value = value.replace("\u200b", "")
    return re.sub(r"\s+", " ", value).strip()


def parse_timestamp(value: str) -> int:
    raw = value.replace(",", ".")
    parts = raw.split(":")
    if len(parts) == 2:
        hours = 0
        minutes, seconds = parts
    elif len(parts) == 3:
        hours, minutes, seconds = parts
    else:
        raise ValueError(f"Invalid WebVTT timestamp: {value}")
    return round((int(hours) * 3600 + int(minutes) * 60 + float(seconds)) * 1000)


def parse_vtt(text: str) -> list[dict[str, Any]]:
    cues: list[dict[str, Any]] = []
    lines = str(text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    index = 0
    while index < len(lines):
        match = TIME_RE.search(lines[index])
        if not match:
            index += 1
            continue
        text_lines: list[str] = []
        index += 1
        while index < len(lines) and not TIME_RE.search(lines[index]):
            line = lines[index].strip()
            if line:
                text_lines.append(line)
            index += 1
        cue_text = normalize_text(" ".join(text_lines))
        if cue_text:
            cues.append(
                {
                    "start_ms": parse_timestamp(match.group("start")),
                    "end_ms": parse_timestamp(match.group("end")),
                    "text": cue_text,
                }
            )
    return deduplicate_cues(cues)


def deduplicate_cues(cues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove rolling/cumulative subtitle overlap from auto-generated VTT cues."""
    result: list[dict[str, Any]] = []
    emitted_words: list[str] = []
    previous_end: int | None = None
    for cue in cues:
        words = normalize_text(cue.get("text", "")).split()
        if not words:
            continue
        overlap = 0
        if previous_end is None or int(cue["start_ms"]) <= previous_end + 1500:
            maximum = min(len(words), len(emitted_words), 80)
            for size in range(maximum, 0, -1):
                if emitted_words[-size:] == words[:size]:
                    overlap = size
                    break
        new_words = words[overlap:]
        if not new_words:
            previous_end = max(previous_end or 0, int(cue["end_ms"]))
            continue
        result.append({**cue, "text": " ".join(new_words)})
        emitted_words = (emitted_words + new_words)[-120:]
        previous_end = int(cue["end_ms"])
    return result


def token_estimate(text: str) -> int:
    ascii_chars = sum(1 for char in text if ord(char) < 128)
    non_ascii_chars = len(text) - ascii_chars
    return max(1, round(ascii_chars / 4 + non_ascii_chars / 2))


def chunk_cues(
    cues: Iterable[dict[str, Any]],
    *,
    max_chars: int = DEFAULT_CHUNK_CHARS,
    overlap_cues: int = DEFAULT_OVERLAP_CUES,
) -> list[dict[str, Any]]:
    materialized = [cue for cue in cues if normalize_text(cue.get("text", ""))]
    chunks: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    current_chars = 0

    def flush(items: list[dict[str, Any]]) -> None:
        if not items:
            return
        text = normalize_text(" ".join(item["text"] for item in items))
        chunks.append(
            {
                "start_ms": int(items[0]["start_ms"]),
                "end_ms": int(items[-1]["end_ms"]),
                "text": text,
            }
        )

    for cue in materialized:
        text = normalize_text(cue["text"])
        if current and current_chars + len(text) + 1 > max_chars:
            flush(current)
            current = current[-overlap_cues:] if overlap_cues else []
            current_chars = sum(len(item["text"]) + 1 for item in current)
        current.append({**cue, "text": text})
        current_chars += len(text) + 1
    flush(current)
    return chunks


def fts_query(query: str) -> str:
    terms = [term for term in WORD_RE.findall(str(query or ""))]
    return " OR ".join(f'"{term.replace(chr(34), "")}"' for term in terms)


class CorpusStore:
    def __init__(
        self,
        connection: sqlite3.Connection,
        *,
        embedder: Callable[[list[str]], Iterable[Any]] | None = None,
        enable_vectors: bool = True,
    ) -> None:
        self.connection = connection
        self.embedder = embedder
        self.enable_vectors = enable_vectors
        self.connection.executescript(SCHEMA)
        if enable_vectors:
            self._ensure_vector_table()

    def _ensure_vector_table(self) -> None:
        try:
            self.connection.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_corpus_chunks "
                f"USING vec0(embedding float[{VECTOR_DIMENSIONS}])"
            )
        except sqlite3.OperationalError as error:
            self.enable_vectors = False
            if "no such module" not in str(error).lower():
                raise

    def _get_embedder(self) -> Callable[[list[str]], Iterable[Any]]:
        if self.embedder:
            return self.embedder
        from memory_service import model

        return lambda texts: model().embed(texts, batch_size=32)

    def ingest_source(self, source: dict[str, Any], vtt_text: str) -> dict[str, Any]:
        source_id = normalize_text(source.get("source_id", ""))
        title = normalize_text(source.get("title", ""))
        if not source_id or not title:
            raise ValueError("source_id and title are required")
        cues = parse_vtt(vtt_text)
        chunks = chunk_cues(cues, max_chars=int(source.get("chunk_chars", DEFAULT_CHUNK_CHARS)))
        transcript_sha256 = hashlib.sha256(vtt_text.encode("utf-8")).hexdigest()
        timestamp = now_iso()

        with self.connection:
            previous = self.connection.execute(
                "SELECT id FROM corpus_chunks WHERE source_id = ?", (source_id,)
            ).fetchall()
            if previous:
                ids = [row[0] for row in previous]
                placeholders = ",".join("?" for _ in ids)
                self.connection.execute(
                    f"DELETE FROM corpus_chunks_fts WHERE rowid IN ({placeholders})", ids
                )
                if self.enable_vectors:
                    self.connection.execute(
                        f"DELETE FROM vec_corpus_chunks WHERE rowid IN ({placeholders})", ids
                    )
            self.connection.execute("DELETE FROM corpus_chunks WHERE source_id = ?", (source_id,))
            self.connection.execute(
                """
                INSERT INTO corpus_sources(
                    source_id, title, source_type, source_uri, local_path,
                    transcript_path, video_path, language, scope, metadata_json,
                    transcript_sha256, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_id) DO UPDATE SET
                    title=excluded.title, source_type=excluded.source_type,
                    source_uri=excluded.source_uri, local_path=excluded.local_path,
                    transcript_path=excluded.transcript_path, video_path=excluded.video_path,
                    language=excluded.language, scope=excluded.scope,
                    metadata_json=excluded.metadata_json,
                    transcript_sha256=excluded.transcript_sha256, updated_at=excluded.updated_at
                """,
                (
                    source_id,
                    title,
                    normalize_text(source.get("source_type", "document")),
                    normalize_text(source.get("source_uri", "")),
                    normalize_text(source.get("local_path", "")),
                    normalize_text(source.get("transcript_path", "")),
                    normalize_text(source.get("video_path", "")),
                    normalize_text(source.get("language", "")),
                    normalize_text(source.get("scope", "ResearchCorpus")) or "ResearchCorpus",
                    json.dumps(source.get("metadata", {}), ensure_ascii=False),
                    transcript_sha256,
                    timestamp,
                    timestamp,
                ),
            )

            vector_payloads: list[str] = []
            inserted: list[tuple[int, str]] = []
            for chunk_index, chunk in enumerate(chunks):
                content_hash = hashlib.sha256(chunk["text"].encode("utf-8")).hexdigest()
                cursor = self.connection.execute(
                    """
                    INSERT INTO corpus_chunks(
                        source_id, chunk_index, start_ms, end_ms, heading,
                        content, content_hash, token_estimate, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        source_id,
                        chunk_index,
                        chunk["start_ms"],
                        chunk["end_ms"],
                        normalize_text(chunk.get("heading", "")),
                        chunk["text"],
                        content_hash,
                        token_estimate(chunk["text"]),
                        json.dumps(chunk.get("metadata", {}), ensure_ascii=False),
                    ),
                )
                chunk_id = int(cursor.lastrowid)
                self.connection.execute(
                    "INSERT INTO corpus_chunks_fts(rowid, source_id, heading, content) VALUES (?, ?, ?, ?)",
                    (chunk_id, source_id, chunk.get("heading", ""), chunk["text"]),
                )
                inserted.append((chunk_id, chunk["text"]))
                vector_payloads.append(f"Source: {title}\n{chunk['text']}")

            if self.enable_vectors and inserted:
                vectors = list(self._get_embedder()(vector_payloads))
                if vectors and len(vectors[0]) != VECTOR_DIMENSIONS:
                    raise RuntimeError("Corpus embedding dimension does not match config")
                import sqlite_vec

                for (chunk_id, _), vector in zip(inserted, vectors):
                    self.connection.execute(
                        "INSERT INTO vec_corpus_chunks(rowid, embedding) VALUES (?, ?)",
                        (chunk_id, sqlite_vec.serialize_float32(vector)),
                    )

        return {
            "source_id": source_id,
            "title": title,
            "chunks": len(chunks),
            "cues": len(cues),
            "transcript_sha256": transcript_sha256,
            "vectors": bool(self.enable_vectors),
        }

    def search(
        self,
        query: str,
        *,
        limit: int = 8,
        source_id: str | None = None,
    ) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 50))
        candidate_limit = max(24, limit * 4)
        lexical: list[sqlite3.Row] = []
        match = fts_query(query)
        if match:
            sql = """
                SELECT c.*, s.title, s.source_type, s.source_uri, s.scope,
                       bm25(corpus_chunks_fts) AS distance
                FROM corpus_chunks_fts
                JOIN corpus_chunks c ON c.id = corpus_chunks_fts.rowid
                JOIN corpus_sources s ON s.source_id = c.source_id
                WHERE corpus_chunks_fts MATCH ?
            """
            params: list[Any] = [match]
            if source_id:
                sql += " AND c.source_id = ?"
                params.append(source_id)
            sql += " ORDER BY distance LIMIT ?"
            params.append(candidate_limit)
            lexical = self.connection.execute(sql, params).fetchall()

        rows: dict[int, dict[str, Any]] = {}
        scores: dict[int, float] = {}
        matched: dict[int, set[str]] = {}
        for rank, row in enumerate(lexical, start=1):
            item = dict(row)
            rows[item["id"]] = item
            scores[item["id"]] = 1 / (60 + rank)
            matched.setdefault(item["id"], set()).add("fts")

        if self.enable_vectors:
            try:
                import sqlite_vec

                vector = list(self._get_embedder()([query]))[0]
                sql = """
                    SELECT c.*, s.title, s.source_type, s.source_uri, s.scope,
                           v.distance
                    FROM vec_corpus_chunks v
                    JOIN corpus_chunks c ON c.id = v.rowid
                    JOIN corpus_sources s ON s.source_id = c.source_id
                    WHERE v.embedding MATCH ? AND k = ?
                """
                params = [sqlite_vec.serialize_float32(vector), candidate_limit]
                vector_rows = self.connection.execute(sql, params).fetchall()
                if source_id:
                    vector_rows = [row for row in vector_rows if row["source_id"] == source_id]
                for rank, row in enumerate(vector_rows, start=1):
                    item = dict(row)
                    rows[item["id"]] = item
                    scores[item["id"]] = scores.get(item["id"], 0) + 1 / (60 + rank)
                    matched.setdefault(item["id"], set()).add("vector")
            except (sqlite3.OperationalError, RuntimeError, IndexError):
                pass

        result: list[dict[str, Any]] = []
        for chunk_id, item in rows.items():
            result.append(
                {
                    "source_id": item["source_id"],
                    "title": item["title"],
                    "source_type": item["source_type"],
                    "source_uri": item["source_uri"],
                    "scope": item["scope"],
                    "chunk_index": item["chunk_index"],
                    "start_ms": item["start_ms"],
                    "end_ms": item["end_ms"],
                    "heading": item["heading"],
                    "text": item["content"],
                    "score": round(scores[chunk_id], 8),
                    "matched_by": sorted(matched[chunk_id]),
                    "source_ref": f"corpus://{item['source_id']}#{item['start_ms']}",
                }
            )
        return sorted(result, key=lambda row: (-row["score"], row["start_ms"]))[:limit]

    def status(self) -> dict[str, Any]:
        return {
            "sources": self.connection.execute("SELECT COUNT(*) FROM corpus_sources").fetchone()[0],
            "chunks": self.connection.execute("SELECT COUNT(*) FROM corpus_chunks").fetchone()[0],
            "vectors": (
                self.connection.execute("SELECT COUNT(*) FROM vec_corpus_chunks").fetchone()[0]
                if self.enable_vectors
                else 0
            ),
            "database": str(self.connection.execute("PRAGMA database_list").fetchone()[2]),
        }


def open_connection(path: str | Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    import sqlite_vec

    connection = sqlite3.connect(Path(path))
    connection.row_factory = sqlite3.Row
    connection.enable_load_extension(True)
    sqlite_vec.load(connection)
    connection.enable_load_extension(False)
    return connection


def command_ingest(args: argparse.Namespace) -> int:
    transcript_path = Path(args.transcript).resolve()
    text = transcript_path.read_text(encoding="utf-8-sig")
    connection = open_connection(args.database)
    try:
        result = CorpusStore(connection).ingest_source(
            {
                "source_id": args.source_id,
                "title": args.title,
                "source_type": args.source_type,
                "source_uri": args.source_uri,
                "scope": args.scope,
                "local_path": args.local_path,
                "transcript_path": str(transcript_path),
                "video_path": args.video_path,
                "language": args.language,
                "metadata": json.loads(args.metadata_json),
            },
            text,
        )
        print(json.dumps(result, ensure_ascii=False))
    finally:
        connection.close()
    return 0


def command_search(args: argparse.Namespace) -> int:
    connection = open_connection(args.database)
    try:
        result = CorpusStore(connection).search(
            args.query, limit=args.limit, source_id=args.source_id
        )
        print(json.dumps({"query": args.query, "results": result}, ensure_ascii=False))
    finally:
        connection.close()
    return 0


def command_status(args: argparse.Namespace) -> int:
    connection = open_connection(args.database)
    try:
        print(json.dumps(CorpusStore(connection).status(), ensure_ascii=False))
    finally:
        connection.close()
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Bounded source corpus index")
    result.add_argument("--database", default=str(DEFAULT_DB_PATH))
    sub = result.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest")
    ingest.add_argument("--source-id", required=True)
    ingest.add_argument("--title", required=True)
    ingest.add_argument("--source-type", default="youtube")
    ingest.add_argument("--source-uri", default="")
    ingest.add_argument("--scope", default="ResearchCorpus/YouTube")
    ingest.add_argument("--local-path", default="")
    ingest.add_argument("--transcript", required=True)
    ingest.add_argument("--video-path", default="")
    ingest.add_argument("--language", default="ru")
    ingest.add_argument("--metadata-json", default="{}")
    ingest.set_defaults(func=command_ingest)

    search = sub.add_parser("search")
    search.add_argument("query")
    search.add_argument("--source-id")
    search.add_argument("--limit", type=int, default=8)
    search.set_defaults(func=command_search)

    status = sub.add_parser("status")
    status.set_defaults(func=command_status)
    return result


def main() -> int:
    args = parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
