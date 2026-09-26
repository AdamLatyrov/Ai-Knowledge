"""Small, deterministic runtime for modes, modules, objects and query tags.

This module is deliberately independent from the document retrieval pipeline. It
resolves the semantic envelope first; the existing memory_service can then be
used for evidence retrieval inside the selected scope.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import sqlite3
from typing import Any


SCHEMA = """
CREATE TABLE IF NOT EXISTS knowledge_modules (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    aliases_json TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('proposed', 'active', 'paused', 'archived')),
    version TEXT NOT NULL,
    retrieval_scope TEXT,
    behavior_prompt_path TEXT,
    schema_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
) STRICT;
CREATE INDEX IF NOT EXISTS idx_knowledge_modules_status
ON knowledge_modules(status, updated_at DESC);

CREATE TABLE IF NOT EXISTS agent_modes (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    aliases_json TEXT NOT NULL,
    prompt_path TEXT NOT NULL,
    version TEXT NOT NULL,
    allowed_modules_json TEXT NOT NULL,
    allowed_tools_json TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('proposed', 'available', 'archived')),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
) STRICT;

CREATE TABLE IF NOT EXISTS knowledge_objects (
    id TEXT PRIMARY KEY,
    module_id TEXT NOT NULL,
    parent_id TEXT,
    object_type TEXT NOT NULL,
    title TEXT NOT NULL,
    summary TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('draft', 'active', 'superseded', 'archived')),
    source TEXT NOT NULL,
    sensitivity TEXT NOT NULL CHECK(sensitivity IN ('public', 'internal', 'personal', 'sensitive')),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(module_id) REFERENCES knowledge_modules(id),
    FOREIGN KEY(parent_id) REFERENCES knowledge_objects(id)
) STRICT;
CREATE INDEX IF NOT EXISTS idx_knowledge_objects_module_parent
ON knowledge_objects(module_id, parent_id, status, updated_at DESC);

CREATE TABLE IF NOT EXISTS knowledge_object_claims (
    id TEXT PRIMARY KEY,
    object_id TEXT NOT NULL,
    predicate TEXT NOT NULL,
    value_json TEXT NOT NULL,
    value_text TEXT NOT NULL,
    confidence REAL NOT NULL CHECK(confidence >= 0.0 AND confidence <= 1.0),
    source TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('active', 'superseded', 'retracted')),
    valid_from TEXT,
    valid_to TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY(object_id) REFERENCES knowledge_objects(id) ON DELETE CASCADE
) STRICT;
CREATE INDEX IF NOT EXISTS idx_knowledge_claims_object_status
ON knowledge_object_claims(object_id, status, created_at DESC);

CREATE TABLE IF NOT EXISTS knowledge_tags (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    display_name TEXT NOT NULL,
    created_at TEXT NOT NULL
) STRICT;

CREATE TABLE IF NOT EXISTS knowledge_object_tags (
    object_id TEXT NOT NULL,
    tag_id INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY(object_id, tag_id),
    FOREIGN KEY(object_id) REFERENCES knowledge_objects(id) ON DELETE CASCADE,
    FOREIGN KEY(tag_id) REFERENCES knowledge_tags(id) ON DELETE CASCADE
) STRICT;
"""


TAG_ALIASES = {
    "java": "java",
    "джав": "java",
    "concurrency": "concurrency",
    "конкурент": "concurrency",
    "многопоточ": "concurrency",
    "jmm": "jmm",
    "интервью": "interview",
    "собеседован": "interview",
    "interview": "interview",
    "пробел": "gap",
    "gap": "gap",
    "грант": "grant",
    "grant": "grant",
    "массаж": "massage",
    "massage": "massage",
}


def now_iso() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _safe_id(value: str, field: str) -> str:
    value = value.strip()
    if not value or len(value) > 160:
        raise ValueError(f"{field} must be 1 to 160 characters")
    return value


def ensure_schema(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript(SCHEMA)
    columns = {row[1] for row in connection.execute("PRAGMA table_info(knowledge_modules)")}
    if "retrieval_scope" not in columns:
        connection.execute("ALTER TABLE knowledge_modules ADD COLUMN retrieval_scope TEXT")
    connection.execute(
        """
        INSERT OR IGNORE INTO knowledge_modules
          (id, title, description, aliases_json, status, version, retrieval_scope,
           behavior_prompt_path, schema_json, created_at, updated_at)
        VALUES (?, ?, ?, ?, 'active', '1.0.0', ?, ?, ?, ?, ?)
        """,
        (
            "java-preparation",
            "Senior Java Preparation",
            "Structured learning state, gaps, attempts and review schedule for Java preparation.",
            json.dumps(["java", "java prep", "java interview", "собеседование java"], ensure_ascii=False),
            "Senior Java Preparation",
            "modes/java-preparation.md",
            json.dumps({"objects": ["skill", "topic", "gap", "attempt", "review"]}, ensure_ascii=False),
            now_iso(),
            now_iso(),
        ),
    )
    connection.execute(
        """
        INSERT OR IGNORE INTO knowledge_objects
          (id, module_id, parent_id, object_type, title, summary, status,
           source, sensitivity, created_at, updated_at)
        VALUES (?, 'java-preparation', NULL, 'module', ?, ?, 'active', ?, 'personal', ?, ?)
        """,
        (
            "module://java-preparation",
            "Senior Java Preparation",
            "Current Java preparation state, topics, gaps, attempts and review schedule.",
            "system:module-registry",
            now_iso(),
            now_iso(),
        ),
    )


def extract_query_tags(query: str) -> list[str]:
    lowered = query.casefold()
    found = set()
    for alias, canonical in TAG_ALIASES.items():
        if alias in lowered:
            found.add(canonical)
    return sorted(found)


def resolve_depth(query: str) -> str:
    lowered = query.casefold()
    if any(word in lowered for word in ("источник", "доказатель", "все факты", "полностью", "evidence")):
        return "evidence"
    if any(word in lowered for word in ("подроб", "деталь", "конкрет", "почему", "как именно")):
        return "detail"
    if any(word in lowered for word in ("какие", "что дальше", "пробел", "статус", "ключев")):
        return "standard"
    return "overview"


def create_module(
    connection: sqlite3.Connection,
    *,
    module_id: str,
    title: str,
    description: str,
    aliases: list[str] | None = None,
    status: str = "active",
    version: str = "1.0.0",
    retrieval_scope: str | None = None,
    behavior_prompt_path: str | None = None,
    schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ensure_schema(connection)
    module_id = _safe_id(module_id, "module_id")
    created = now_iso()
    aliases = sorted({a.strip().casefold() for a in (aliases or []) if a.strip()})
    with connection:
        connection.execute(
            """
            INSERT INTO knowledge_modules
              (id, title, description, aliases_json, status, version,
               retrieval_scope, behavior_prompt_path, schema_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
              title=excluded.title, description=excluded.description,
              aliases_json=excluded.aliases_json, status=excluded.status,
              version=excluded.version, retrieval_scope=excluded.retrieval_scope,
              behavior_prompt_path=excluded.behavior_prompt_path,
              schema_json=excluded.schema_json, updated_at=excluded.updated_at
            """,
            (module_id, title.strip(), description.strip(), json.dumps(aliases),
             status, version, retrieval_scope, behavior_prompt_path,
             json.dumps(schema or {}, ensure_ascii=False, sort_keys=True), created, created),
        )
    return {"id": module_id, "title": title, "status": status, "version": version}


def activate_module(
    connection: sqlite3.Connection,
    *,
    module_id: str,
    title: str,
    description: str,
    aliases: list[str],
    retrieval_scope: str | None,
    behavior_prompt_path: str | None,
    schema: dict[str, Any] | None,
    confirm: bool,
) -> dict[str, Any]:
    if not confirm:
        raise ValueError("Explicit confirmation is required before module activation")
    if behavior_prompt_path and not behavior_prompt_path.startswith("modes/"):
        raise ValueError("behavior_prompt_path must stay inside modes/")
    module = create_module(
        connection,
        module_id=module_id,
        title=title,
        description=description,
        aliases=aliases,
        status="active",
        retrieval_scope=retrieval_scope,
        behavior_prompt_path=behavior_prompt_path,
        schema=schema,
    )
    root_id = f"module://{module_id}"
    create_object(
        connection,
        module_id=module_id,
        object_id=root_id,
        object_type="module",
        title=title,
        summary=description,
        source="user:confirmed-module",
    )
    return {**module, "root_object": root_id, "activated": True}


def create_object(
    connection: sqlite3.Connection,
    *,
    module_id: str,
    object_id: str,
    object_type: str,
    title: str,
    summary: str,
    parent_id: str | None = None,
    source: str = "user:explicit",
    sensitivity: str = "personal",
) -> dict[str, Any]:
    ensure_schema(connection)
    object_id = _safe_id(object_id, "object_id")
    created = now_iso()
    with connection:
        connection.execute(
            """
            INSERT INTO knowledge_objects
              (id, module_id, parent_id, object_type, title, summary, status,
               source, sensitivity, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, 'active', ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
              parent_id=excluded.parent_id, object_type=excluded.object_type,
              title=excluded.title, summary=excluded.summary,
              source=excluded.source, sensitivity=excluded.sensitivity,
              updated_at=excluded.updated_at
            """,
            (object_id, module_id, parent_id, object_type, title.strip(), summary.strip(),
             source, sensitivity, created, created),
        )
    return {"id": object_id, "module_id": module_id, "parent_id": parent_id, "title": title}


def find_module(connection: sqlite3.Connection, query: str) -> str | None:
    ensure_schema(connection)
    lowered = query.casefold()
    rows = connection.execute(
        "SELECT id, title, aliases_json FROM knowledge_modules WHERE status = 'active'"
    ).fetchall()
    for row in rows:
        values = [row[0], row[1], *json.loads(row[2])]
        if any(str(value).casefold() in lowered for value in values):
            return row[0]
    tags = set(extract_query_tags(query))
    for row in rows:
        values = {str(row[0]).casefold(), str(row[1]).casefold()}
        if tags & {value for value in values if value}:
            return row[0]
    return None


def build_runtime_plan(
    connection: sqlite3.Connection,
    *,
    query: str,
    module: str | None = None,
    mode: str | None = None,
    depth: str | None = None,
) -> dict[str, Any]:
    ensure_schema(connection)
    selected_module = module or find_module(connection, query)
    return {
        "query": query,
        "module": selected_module,
        "mode": mode,
        "tags": extract_query_tags(query),
        "depth": depth if depth and depth != "auto" else resolve_depth(query),
        "needs_module_proposal": selected_module is None,
        "retrieval": "sql+fts+vector+graph",
    }


def matching_modes(modes: list[dict[str, Any]], query: str) -> list[dict[str, Any]]:
    lowered = query.casefold()
    matches = []
    for mode in modes:
        haystack = " ".join(
            str(mode.get(key, ""))
            for key in ("mode", "title", "summary", "activation")
        ).casefold()
        if any(token and token in haystack for token in re.findall(r"[\wа-яё-]+", lowered)):
            matches.append(mode)
    return matches


def list_modules(connection: sqlite3.Connection, query: str = "") -> list[dict[str, Any]]:
    ensure_schema(connection)
    rows = connection.execute(
        "SELECT id, title, description, aliases_json, version, retrieval_scope FROM knowledge_modules "
        "WHERE status = 'active' ORDER BY title"
    ).fetchall()
    lowered = query.casefold().strip()
    result = []
    for row in rows:
        aliases = json.loads(row["aliases_json"])
        haystack = " ".join([row["id"], row["title"], row["description"], *aliases]).casefold()
        if not lowered or lowered in haystack:
            result.append({"id": row["id"], "title": row["title"], "description": row["description"], "version": row["version"], "retrieval_scope": row["retrieval_scope"]})
    return result


def module_retrieval_scope(connection: sqlite3.Connection, module_id: str | None) -> str | None:
    if not module_id:
        return None
    row = connection.execute(
        "SELECT retrieval_scope FROM knowledge_modules WHERE id = ? AND status = 'active'",
        (module_id,),
    ).fetchone()
    return row[0] if row else None


def object_context(
    connection: sqlite3.Connection,
    *,
    module_id: str | None,
    tags: list[str],
    depth: str,
    limit: int = 12,
) -> list[dict[str, Any]]:
    """Return a compact object tree; evidence remains a separate retrieval step."""
    if not module_id:
        return []
    rows = connection.execute(
        """
        SELECT o.id, o.parent_id, o.object_type, o.title, o.summary, o.status,
               o.source, o.updated_at,
               GROUP_CONCAT(t.name, ',') AS tags
        FROM knowledge_objects o
        LEFT JOIN knowledge_object_tags ot ON ot.object_id = o.id
        LEFT JOIN knowledge_tags t ON t.id = ot.tag_id
        WHERE o.module_id = ? AND o.status = 'active'
        GROUP BY o.id
        ORDER BY o.parent_id IS NOT NULL, o.updated_at DESC
        LIMIT ?
        """,
        (module_id, limit),
    ).fetchall()
    requested = set(tags)
    result = []
    for row in rows:
        row_tags = set((row["tags"] or "").split(",")) - {""}
        tag_score = len(requested & row_tags)
        if requested and tag_score == 0 and depth in {"detail", "evidence"}:
            continue
        result.append(
            {
                "id": row["id"],
                "parent_id": row["parent_id"],
                "type": row["object_type"],
                "title": row["title"],
                "summary": row["summary"],
                "status": row["status"],
                "source": row["source"],
                "updated_at": row["updated_at"],
                "tags": sorted(row_tags),
                "tag_score": tag_score,
            }
        )
    return sorted(result, key=lambda item: (-item["tag_score"], item["parent_id"] is not None))
