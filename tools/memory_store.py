from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sqlite3
import sys
import uuid
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
CONFIG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
DEFAULT_DB_PATH = Path(
    os.environ.get("AI_KNOWLEDGE_DATABASE", ROOT / CONFIG["database"])
).expanduser().resolve()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


SCHEMA = """
CREATE TABLE IF NOT EXISTS memory_facts (
    id TEXT PRIMARY KEY,
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    value_json TEXT NOT NULL,
    value_text TEXT NOT NULL,
    scope TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('active', 'superseded', 'retracted')),
    confidence REAL NOT NULL CHECK(confidence >= 0.0 AND confidence <= 1.0),
    sensitivity TEXT NOT NULL CHECK(sensitivity IN ('public', 'internal', 'personal', 'sensitive')),
    source TEXT NOT NULL,
    valid_from TEXT,
    valid_to TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
) STRICT;
CREATE UNIQUE INDEX IF NOT EXISTS idx_memory_facts_active_key
ON memory_facts(subject, predicate, scope)
WHERE status = 'active';
CREATE INDEX IF NOT EXISTS idx_memory_facts_scope_status
ON memory_facts(scope, status);

CREATE TABLE IF NOT EXISTS memory_write_proposals (
    id TEXT PRIMARY KEY,
    operation TEXT NOT NULL CHECK(operation IN ('add', 'correct')),
    status TEXT NOT NULL CHECK(status IN ('pending', 'committed', 'rejected')),
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    value_json TEXT NOT NULL,
    value_text TEXT NOT NULL,
    scope TEXT NOT NULL,
    source TEXT NOT NULL,
    confidence REAL NOT NULL CHECK(confidence >= 0.0 AND confidence <= 1.0),
    sensitivity TEXT NOT NULL CHECK(sensitivity IN ('public', 'internal', 'personal', 'sensitive')),
    replaces_fact_id TEXT,
    committed_fact_id TEXT,
    created_at TEXT NOT NULL,
    committed_at TEXT,
    FOREIGN KEY(replaces_fact_id) REFERENCES memory_facts(id),
    FOREIGN KEY(committed_fact_id) REFERENCES memory_facts(id)
) STRICT;
CREATE INDEX IF NOT EXISTS idx_memory_write_proposals_status
ON memory_write_proposals(status, created_at);

CREATE TABLE IF NOT EXISTS memory_fact_revisions (
    id INTEGER PRIMARY KEY,
    fact_id TEXT NOT NULL,
    proposal_id TEXT,
    action TEXT NOT NULL CHECK(action IN ('insert', 'supersede', 'retract')),
    before_json TEXT,
    after_json TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY(fact_id) REFERENCES memory_facts(id),
    FOREIGN KEY(proposal_id) REFERENCES memory_write_proposals(id)
) STRICT;
CREATE INDEX IF NOT EXISTS idx_memory_fact_revisions_fact
ON memory_fact_revisions(fact_id, created_at);

CREATE TABLE IF NOT EXISTS memory_handoffs (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    project TEXT NOT NULL,
    result TEXT NOT NULL,
    decisions_json TEXT NOT NULL,
    changed_json TEXT NOT NULL,
    verified_json TEXT NOT NULL,
    next_json TEXT NOT NULL,
    open_questions_json TEXT NOT NULL,
    source_session TEXT NOT NULL,
    created_at TEXT NOT NULL
) STRICT;
CREATE INDEX IF NOT EXISTS idx_memory_handoffs_project_created
ON memory_handoffs(project, created_at DESC);

CREATE TABLE IF NOT EXISTS memory_items (
    id TEXT PRIMARY KEY,
    item_type TEXT NOT NULL,
    title TEXT NOT NULL,
    summary TEXT NOT NULL,
    content_json TEXT NOT NULL,
    scope TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('idea', 'draft', 'ready', 'published', 'archived')),
    source TEXT NOT NULL,
    sensitivity TEXT NOT NULL CHECK(sensitivity IN ('public', 'internal', 'personal', 'sensitive')),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
) STRICT;
CREATE INDEX IF NOT EXISTS idx_memory_items_scope_status
ON memory_items(scope, status, updated_at DESC);

CREATE TABLE IF NOT EXISTS memory_tags (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    display_name TEXT NOT NULL,
    created_at TEXT NOT NULL
) STRICT;

CREATE TABLE IF NOT EXISTS memory_item_tags (
    item_id TEXT NOT NULL,
    tag_id INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY(item_id, tag_id),
    FOREIGN KEY(item_id) REFERENCES memory_items(id) ON DELETE CASCADE,
    FOREIGN KEY(tag_id) REFERENCES memory_tags(id) ON DELETE CASCADE
) STRICT;

CREATE TABLE IF NOT EXISTS memory_relations (
    id TEXT PRIMARY KEY,
    source_uri TEXT NOT NULL,
    relation_type TEXT NOT NULL,
    target_uri TEXT NOT NULL,
    label TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    created_at TEXT NOT NULL
) STRICT;
CREATE INDEX IF NOT EXISTS idx_memory_relations_source
ON memory_relations(source_uri, relation_type);
CREATE INDEX IF NOT EXISTS idx_memory_relations_target
ON memory_relations(target_uri, relation_type);

CREATE TABLE IF NOT EXISTS external_projections (
    id TEXT PRIMARY KEY,
    source_system TEXT NOT NULL,
    source_event_id INTEGER NOT NULL,
    entity_key TEXT NOT NULL,
    projection_type TEXT NOT NULL,
    scope TEXT NOT NULL,
    title TEXT NOT NULL,
    summary TEXT NOT NULL,
    content_json TEXT NOT NULL,
    sensitivity TEXT NOT NULL CHECK(sensitivity IN ('public', 'internal', 'personal', 'sensitive')),
    source_version TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active' CHECK(status IN ('active', 'deleted')),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(source_system, entity_key, projection_type)
) STRICT;
CREATE INDEX IF NOT EXISTS idx_external_projections_scope_status
ON external_projections(scope, status, updated_at DESC);
"""


SECRET_PATTERNS = (
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----", re.IGNORECASE),
    re.compile(
        r"\b(password|passwd|пароль|token|api[ _-]?key|secret)\s*[:=]\s*\S+",
        re.IGNORECASE,
    ),
    re.compile(r"https?://[^\s:/]+:[^@\s]+@", re.IGNORECASE),
)


class MemoryStoreError(ValueError):
    pass


class ConfirmationRequired(MemoryStoreError):
    pass


class SecretRejected(MemoryStoreError):
    pass


class FactConflict(MemoryStoreError):
    pass


def now_iso() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def connect(path: str | Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    connection = sqlite3.connect(Path(path))
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript(SCHEMA)
    return connection


def ensure_schema(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript(SCHEMA)


def secret_like(value: str) -> bool:
    return any(pattern.search(value) for pattern in SECRET_PATTERNS)


def normalize_name(value: str, field: str) -> str:
    normalized = re.sub(r"\s+", " ", value).strip()
    if not normalized:
        raise MemoryStoreError(f"{field} must not be empty")
    if len(normalized) > 240:
        raise MemoryStoreError(f"{field} is too long")
    return normalized


def serialize_value(value: Any) -> tuple[str, str]:
    try:
        value_json = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as error:
        raise MemoryStoreError(f"value is not JSON-serializable: {error}") from error
    if len(value_json.encode("utf-8")) > 16_000:
        raise MemoryStoreError("atomic fact value exceeds 16 KB")
    value_text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)
    if secret_like(value_text):
        raise SecretRejected(
            "Fact rejected: secret-like content detected. Store only a safe secret alias."
        )
    return value_json, value_text


def proposal_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "operation": row["operation"],
        "status": row["status"],
        "subject": row["subject"],
        "predicate": row["predicate"],
        "value": json.loads(row["value_json"]),
        "scope": row["scope"],
        "source": row["source"],
        "confidence": row["confidence"],
        "sensitivity": row["sensitivity"],
        "replaces_fact_id": row["replaces_fact_id"],
        "committed_fact_id": row["committed_fact_id"],
        "created_at": row["created_at"],
        "committed_at": row["committed_at"],
    }


def fact_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "subject": row["subject"],
        "predicate": row["predicate"],
        "value": json.loads(row["value_json"]),
        "value_text": row["value_text"],
        "scope": row["scope"],
        "status": row["status"],
        "confidence": row["confidence"],
        "sensitivity": row["sensitivity"],
        "source": row["source"],
        "valid_from": row["valid_from"],
        "valid_to": row["valid_to"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def propose_fact(
    connection: sqlite3.Connection,
    *,
    subject: str,
    predicate: str,
    value: Any,
    scope: str,
    source: str,
    confidence: float,
    sensitivity: str,
    operation: str = "add",
    replaces_fact_id: str | None = None,
) -> dict[str, Any]:
    subject = normalize_name(subject, "subject")
    predicate = normalize_name(predicate, "predicate")
    scope = normalize_name(scope, "scope")
    source = normalize_name(source, "source")
    if operation not in {"add", "correct"}:
        raise MemoryStoreError("operation must be add or correct")
    if operation == "correct" and not replaces_fact_id:
        raise MemoryStoreError("correction requires replaces_fact_id")
    if operation == "add" and replaces_fact_id:
        raise MemoryStoreError("add operation cannot replace a fact")
    if sensitivity not in {"public", "internal", "personal", "sensitive"}:
        raise MemoryStoreError("unsupported sensitivity")
    if not 0.0 <= float(confidence) <= 1.0:
        raise MemoryStoreError("confidence must be between 0 and 1")
    value_json, value_text = serialize_value(value)
    if secret_like("\n".join((subject, predicate, scope, source))):
        raise SecretRejected("Fact metadata contains secret-like content")

    proposal_id = uuid.uuid4().hex
    created_at = now_iso()
    with connection:
        connection.execute(
            """
            INSERT INTO memory_write_proposals (
                id, operation, status, subject, predicate, value_json, value_text,
                scope, source, confidence, sensitivity, replaces_fact_id, created_at
            ) VALUES (?, ?, 'pending', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                proposal_id,
                operation,
                subject,
                predicate,
                value_json,
                value_text,
                scope,
                source,
                float(confidence),
                sensitivity,
                replaces_fact_id,
                created_at,
            ),
        )
    row = connection.execute(
        "SELECT * FROM memory_write_proposals WHERE id = ?", (proposal_id,)
    ).fetchone()
    return proposal_dict(row)


def _fact_snapshot(row: sqlite3.Row) -> str:
    return json.dumps(fact_dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def commit_proposal(
    connection: sqlite3.Connection, proposal_id: str, *, confirmed: bool
) -> dict[str, Any]:
    if not confirmed:
        raise ConfirmationRequired(
            "Explicit confirmation is required before durable memory is changed"
        )
    proposal = connection.execute(
        "SELECT * FROM memory_write_proposals WHERE id = ?", (proposal_id,)
    ).fetchone()
    if proposal is None:
        raise MemoryStoreError("proposal not found")
    if proposal["status"] == "committed":
        return {
            "proposal_id": proposal_id,
            "fact_id": proposal["committed_fact_id"],
            "status": "committed",
            "idempotent_replay": True,
            "index_stale": True,
        }
    if proposal["status"] != "pending":
        raise MemoryStoreError(f"proposal is {proposal['status']}")

    created_at = now_iso()
    fact_id = uuid.uuid4().hex
    with connection:
        if proposal["operation"] == "correct":
            previous = connection.execute(
                "SELECT * FROM memory_facts WHERE id = ?",
                (proposal["replaces_fact_id"],),
            ).fetchone()
            if previous is None or previous["status"] != "active":
                raise FactConflict("fact to correct is missing or no longer active")
            if (
                previous["subject"],
                previous["predicate"],
                previous["scope"],
            ) != (proposal["subject"], proposal["predicate"], proposal["scope"]):
                raise FactConflict("correction must keep subject, predicate, and scope")
            before = _fact_snapshot(previous)
            connection.execute(
                "UPDATE memory_facts SET status = 'superseded', valid_to = ?, updated_at = ? WHERE id = ?",
                (created_at, created_at, previous["id"]),
            )
            after_previous = connection.execute(
                "SELECT * FROM memory_facts WHERE id = ?", (previous["id"],)
            ).fetchone()
            connection.execute(
                """
                INSERT INTO memory_fact_revisions
                    (fact_id, proposal_id, action, before_json, after_json, created_at)
                VALUES (?, ?, 'supersede', ?, ?, ?)
                """,
                (
                    previous["id"],
                    proposal_id,
                    before,
                    _fact_snapshot(after_previous),
                    created_at,
                ),
            )
        else:
            active = connection.execute(
                """
                SELECT * FROM memory_facts
                WHERE subject = ? AND predicate = ? AND scope = ? AND status = 'active'
                """,
                (proposal["subject"], proposal["predicate"], proposal["scope"]),
            ).fetchone()
            if active is not None:
                if active["value_json"] == proposal["value_json"]:
                    connection.execute(
                        """
                        UPDATE memory_write_proposals
                        SET status = 'committed', committed_fact_id = ?, committed_at = ?
                        WHERE id = ?
                        """,
                        (active["id"], created_at, proposal_id),
                    )
                    return {
                        "proposal_id": proposal_id,
                        "fact_id": active["id"],
                        "status": "committed",
                        "idempotent_replay": True,
                        "index_stale": False,
                    }
                raise FactConflict(
                    "an active fact already exists; create a correction proposal with replaces_fact_id"
                )

        connection.execute(
            """
            INSERT INTO memory_facts (
                id, subject, predicate, value_json, value_text, scope, status,
                confidence, sensitivity, source, valid_from, valid_to,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, 'active', ?, ?, ?, ?, NULL, ?, ?)
            """,
            (
                fact_id,
                proposal["subject"],
                proposal["predicate"],
                proposal["value_json"],
                proposal["value_text"],
                proposal["scope"],
                proposal["confidence"],
                proposal["sensitivity"],
                proposal["source"],
                created_at,
                created_at,
                created_at,
            ),
        )
        inserted = connection.execute(
            "SELECT * FROM memory_facts WHERE id = ?", (fact_id,)
        ).fetchone()
        connection.execute(
            """
            INSERT INTO memory_fact_revisions
                (fact_id, proposal_id, action, before_json, after_json, created_at)
            VALUES (?, ?, 'insert', NULL, ?, ?)
            """,
            (fact_id, proposal_id, _fact_snapshot(inserted), created_at),
        )
        connection.execute(
            """
            UPDATE memory_write_proposals
            SET status = 'committed', committed_fact_id = ?, committed_at = ?
            WHERE id = ?
            """,
            (fact_id, created_at, proposal_id),
        )

    return {
        "proposal_id": proposal_id,
        "fact_id": fact_id,
        "status": "committed",
        "idempotent_replay": False,
        "index_stale": True,
    }


def list_facts(
    connection: sqlite3.Connection,
    *,
    status: str = "active",
    scope: str | None = None,
) -> list[dict[str, Any]]:
    sql = "SELECT * FROM memory_facts WHERE status = ?"
    params: list[Any] = [status]
    if scope:
        sql += " AND lower(scope) LIKE ?"
        params.append(f"%{scope.lower()}%")
    sql += " ORDER BY subject, predicate, created_at DESC"
    return [fact_dict(row) for row in connection.execute(sql, params).fetchall()]


def fact_documents(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    for fact in list_facts(connection, status="active"):
        content = (
            f"# Durable fact: {fact['subject']} / {fact['predicate']}\n\n"
            f"Subject: {fact['subject']}\n"
            f"Predicate: {fact['predicate']}\n"
            f"Value: {fact['value_text']}\n"
            f"Scope: {fact['scope']}\n"
            f"Source: {fact['source']}\n"
            f"Confidence: {fact['confidence']:.2f}\n"
            f"Sensitivity: {fact['sensitivity']}\n"
            f"Updated: {fact['updated_at']}\n"
        )
        documents.append(
            {
                "path": f"memory://facts/{fact['id']}",
                "collection": "durable-memory",
                "scope": fact["scope"],
                "doc_type": "durable_fact",
                "mode": "content",
                "title": f"{fact['subject']}: {fact['predicate']}",
                "modified_at": fact["updated_at"],
                "content": content,
            }
        )
    return documents


def _clean_handoff_items(items: list[str], field: str, maximum: int) -> list[str]:
    if len(items) > maximum:
        raise MemoryStoreError(f"{field} contains more than {maximum} items")
    cleaned = [re.sub(r"\s+", " ", item).strip() for item in items]
    if any(not item for item in cleaned):
        raise MemoryStoreError(f"{field} contains an empty item")
    if any(len(item) > 2000 for item in cleaned):
        raise MemoryStoreError(f"{field} contains an item longer than 2000 characters")
    return cleaned


def save_handoff(
    connection: sqlite3.Connection,
    *,
    title: str,
    project: str,
    result: str,
    decisions: list[str],
    changed: list[str],
    verified: list[str],
    next_steps: list[str],
    open_questions: list[str],
    source_session: str,
    commit: bool = True,
) -> dict[str, Any]:
    title = normalize_name(title, "title")
    project = normalize_name(project, "project")
    source_session = normalize_name(source_session, "source_session")
    result = result.strip()
    if not result or len(result) > 8000:
        raise MemoryStoreError("result must contain 1 to 8000 characters")
    decisions = _clean_handoff_items(decisions, "decisions", 20)
    changed = _clean_handoff_items(changed, "changed", 30)
    verified = _clean_handoff_items(verified, "verified", 30)
    next_steps = _clean_handoff_items(next_steps, "next", 20)
    open_questions = _clean_handoff_items(open_questions, "open_questions", 20)
    payload_text = "\n".join(
        [
            title,
            project,
            result,
            source_session,
            *decisions,
            *changed,
            *verified,
            *next_steps,
            *open_questions,
        ]
    )
    if secret_like(payload_text):
        raise SecretRejected(
            "Handoff rejected: secret-like content detected. Store only safe aliases."
        )
    handoff_id = uuid.uuid4().hex
    created_at = now_iso()
    connection.execute(
            """
            INSERT INTO memory_handoffs (
                id, title, project, result, decisions_json, changed_json,
                verified_json, next_json, open_questions_json, source_session,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                handoff_id,
                title,
                project,
                result,
                json.dumps(decisions, ensure_ascii=False),
                json.dumps(changed, ensure_ascii=False),
                json.dumps(verified, ensure_ascii=False),
                json.dumps(next_steps, ensure_ascii=False),
                json.dumps(open_questions, ensure_ascii=False),
                source_session,
                created_at,
            ),
        )
    if commit:
        connection.commit()
    return {
        "id": handoff_id,
        "uri": f"memory://handoffs/{handoff_id}",
        "stored": True,
        "storage": "sqlite",
        "created_at": created_at,
        "index_stale": True,
    }


def _handoff_bullets(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items) if items else "- None recorded."


def handoff_documents(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    rows = connection.execute(
        "SELECT * FROM memory_handoffs ORDER BY created_at DESC"
    ).fetchall()
    for row in rows:
        decisions = json.loads(row["decisions_json"])
        changed = json.loads(row["changed_json"])
        verified = json.loads(row["verified_json"])
        next_steps = json.loads(row["next_json"])
        open_questions = json.loads(row["open_questions_json"])
        content = (
            f"# Session handoff: {row['title']}\n\n"
            f"Date: {row['created_at']}\n"
            f"Project: {row['project']}\n"
            f"Source session: {row['source_session']}\n\n"
            f"## Result\n\n{row['result']}\n\n"
            f"## Decisions\n\n{_handoff_bullets(decisions)}\n\n"
            f"## Changed\n\n{_handoff_bullets(changed)}\n\n"
            f"## Verified\n\n{_handoff_bullets(verified)}\n\n"
            f"## Next\n\n{_handoff_bullets(next_steps)}\n\n"
            f"## Open questions\n\n{_handoff_bullets(open_questions)}\n"
        )
        documents.append(
            {
                "path": f"memory://handoffs/{row['id']}",
                "collection": "durable-memory",
                "scope": row["project"],
                "doc_type": "handoff",
                "mode": "content",
                "title": row["title"],
                "modified_at": row["created_at"],
                "content": content,
            }
        )
    return documents


def normalize_tag(value: str) -> str:
    normalized = value.strip().lower()
    normalized = re.sub(r"[\s_]+", "-", normalized, flags=re.UNICODE)
    normalized = re.sub(r"[^\w\-а-яё]+", "-", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"-+", "-", normalized).strip("-")
    if not normalized or len(normalized) > 80:
        raise MemoryStoreError("tag must contain 1 to 80 safe characters")
    return normalized


def save_knowledge_item(
    connection: sqlite3.Connection,
    *,
    item_type: str,
    title: str,
    summary: str,
    content: Any,
    scope: str,
    status: str,
    source: str,
    sensitivity: str,
    tags: list[str],
    relations: list[dict[str, Any]],
    confirmed: bool,
) -> dict[str, Any]:
    if not confirmed:
        raise ConfirmationRequired(
            "Explicit confirmation is required before a knowledge item is saved"
        )
    item_type = normalize_name(item_type, "item_type")
    title = normalize_name(title, "title")
    summary = summary.strip()
    scope = normalize_name(scope, "scope")
    source = normalize_name(source, "source")
    if not summary or len(summary) > 4000:
        raise MemoryStoreError("summary must contain 1 to 4000 characters")
    if status not in {"idea", "draft", "ready", "published", "archived"}:
        raise MemoryStoreError("unsupported item status")
    if sensitivity not in {"public", "internal", "personal", "sensitive"}:
        raise MemoryStoreError("unsupported sensitivity")
    try:
        content_json = json.dumps(
            content, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
    except (TypeError, ValueError) as error:
        raise MemoryStoreError(f"content is not JSON-serializable: {error}") from error
    if len(content_json.encode("utf-8")) > 64_000:
        raise MemoryStoreError("knowledge item content exceeds 64 KB")
    normalized_tags = sorted({normalize_tag(tag) for tag in tags})
    if len(normalized_tags) > 30:
        raise MemoryStoreError("knowledge item contains more than 30 tags")
    if len(relations) > 30:
        raise MemoryStoreError("knowledge item contains more than 30 relations")

    cleaned_relations: list[dict[str, Any]] = []
    for relation in relations:
        relation_type = normalize_tag(str(relation.get("type", "")))
        target = str(relation.get("target", "")).strip()
        label = str(relation.get("label", "")).strip()
        metadata = relation.get("metadata", {})
        if not target or len(target) > 1000:
            raise MemoryStoreError("relation target must contain 1 to 1000 characters")
        if len(label) > 500:
            raise MemoryStoreError("relation label exceeds 500 characters")
        try:
            metadata_json = json.dumps(
                metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
        except (TypeError, ValueError) as error:
            raise MemoryStoreError(
                f"relation metadata is not JSON-serializable: {error}"
            ) from error
        cleaned_relations.append(
            {
                "type": relation_type,
                "target": target,
                "label": label,
                "metadata_json": metadata_json,
            }
        )

    secret_check = "\n".join(
        [title, summary, content_json, scope, source]
        + normalized_tags
        + [
            f"{relation['type']} {relation['target']} {relation['label']} {relation['metadata_json']}"
            for relation in cleaned_relations
        ]
    )
    if secret_like(secret_check):
        raise SecretRejected(
            "Knowledge item rejected: secret-like content detected. Store only safe aliases."
        )

    item_id = uuid.uuid4().hex
    item_uri = f"memory://items/{item_id}"
    created_at = now_iso()
    saved_relations: list[dict[str, str]] = []
    with connection:
        connection.execute(
            """
            INSERT INTO memory_items (
                id, item_type, title, summary, content_json, scope, status,
                source, sensitivity, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                item_id,
                item_type,
                title,
                summary,
                content_json,
                scope,
                status,
                source,
                sensitivity,
                created_at,
                created_at,
            ),
        )
        display_by_normalized = {
            normalize_tag(tag): re.sub(r"\s+", " ", tag).strip() for tag in tags
        }
        for tag in normalized_tags:
            connection.execute(
                """
                INSERT INTO memory_tags(name, display_name, created_at)
                VALUES (?, ?, ?)
                ON CONFLICT(name) DO NOTHING
                """,
                (tag, display_by_normalized[tag], created_at),
            )
            tag_id = connection.execute(
                "SELECT id FROM memory_tags WHERE name = ?", (tag,)
            ).fetchone()[0]
            connection.execute(
                """
                INSERT INTO memory_item_tags(item_id, tag_id, created_at)
                VALUES (?, ?, ?)
                """,
                (item_id, tag_id, created_at),
            )
        for relation in cleaned_relations:
            relation_id = uuid.uuid4().hex
            connection.execute(
                """
                INSERT INTO memory_relations (
                    id, source_uri, relation_type, target_uri, label,
                    metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    relation_id,
                    item_uri,
                    relation["type"],
                    relation["target"],
                    relation["label"],
                    relation["metadata_json"],
                    created_at,
                ),
            )
            saved_relations.append(
                {
                    "id": relation_id,
                    "type": relation["type"],
                    "target": relation["target"],
                    "label": relation["label"],
                }
            )
    return {
        "id": item_id,
        "uri": item_uri,
        "stored": True,
        "storage": "sqlite",
        "tags": normalized_tags,
        "relations": saved_relations,
        "created_at": created_at,
        "index_stale": True,
    }


def upsert_external_projection(
    connection: sqlite3.Connection,
    *,
    source_system: str,
    source_event_id: int,
    entity_key: str,
    projection_type: str,
    scope: str,
    title: str,
    summary: str,
    content: Any,
    sensitivity: str,
    source_version: str,
) -> dict[str, Any]:
    source_system = normalize_name(source_system, "source_system")
    entity_key = normalize_name(entity_key, "entity_key")
    projection_type = normalize_name(projection_type, "projection_type")
    scope = normalize_name(scope, "scope")
    title = normalize_name(title, "title")
    source_version = normalize_name(source_version, "source_version")
    summary = summary.strip()
    if not summary or len(summary) > 4000:
        raise MemoryStoreError("summary must contain 1 to 4000 characters")
    if sensitivity not in {"public", "internal", "personal", "sensitive"}:
        raise MemoryStoreError("unsupported sensitivity")
    try:
        content_json = json.dumps(content, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as error:
        raise MemoryStoreError(f"projection content is not JSON-serializable: {error}") from error
    if len(content_json.encode("utf-8")) > 32_000:
        raise MemoryStoreError("external projection content exceeds 32 KB")
    if secret_like("\n".join([source_system, entity_key, projection_type, scope, title, summary, content_json, source_version])):
        raise SecretRejected("External projection rejected: secret-like content detected")

    projection_id = uuid.uuid5(
        uuid.NAMESPACE_URL, f"projection:{source_system}:{entity_key}:{projection_type}"
    ).hex
    timestamp = now_iso()
    with connection:
        connection.execute(
            """
            INSERT INTO external_projections(
                id, source_system, source_event_id, entity_key, projection_type,
                scope, title, summary, content_json, sensitivity, source_version,
                status, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?)
            ON CONFLICT(source_system, entity_key, projection_type) DO UPDATE SET
                source_event_id=excluded.source_event_id,
                scope=excluded.scope,
                title=excluded.title,
                summary=excluded.summary,
                content_json=excluded.content_json,
                sensitivity=excluded.sensitivity,
                source_version=excluded.source_version,
                status='active',
                updated_at=excluded.updated_at
            """,
            (
                projection_id, source_system, int(source_event_id), entity_key,
                projection_type, scope, title, summary, content_json, sensitivity,
                source_version, timestamp, timestamp,
            ),
        )
    return {"id": projection_id, "stored": True, "derived": True, "updated_at": timestamp}


def delete_external_projection(
    connection: sqlite3.Connection, *, source_system: str, entity_key: str, projection_type: str
) -> bool:
    with connection:
        cursor = connection.execute(
            """
            UPDATE external_projections SET status='deleted', updated_at=?
            WHERE source_system=? AND entity_key=? AND projection_type=?
            """,
            (now_iso(), source_system, entity_key, projection_type),
        )
    return cursor.rowcount > 0


def reconcile_external_projections(
    connection: sqlite3.Connection,
    *,
    source_system: str,
    projection_type: str,
    active_entity_keys: list[str],
) -> int:
    source_system = normalize_name(source_system, "source_system")
    projection_type = normalize_name(projection_type, "projection_type")
    active = {normalize_name(value, "active_entity_key") for value in active_entity_keys}
    rows = connection.execute(
        """
        SELECT entity_key FROM external_projections
        WHERE source_system=? AND projection_type=? AND status='active'
        """,
        (source_system, projection_type),
    ).fetchall()
    stale = [row["entity_key"] for row in rows if row["entity_key"] not in active]
    if not stale:
        return 0
    timestamp = now_iso()
    with connection:
        connection.executemany(
            """
            UPDATE external_projections SET status='deleted', updated_at=?
            WHERE source_system=? AND entity_key=? AND projection_type=?
            """,
            [(timestamp, source_system, entity_key, projection_type) for entity_key in stale],
        )
    return len(stale)


def external_projection_documents(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    rows = connection.execute(
        "SELECT * FROM external_projections WHERE status='active' ORDER BY updated_at DESC"
    ).fetchall()
    for row in rows:
        content = json.loads(row["content_json"])
        rendered = (
            f"# External projection: {row['title']}\n\n"
            f"Scope: {row['scope']}\n"
            f"Entity: {row['entity_key']}\n"
            f"Source: {row['source_system']} event {row['source_event_id']} ({row['source_version']})\n"
            f"Sensitivity: {row['sensitivity']}\n"
            f"Updated: {row['updated_at']}\n\n"
            f"## Summary\n\n{row['summary']}\n\n"
            f"## Bounded data\n\n```json\n{json.dumps(content, ensure_ascii=False, indent=2, sort_keys=True)}\n```\n"
        )
        documents.append({
            "path": f"projection://{row['source_system']}/{row['id']}",
            "collection": "external-projections",
            "scope": row["scope"],
            "doc_type": "external_projection",
            "mode": "content",
            "title": row["title"],
            "modified_at": row["updated_at"],
            "content": rendered,
        })
    return documents


def knowledge_item_documents(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    items = connection.execute(
        "SELECT * FROM memory_items WHERE status <> 'archived' ORDER BY updated_at DESC"
    ).fetchall()
    for item in items:
        item_uri = f"memory://items/{item['id']}"
        tags = [
            row[0]
            for row in connection.execute(
                """
                SELECT t.name FROM memory_item_tags it
                JOIN memory_tags t ON t.id = it.tag_id
                WHERE it.item_id = ? ORDER BY t.name
                """,
                (item["id"],),
            ).fetchall()
        ]
        relations = connection.execute(
            """
            SELECT relation_type, target_uri, label
            FROM memory_relations WHERE source_uri = ?
            ORDER BY relation_type, target_uri
            """,
            (item_uri,),
        ).fetchall()
        content = json.loads(item["content_json"])
        rendered_content = json.dumps(content, ensure_ascii=False, indent=2, sort_keys=True)
        relation_lines = (
            "\n".join(
                f"- {row['relation_type']} -> {row['target_uri']}"
                + (f" ({row['label']})" if row["label"] else "")
                for row in relations
            )
            or "- None"
        )
        rendered = (
            f"# Knowledge item: {item['title']}\n\n"
            f"Type: {item['item_type']}\n"
            f"Status: {item['status']}\n"
            f"Scope: {item['scope']}\n"
            f"Tags: {', '.join(tags)}\n"
            f"Source: {item['source']}\n"
            f"Sensitivity: {item['sensitivity']}\n"
            f"Updated: {item['updated_at']}\n\n"
            f"## Summary\n\n{item['summary']}\n\n"
            f"## Material\n\n```json\n{rendered_content}\n```\n\n"
            f"## Relations\n\n{relation_lines}\n"
        )
        documents.append(
            {
                "path": item_uri,
                "collection": "durable-memory",
                "scope": item["scope"],
                "doc_type": "knowledge_item",
                "mode": "content",
                "title": item["title"],
                "modified_at": item["updated_at"],
                "content": rendered,
            }
        )
    return documents


def command_propose(args: argparse.Namespace) -> int:
    connection = connect(args.database)
    value = json.loads(args.value_json)
    proposal = propose_fact(
        connection,
        subject=args.subject,
        predicate=args.predicate,
        value=value,
        scope=args.scope,
        source=args.source,
        confidence=args.confidence,
        sensitivity=args.sensitivity,
        operation=args.operation,
        replaces_fact_id=args.replaces_fact_id,
    )
    print(json.dumps(proposal, ensure_ascii=False, indent=2))
    connection.close()
    return 0


def command_commit(args: argparse.Namespace) -> int:
    connection = connect(args.database)
    result = commit_proposal(connection, args.proposal_id, confirmed=args.confirm)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    connection.close()
    return 0


def command_list(args: argparse.Namespace) -> int:
    connection = connect(args.database)
    print(
        json.dumps(
            list_facts(connection, status=args.status, scope=args.scope),
            ensure_ascii=False,
            indent=2,
        )
    )
    connection.close()
    return 0


def command_handoff(args: argparse.Namespace) -> int:
    connection = connect(args.database)
    result = save_handoff(
        connection,
        title=args.title,
        project=args.project,
        result=args.result,
        decisions=json.loads(args.decisions_json),
        changed=json.loads(args.changed_json),
        verified=json.loads(args.verified_json),
        next_steps=json.loads(args.next_json),
        open_questions=json.loads(args.open_questions_json),
        source_session=args.source_session,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    connection.close()
    return 0


def command_item(args: argparse.Namespace) -> int:
    connection = connect(args.database)
    result = save_knowledge_item(
        connection,
        item_type=args.item_type,
        title=args.title,
        summary=args.summary,
        content=json.loads(args.content_json),
        scope=args.scope,
        status=args.status,
        source=args.source,
        sensitivity=args.sensitivity,
        tags=json.loads(args.tags_json),
        relations=json.loads(args.relations_json),
        confirmed=args.confirm,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    connection.close()
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Strict durable memory store")
    result.add_argument("--database", type=Path, default=DEFAULT_DB_PATH)
    sub = result.add_subparsers(dest="command", required=True)

    propose = sub.add_parser("propose", help="Validate and stage an atomic fact")
    propose.add_argument("--subject", required=True)
    propose.add_argument("--predicate", required=True)
    propose.add_argument("--value-json", required=True)
    propose.add_argument("--scope", required=True)
    propose.add_argument("--source", required=True)
    propose.add_argument("--confidence", type=float, default=1.0)
    propose.add_argument(
        "--sensitivity",
        choices=["public", "internal", "personal", "sensitive"],
        default="personal",
    )
    propose.add_argument("--operation", choices=["add", "correct"], default="add")
    propose.add_argument("--replaces-fact-id")
    propose.set_defaults(func=command_propose)

    commit = sub.add_parser("commit", help="Commit a staged fact after confirmation")
    commit.add_argument("proposal_id")
    commit.add_argument("--confirm", action="store_true")
    commit.set_defaults(func=command_commit)

    listing = sub.add_parser("list", help="List durable facts")
    listing.add_argument(
        "--status", choices=["active", "superseded", "retracted"], default="active"
    )
    listing.add_argument("--scope")
    listing.set_defaults(func=command_list)

    handoff = sub.add_parser("handoff", help="Save a structured handoff in SQLite")
    handoff.add_argument("--title", required=True)
    handoff.add_argument("--project", default="global")
    handoff.add_argument("--result", required=True)
    handoff.add_argument("--decisions-json", default="[]")
    handoff.add_argument("--changed-json", default="[]")
    handoff.add_argument("--verified-json", default="[]")
    handoff.add_argument("--next-json", default="[]")
    handoff.add_argument("--open-questions-json", default="[]")
    handoff.add_argument("--source-session", default="external-agent")
    handoff.set_defaults(func=command_handoff)

    item = sub.add_parser("item", help="Save a tagged, graph-ready knowledge item")
    item.add_argument("--item-type", required=True)
    item.add_argument("--title", required=True)
    item.add_argument("--summary", required=True)
    item.add_argument("--content-json", required=True)
    item.add_argument("--scope", required=True)
    item.add_argument(
        "--status",
        choices=["idea", "draft", "ready", "published", "archived"],
        default="idea",
    )
    item.add_argument("--source", required=True)
    item.add_argument(
        "--sensitivity",
        choices=["public", "internal", "personal", "sensitive"],
        default="internal",
    )
    item.add_argument("--tags-json", default="[]")
    item.add_argument("--relations-json", default="[]")
    item.add_argument("--confirm", action="store_true")
    item.set_defaults(func=command_item)
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        return args.func(args)
    except MemoryStoreError as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
