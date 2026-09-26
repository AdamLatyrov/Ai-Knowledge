"""Deterministic lint for durable knowledge quality and likely noise."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from memory_store import ensure_schema


def run(database: str | Path) -> dict[str, object]:
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    ensure_schema(connection)
    warnings: list[dict[str, str]] = []

    facts = connection.execute(
        "SELECT id, subject, predicate, value_text, scope, source, confidence "
        "FROM memory_facts WHERE status = 'active'"
    ).fetchall()
    for row in facts:
        if len(row["value_text"].strip()) < 3:
            warnings.append({"kind": "weak_fact", "id": row["id"], "reason": "value is too short"})
        if not row["source"].strip():
            warnings.append({"kind": "missing_source", "id": row["id"], "reason": "fact has no source"})

    items = connection.execute(
        "SELECT id, title, summary, content_json, scope, source FROM memory_items "
        "WHERE status <> 'archived'"
    ).fetchall()
    for row in items:
        if len(row["summary"].strip()) < 12:
            warnings.append({"kind": "weak_item", "id": row["id"], "reason": "summary is too short"})
        if len(row["content_json"].encode("utf-8")) <= 2:
            warnings.append({"kind": "empty_item", "id": row["id"], "reason": "content is empty"})
        if not row["source"].strip():
            warnings.append({"kind": "missing_source", "id": row["id"], "reason": "item has no source"})

    duplicates = connection.execute(
        "SELECT scope, lower(title) AS title, COUNT(*) AS count FROM memory_items "
        "WHERE status <> 'archived' GROUP BY scope, lower(title) HAVING COUNT(*) > 1"
    ).fetchall()
    for row in duplicates:
        warnings.append({
            "kind": "duplicate_title",
            "id": f"{row['scope']}::{row['title']}",
            "reason": f"{row['count']} active items share the same title",
        })

    handoffs = connection.execute(
        "SELECT id, result, source_session FROM memory_handoffs"
    ).fetchall()
    for row in handoffs:
        if len(row["result"].strip()) < 20:
            warnings.append({"kind": "weak_handoff", "id": row["id"], "reason": "result is too short"})
        if not row["source_session"].strip():
            warnings.append({"kind": "missing_source", "id": row["id"], "reason": "handoff has no source session"})

    report = {
        "database": str(database),
        "counts": {"facts": len(facts), "items": len(items), "handoffs": len(handoffs)},
        "warning_count": len(warnings),
        "warnings": warnings,
        "policy": "warnings require review; they do not silently delete knowledge",
    }
    connection.close()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", default=str(Path(__file__).resolve().parents[1] / "knowledge.sqlite3"))
    args = parser.parse_args()
    print(json.dumps(run(args.database), ensure_ascii=False, indent=2))
