from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from memory_store import (
    DEFAULT_DB_PATH,
    connect,
    delete_external_projection,
    reconcile_external_projections,
    upsert_external_projection,
)


if hasattr(sys.stdin, "reconfigure"):
    sys.stdin.reconfigure(encoding="utf-8")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def read_payload() -> dict:
    payload = json.load(sys.stdin)
    if not isinstance(payload, dict):
        raise ValueError("stdin must contain one JSON object")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply rebuildable Transform projections")
    parser.add_argument("command", choices=("apply", "delete", "reconcile", "status"))
    parser.add_argument("--database", type=Path, default=DEFAULT_DB_PATH)
    args = parser.parse_args()

    connection = connect(args.database)
    try:
        if args.command == "apply":
            result = upsert_external_projection(connection, **read_payload())
        elif args.command == "delete":
            result = {"deleted": delete_external_projection(connection, **read_payload())}
        elif args.command == "reconcile":
            result = {"deleted": reconcile_external_projections(connection, **read_payload())}
        else:
            row = connection.execute(
                """
                SELECT COUNT(*) AS active,
                       COALESCE(MAX(source_event_id), 0) AS max_source_event_id,
                       MAX(updated_at) AS updated_at
                FROM external_projections WHERE status='active'
                """
            ).fetchone()
            result = dict(row)
        print(json.dumps({"ok": True, **result}, ensure_ascii=False))
        return 0
    except Exception as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 1
    finally:
        connection.close()


if __name__ == "__main__":
    raise SystemExit(main())
