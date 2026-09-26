from __future__ import annotations

import json
import sys

from memory_service import build_context, connect


if hasattr(sys.stdin, "reconfigure"):
    sys.stdin.reconfigure(encoding="utf-8")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main() -> int:
    payload = json.load(sys.stdin)
    if not isinstance(payload, dict):
        raise ValueError("stdin must contain one JSON object")
    query = str(payload.get("query") or "").strip()
    if not query or len(query) > 2000:
        raise ValueError("query must contain 1 to 2000 characters")
    budget = max(300, min(int(payload.get("budget") or 10000), 10000))
    limit = max(1, min(int(payload.get("limit") or 12), 20))
    project = str(payload.get("project") or "").strip() or None
    intent = str(payload.get("intent") or "").strip() or None
    connection = connect()
    try:
        packet = build_context(
            connection,
            query=query,
            project=project,
            intent=intent,
            budget=budget,
            limit=limit,
            write_log=False,
        )
        print(json.dumps(packet, ensure_ascii=False, separators=(",", ":")))
        return 0
    finally:
        connection.close()


if __name__ == "__main__":
    raise SystemExit(main())
