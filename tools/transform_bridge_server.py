from __future__ import annotations

import json
import os
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from memory_service import build_context, connect
from memory_store import (
    delete_external_projection,
    ensure_schema,
    reconcile_external_projections,
    upsert_external_projection,
)


ROOT = Path(__file__).resolve().parent.parent
TOKEN = os.environ.get("AI_KNOWLEDGE_BRIDGE_TOKEN", "").strip()
MAX_BODY_BYTES = 64 * 1024


class Handler(BaseHTTPRequestHandler):
    server_version = "AIKnowledgeBridge/1.0"

    def log_message(self, message: str, *args) -> None:
        # Never log request bodies or query text.
        sys.stderr.write(f"bridge {self.command} {self.path} {message % args}\n")

    def send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def authorized(self) -> bool:
        if not TOKEN:
            return True
        return self.headers.get("Authorization", "") == f"Bearer {TOKEN}"

    def read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length < 0 or length > MAX_BODY_BYTES:
            raise ValueError("request body is too large")
        payload = json.loads(self.rfile.read(length) or b"{}")
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        return payload

    def do_GET(self) -> None:
        if self.path == "/health":
            self.send_json(200, {"ok": True, "service": "ai-knowledge-bridge"})
            return
        if not self.authorized():
            self.send_json(401, {"ok": False, "error": "unauthorized"})
            return
        if self.path == "/projection/status":
            connection = connect()
            try:
                ensure_schema(connection)
                row = connection.execute(
                    "SELECT COUNT(*) AS active, COALESCE(MAX(source_event_id),0) AS max_source_event_id, MAX(updated_at) AS updated_at FROM external_projections WHERE status='active'"
                ).fetchone()
                self.send_json(200, {"ok": True, **dict(row)})
            finally:
                connection.close()
            return
        self.send_json(404, {"ok": False, "error": "not found"})

    def do_POST(self) -> None:
        if not self.authorized():
            self.send_json(401, {"ok": False, "error": "unauthorized"})
            return
        try:
            payload = self.read_json()
            if self.path == "/context":
                connection = connect()
                try:
                    packet = build_context(
                        connection,
                        query=str(payload.get("query") or "").strip(),
                        project=str(payload.get("project") or "").strip() or None,
                        intent=str(payload.get("intent") or "").strip() or None,
                        budget=max(300, min(int(payload.get("budget") or 10000), 10000)),
                        limit=max(1, min(int(payload.get("limit") or 12), 20)),
                        write_log=False,
                    )
                    self.send_json(200, packet)
                finally:
                    connection.close()
                return
            if self.path == "/projection/refresh":
                subprocess.run([sys.executable, ROOT / "tools" / "knowledge.py", "sync"], cwd=ROOT, check=True, timeout=180, capture_output=True)
                subprocess.run([sys.executable, ROOT / "tools" / "memory_service.py", "build"], cwd=ROOT, check=True, timeout=300, capture_output=True)
                self.send_json(200, {"ok": True})
                return
            connection = connect()
            try:
                ensure_schema(connection)
                if self.path == "/projection/apply":
                    result = upsert_external_projection(connection, **payload)
                elif self.path == "/projection/delete":
                    result = {"deleted": delete_external_projection(connection, **payload)}
                elif self.path == "/projection/reconcile":
                    result = {"deleted": reconcile_external_projections(connection, **payload)}
                else:
                    self.send_json(404, {"ok": False, "error": "not found"})
                    return
                self.send_json(200, {"ok": True, **result})
            finally:
                connection.close()
        except Exception as error:
            self.send_json(400, {"ok": False, "error": str(error)[:1000]})


def main() -> int:
    host = os.environ.get("AI_KNOWLEDGE_BRIDGE_HOST", "127.0.0.1")
    port = int(os.environ.get("AI_KNOWLEDGE_BRIDGE_PORT", "8787"))
    if host not in {"127.0.0.1", "::1", "localhost"} and not TOKEN:
        raise RuntimeError("AI_KNOWLEDGE_BRIDGE_TOKEN is required for a non-loopback bridge")
    server = ThreadingHTTPServer((host, port), Handler)
    print(json.dumps({"ok": True, "host": host, "port": port}), flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
