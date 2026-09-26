from __future__ import annotations

import json
import os
import queue
import re
import shutil
import subprocess
import threading
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


DEFAULT_SERENA_COMMAND = ("serena",)
DEFAULT_TIMEOUT_SECONDS = 180.0
STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "in",
    "is", "of", "on", "or", "the", "to", "what", "where", "with", "а", "в", "где",
    "для", "и", "как", "на", "по", "это", "что", "через", "из", "или",
}


class SerenaProviderError(RuntimeError):
    """A bounded, user-facing Serena integration failure."""


class SerenaUnavailableError(SerenaProviderError):
    """Serena is not installed or cannot be started."""


def token_estimate(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


def query_terms(query: str, *, maximum: int = 8) -> list[str]:
    terms = re.findall(r"[A-Za-zА-Яа-яЁё0-9_]{3,}", query, flags=re.UNICODE)
    result: list[str] = []
    seen: set[str] = set()
    for term in terms:
        normalized = term.lower()
        if normalized in STOP_WORDS or normalized in seen:
            continue
        seen.add(normalized)
        result.append(term)
        if len(result) >= maximum:
            break
    return result or ["."]


def build_search_pattern(query: str) -> str:
    terms = query_terms(query)
    if terms == ["."]:
        return r"(?s)."
    alternatives = "|".join(re.escape(term) for term in terms)
    return rf"(?i)(?:\b(?:{alternatives})\b)"


def _json_from_text(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        if isinstance(value, dict) and "result" in value:
            return _json_from_text(value["result"])
        return value
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return None
    decoder = json.JSONDecoder()
    for marker in ("{", "["):
        start = text.find(marker)
        if start < 0:
            continue
        try:
            parsed, _ = decoder.raw_decode(text[start:])
            return parsed
        except json.JSONDecodeError:
            continue
    return None


def _tool_text(result: Any) -> Any:
    if not isinstance(result, dict):
        return result
    if result.get("isError"):
        raise SerenaProviderError(str(result.get("content") or "Serena tool returned an error"))
    if result.get("structuredContent") is not None:
        structured = result["structuredContent"]
        if isinstance(structured, dict) and "result" in structured:
            return structured["result"]
        return structured
    content = result.get("content")
    if isinstance(content, list):
        texts = [item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text"]
        return "\n".join(texts)
    return result


class SerenaMcpClient:
    """Minimal line-oriented MCP client for Serena's stdio transport."""

    def __init__(
        self,
        command: Sequence[str],
        project_root: Path,
        *,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.command = tuple(command)
        self.project_root = project_root
        self.timeout = timeout
        self.process: subprocess.Popen[str] | None = None
        self._messages: queue.Queue[dict[str, Any]] = queue.Queue()
        self._reader: threading.Thread | None = None
        self._request_id = 0

    def __enter__(self) -> SerenaMcpClient:
        self.start()
        return self

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        self.close()

    def start(self) -> None:
        if self.process is not None:
            return
        if not self.command:
            raise SerenaUnavailableError("Serena command is not configured")
        executable = shutil.which(self.command[0])
        if executable is None and not Path(self.command[0]).exists():
            raise SerenaUnavailableError(
                f"Serena is not installed or not reachable: {self.command[0]}. "
                "Install the official Serena runtime or set code.serena.command."
            )
        argv = [
            *self.command,
            "start-mcp-server",
            "--project",
            str(self.project_root),
            "--enable-web-dashboard",
            "False",
            "--open-web-dashboard",
            "False",
            "--log-level",
            "ERROR",
        ]
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        env = os.environ.copy()
        command_path = Path(self.command[0])
        if command_path.parent != Path("."):
            env["PATH"] = str(command_path.parent.resolve()) + os.pathsep + env.get("PATH", "")
        try:
            self.process = subprocess.Popen(
                argv,
                cwd=str(self.project_root),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=creationflags,
                env=env,
            )
        except OSError as error:
            raise SerenaUnavailableError(f"Could not start Serena: {error}") from error
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()
        self._request(
            "initialize",
            {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "ai-knowledge", "version": "1.0"},
            },
        )
        self._notify("notifications/initialized", {})

    def _read_loop(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        for line in self.process.stdout:
            if not line.strip():
                continue
            with suppress(json.JSONDecodeError):
                message = json.loads(line)
                if isinstance(message, dict):
                    self._messages.put(message)

    def _notify(self, method: str, params: dict[str, Any]) -> None:
        if self.process is None or self.process.stdin is None:
            raise SerenaProviderError("Serena process is not running")
        self.process.stdin.write(json.dumps({"jsonrpc": "2.0", "method": method, "params": params}) + "\n")
        self.process.stdin.flush()

    def _request(self, method: str, params: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
        if self.process is None or self.process.stdin is None:
            raise SerenaProviderError("Serena process is not running")
        self._request_id += 1
        request_id = self._request_id
        self.process.stdin.write(
            json.dumps({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}) + "\n"
        )
        self.process.stdin.flush()
        deadline = threading.Event()
        # queue.get has a finite wait so a dead Serena process cannot hang the MCP server forever.
        request_timeout = self.timeout if timeout is None else timeout
        waited = 0.0
        while waited < request_timeout:
            if self.process.poll() is not None:
                raise SerenaProviderError(f"Serena exited before responding (code {self.process.returncode})")
            try:
                message = self._messages.get(timeout=min(1.0, request_timeout - waited))
            except queue.Empty:
                waited += 1.0
                continue
            if message.get("id") != request_id:
                continue
            if "error" in message:
                raise SerenaProviderError(str(message["error"]))
            return message
        raise SerenaProviderError(f"Serena timed out while handling {method} after {request_timeout:.0f}s")

    def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        response = self._request(
            "tools/call",
            {"name": name, "arguments": arguments},
            timeout=min(self.timeout, 45.0),
        )
        return _tool_text(response.get("result"))

    def close(self) -> None:
        process, self.process = self.process, None
        if process is None:
            return
        with suppress(Exception):
            if process.stdin:
                process.stdin.close()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            with suppress(Exception):
                process.wait(timeout=2)


@dataclass(frozen=True)
class SerenaProvider:
    command: tuple[str, ...] = DEFAULT_SERENA_COMMAND
    timeout: float = DEFAULT_TIMEOUT_SECONDS
    client_factory: Callable[[Path], Any] | None = None

    def _new_client(self, root: Path) -> Any:
        if self.client_factory is not None:
            return self.client_factory(root)
        return SerenaMcpClient(self.command, root, timeout=self.timeout)

    def _with_client(self, root: Path, callback: Callable[[Any], dict[str, Any]]) -> dict[str, Any]:
        client = self._new_client(root)
        if hasattr(client, "__enter__") and hasattr(client, "__exit__"):
            with client as active:
                return callback(active)
        try:
            return callback(client)
        finally:
            close = getattr(client, "close", None)
            if close is not None:
                close()

    @staticmethod
    def _symbol_items(raw: Any, query: str) -> list[dict[str, Any]]:
        parsed = _json_from_text(raw)
        if isinstance(parsed, dict):
            parsed = parsed.get("symbols", parsed.get("results", []))
        if not isinstance(parsed, list):
            return []
        result: list[dict[str, Any]] = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            path = str(item.get("relative_path") or item.get("path") or "")
            name = str(item.get("name_path") or item.get("name") or "")
            body = item.get("body") or item.get("info") or ""
            if not path or not name:
                continue
            location = item.get("body_location") or {}
            result.append(
                {
                    "path": path,
                    "language": Path(path).suffix.lower().lstrip(".") or "text",
                    "symbol": name,
                    "symbol_kind": str(item.get("kind") or "symbol"),
                    "start_line": int(location.get("start_line", 1)) if isinstance(location, dict) else 1,
                    "end_line": int(location.get("end_line", 1)) if isinstance(location, dict) else 1,
                    "content": str(body) or json.dumps(item, ensure_ascii=False),
                    "query_match": query,
                }
            )
        return result

    @staticmethod
    def _pattern_items(raw: Any) -> list[dict[str, Any]]:
        parsed = _json_from_text(raw)
        if not isinstance(parsed, dict):
            return []
        result: list[dict[str, Any]] = []
        for path, matches in parsed.items():
            if not isinstance(matches, list):
                continue
            for match in matches:
                if isinstance(match, int):
                    content = f"Matched line {match}; Serena returned no surrounding content because the response was shortened."
                    line_numbers = [match]
                else:
                    content = str(match)
                    line_numbers = [int(value) for value in re.findall(r"(?:>|\.\.\.)\s*(\d+):", content)]
                start_line = min(line_numbers) if line_numbers else 1
                end_line = max(line_numbers) if line_numbers else start_line
                result.append(
                    {
                        "path": str(path),
                        "language": Path(str(path)).suffix.lower().lstrip(".") or "text",
                        "symbol": Path(str(path)).name,
                        "symbol_kind": "serena-match",
                        "start_line": start_line,
                        "end_line": end_line,
                        "content": content,
                    }
                )
        return result

    def context(self, root: Path, query: str, *, max_tokens: int = 5000, limit: int = 16) -> dict[str, Any]:
        # One name-path probe keeps the bounded MCP primitive responsive. The pattern
        # search still carries all query terms and returns broader file-level evidence.
        terms = query_terms(query, maximum=1)
        tool_errors: list[str] = []

        def collect(client: Any) -> dict[str, Any]:
            candidates: list[dict[str, Any]] = []
            for term in terms:
                try:
                    raw = client.call_tool(
                        "find_symbol",
                        {
                            "name_path_pattern": term,
                            "substring_matching": True,
                            "include_body": True,
                            "max_matches": max(1, min(4, limit)),
                            "max_answer_chars": max(4000, max_tokens * 8),
                        },
                    )
                    candidates.extend(self._symbol_items(raw, term))
                except SerenaProviderError as error:
                    tool_errors.append(f"find_symbol({term}): {error}")

            try:
                raw_search = client.call_tool(
                    "search_for_pattern",
                    {
                        "substring_pattern": build_search_pattern(query),
                        "context_lines_before": 2,
                        "context_lines_after": 8,
                        "restrict_search_to_code_files": True,
                        "skip_ignored_files": True,
                        "multiline": False,
                        "max_answer_chars": max(50000, max_tokens * 20),
                    },
                )
            except SerenaProviderError as error:
                tool_errors.append(f"search_for_pattern: {error}")
                raw_search = "{}"
            pattern_items = self._pattern_items(raw_search)
            for item in pattern_items:
                if not str(item.get("content", "")).startswith("Matched line "):
                    continue
                try:
                    content = client.call_tool(
                        "read_file",
                        {
                            "relative_path": item["path"],
                            "start_line": max(0, int(item["start_line"]) - 3),
                            "end_line": int(item["end_line"]) + 5,
                            "max_answer_chars": 6000,
                        },
                    )
                    if content:
                        item["content"] = str(content)
                        item["start_line"] = max(1, int(item["start_line"]) - 3)
                except SerenaProviderError:
                    # The match location remains useful even if a single file cannot be read.
                    pass
            candidates.extend(pattern_items)

            seen: set[tuple[str, int, int, str]] = set()
            selected: list[dict[str, Any]] = []
            used = 0
            for rank, item in enumerate(candidates, start=1):
                content = str(item.get("content", "")).strip()
                key = (
                    str(item.get("path", "")),
                    int(item.get("start_line", 1)),
                    int(item.get("end_line", 1)),
                    content,
                )
                if key in seen or not content:
                    continue
                seen.add(key)
                estimate = token_estimate(content)
                if used + estimate > max_tokens:
                    continue
                selected.append({**item, "token_estimate": estimate, "score": 1.0 / rank})
                used += estimate
                if len(selected) >= limit:
                    break
            result = {
                "provider": "serena",
                "method": "Serena MCP symbolic + pattern retrieval",
                "query": query,
                "max_tokens": max_tokens,
                "estimated_tokens": used,
                "items": selected,
            }
            if tool_errors:
                result["tool_errors"] = tool_errors[:8]
            return result

        return self._with_client(root, collect)


def command_available(command: Sequence[str]) -> bool:
    if not command:
        return False
    return shutil.which(command[0]) is not None or Path(command[0]).exists()
