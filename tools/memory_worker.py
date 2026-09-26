from __future__ import annotations

import json
import os
import sys
import time
from typing import Any

import memory_service as memory
import module_runtime
import agent_modes
import corpus_service as corpus


def fit_runtime_budget(packet: dict[str, Any], budget: int) -> dict[str, Any]:
    """Keep the combined memory + runtime envelope within one token budget."""
    def estimate() -> int:
        return memory.token_estimate(json.dumps(packet, ensure_ascii=False, separators=(",", ":")))

    def stabilize_usage() -> int:
        estimate_value = 0
        for _ in range(10):
            estimate_value = estimate()
            packet.setdefault("usage", {})["runtime_serialized_estimated_tokens"] = estimate_value
            packet["estimated_tokens"] = estimate_value
            stabilized = estimate()
            if stabilized == estimate_value:
                return stabilized
            estimate_value = stabilized
        return estimate_value

    runtime = packet.get("runtime", {})
    while stabilize_usage() > budget:
        if runtime.get("objects"):
            runtime["objects"].pop()
            continue
        if runtime.get("modes"):
            runtime["modes"].pop()
            continue
        if runtime.get("modules"):
            runtime["modules"].pop()
            continue
        if packet.get("evidence"):
            packet["evidence"].pop()
            continue
        break
    stabilize_usage()
    packet.setdefault("quality", {})["within_budget"] = packet["estimated_tokens"] <= budget
    return packet


if hasattr(sys.stdin, "reconfigure"):
    sys.stdin.reconfigure(encoding="utf-8")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def dispatch(operation: str, args: dict[str, Any]) -> Any:
    connection = memory.connect()
    try:
        if operation == "context":
            return memory.build_context(
                connection,
                query=args["query"],
                project=args.get("project"),
                intent=args.get("intent"),
                budget=int(args.get("maxTokens", 5000)),
                limit=int(args.get("limit", 20)),
            )
        if operation == "runtime_context":
            started = time.perf_counter()
            query = args["query"]
            plan = module_runtime.build_runtime_plan(
                connection,
                query=query,
                module=args.get("module"),
                mode=args.get("mode"),
                depth=args.get("depth"),
            )
            retrieval_scope = module_runtime.module_retrieval_scope(
                connection, plan.get("module")
            )
            context = memory.build_context(
                connection,
                query=query,
                project=retrieval_scope or args.get("project"),
                intent=args.get("intent"),
                budget=int(args.get("maxTokens", 3000)),
                limit=int(args.get("limit", 20)),
            )
            available_modes = agent_modes.list_modes(memory.ROOT).get("available_modes", [])
            mode_candidates = module_runtime.matching_modes(available_modes, query)
            context["runtime"] = {
                "plan": plan,
                "objects": module_runtime.object_context(
                    connection,
                    module_id=plan.get("module"),
                    tags=plan["tags"],
                    depth=plan["depth"],
                    limit=int(args.get("objectLimit", 12)),
                ),
                "modules": module_runtime.list_modules(connection, ""),
                "modes": mode_candidates,
                "mode_catalog_count": len(available_modes),
                "proposal": (
                    {
                        "type": "module",
                        "status": "not_found",
                        "query": query,
                        "suggestion": "Offer a module proposal; do not activate it without confirmation.",
                    }
                    if plan["needs_module_proposal"]
                    else None
                ),
            }
            context["runtime"]["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
            return fit_runtime_budget(context, int(args.get("maxTokens", 3000)))
        if operation == "module_activate":
            result = module_runtime.activate_module(
                connection,
                module_id=args["moduleId"],
                title=args["title"],
                description=args["description"],
                aliases=list(args.get("aliases", [])),
                retrieval_scope=args.get("retrievalScope"),
                behavior_prompt_path=args.get("behaviorPromptPath"),
                schema=args.get("schema"),
                confirm=bool(args.get("confirm", False)),
            )
            return result
        if operation == "search":
            started = time.perf_counter()
            rows = memory.hybrid_results(
                connection,
                args["query"],
                int(args.get("limit", 10)),
                args.get("project"),
                args.get("intent"),
            )
            packet = memory.build_search_packet(
                args["query"], args.get("project"), args.get("intent"), rows
            )
            memory.record_retrieval_log(
                connection,
                operation="search",
                query=args["query"],
                project=args.get("project"),
                intent=args.get("intent"),
                budget=None,
                packet=packet,
                latency_ms=(time.perf_counter() - started) * 1000,
            )
            return packet
        if operation == "write_context":
            started = time.perf_counter()
            packet = memory.build_write_context(
                connection,
                request=args["request"],
                material_summary=args.get("materialSummary", ""),
                material_chars=int(args.get("materialChars", 0)),
                project=args.get("project"),
                budget=int(args.get("maxTokens", 1800)),
                limit=int(args.get("limit", 8)),
            )
            sources = {
                f"s{index}": item.get("source", item.get("uri", ""))
                for index, item in enumerate(
                    [*packet.get("context", []), *packet.get("similar_items", [])], start=1
                )
                if item.get("source") or item.get("uri")
            }
            log_packet = {
                "usage": {"serialized_estimated_tokens": packet.get("estimated_tokens")},
                "core": [],
                "current": [],
                "evidence": [{}] * (
                    len(packet.get("context", [])) + len(packet.get("similar_items", []))
                ),
                "sources": sources,
                "quality": packet.get("quality", {}),
            }
            memory.record_retrieval_log(
                connection,
                operation="write_context",
                query=f"{args['request']} {args.get('materialSummary', '')}".strip(),
                project=args.get("project"),
                intent="artifact",
                budget=int(args.get("maxTokens", 1800)),
                packet=log_packet,
                latency_ms=(time.perf_counter() - started) * 1000,
            )
            return packet
        if operation == "metadata_suggest":
            return memory.save_metadata_suggestion(
                connection,
                source_uri=args["sourceUri"],
                source_hash=args["sourceHash"],
                scope=args["scope"],
                document_type=args["documentType"],
                tags=list(args.get("tags", [])),
                relations=list(args.get("relations", [])),
                confidence=float(args["confidence"]),
                model=args.get("model", "agent:memory-curator"),
            )
        if operation == "metadata_commit":
            return memory.commit_metadata_suggestion(
                connection,
                args["suggestionId"],
                confirm=bool(args.get("confirm", False)),
            )
        if operation == "metadata_list":
            return {
                "suggestions": memory.list_metadata_suggestions(
                    connection,
                    status=args.get("status"),
                    limit=int(args.get("limit", 50)),
                )
            }
        if operation == "status":
            return memory.status_packet(connection)
        if operation == "logs":
            return {
                "logs": memory.list_retrieval_logs(
                    connection,
                    limit=int(args.get("limit", 50)),
                    project=args.get("project"),
                    operation=args.get("operation"),
                )
            }
        if operation == "corpus_status":
            return corpus.CorpusStore(connection).status()
        if operation == "corpus_search":
            return {
                "query": args["query"],
                "results": corpus.CorpusStore(connection).search(
                    args["query"],
                    limit=int(args.get("limit", 8)),
                    source_id=args.get("sourceId"),
                ),
            }
        if operation == "corpus_ingest":
            transcript_path = corpus.allowed_corpus_path(args["transcriptPath"])
            video_path = corpus.allowed_corpus_path(args.get("videoPath", ""), allow_empty=True)
            transcript_text = transcript_path.read_text(encoding="utf-8-sig")
            return corpus.CorpusStore(connection).ingest_source(
                {
                    "source_id": args["sourceId"],
                    "title": args["title"],
                    "source_type": args.get("sourceType", "youtube"),
                    "source_uri": args.get("sourceUri", ""),
                    "scope": args.get("scope", "ResearchCorpus/YouTube"),
                    "transcript_path": str(transcript_path),
                    "video_path": str(video_path) if video_path else "",
                    "language": args.get("language", "ru"),
                    "metadata": args.get("metadata", {}),
                },
                transcript_text,
            )
        raise ValueError(f"Unknown memory worker operation: {operation}")
    finally:
        connection.close()


def main() -> int:
    for line in sys.stdin:
        if not line.strip():
            continue
        request_id = None
        try:
            request = json.loads(line)
            request_id = request.get("id")
            result = dispatch(request["op"], request.get("args", {}))
            response = {
                "id": request_id,
                "ok": True,
                "worker_pid": os.getpid(),
                "result": result,
            }
        except Exception as error:  # The protocol returns a bounded error, never a traceback.
            response = {
                "id": request_id,
                "ok": False,
                "worker_pid": os.getpid(),
                "error": str(error)[:2000],
            }
        print(json.dumps(response, ensure_ascii=False, separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
