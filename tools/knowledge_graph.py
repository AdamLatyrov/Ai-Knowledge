"""Build a bounded graph view from the canonical knowledge SQLite database."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import sys
from collections import deque
from pathlib import Path
from typing import Any


if hasattr(sys.stdin, "reconfigure"):
    sys.stdin.reconfigure(encoding="utf-8")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


TOKEN_RE = re.compile(r"[\w\-]{2,}", re.UNICODE)
SCOPE_SEPARATOR_RE = re.compile(r"[/\\>|·]+")
GENERIC_SCOPE_KEYS = {
    "",
    "ai-knowledge",
    "default",
    "global",
    "knowledge",
    "personal",
    "test",
    "transform",
    "unknown",
}
GENERIC_DUPLICATE_TITLES = {
    "project history",
    "project memory",
    "readme",
}
GENERIC_SCOPE_PARTS = {
    "01_study",
    "archive",
    "content",
    "content-backlog",
    "grants",
    "products",
}
GRAPH_DERIVATION_VERSION = "scope-hubs-and-title-copies-v4"
RELEVANT_TABLES = (
    "documents",
    "memory_facts",
    "memory_handoffs",
    "memory_items",
    "memory_relations",
    "memory_tags",
    "memory_item_tags",
    "knowledge_objects",
    "knowledge_object_claims",
)


def table_names(connection: sqlite3.Connection) -> set[str]:
    return {
        row[0]
        for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }


def rows(connection: sqlite3.Connection, table: str) -> list[dict[str, Any]]:
    try:
        result = [dict(row) for row in connection.execute(f'SELECT * FROM "{table}"')]
        return sorted(result, key=lambda row: json.dumps(row, ensure_ascii=False, sort_keys=True, default=str))
    except sqlite3.OperationalError:
        return []


def first(row: dict[str, Any], *names: str, default: Any = "") -> Any:
    for name in names:
        if row.get(name) not in (None, ""):
            return row[name]
    return default


def text(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value or "")


def stable_version(connection: sqlite3.Connection, available: set[str]) -> str:
    parts: list[str] = [GRAPH_DERIVATION_VERSION]
    for table in RELEVANT_TABLES:
        if table not in available:
            continue
        for row in rows(connection, table):
            parts.append(table + "|" + json.dumps(row, ensure_ascii=False, sort_keys=True, default=str))
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:20]


def external_id(value: str) -> str:
    value = value.strip()
    if value.startswith(("memory://", "document://", "project://", "tool://", "board://", "entity://")):
        return value
    return "entity://external/" + hashlib.sha256(value.casefold().encode("utf-8")).hexdigest()[:24]


def kind_from_uri(uri: str) -> str:
    if uri.startswith("memory://items/"):
        return "item"
    if uri.startswith("memory://facts/"):
        return "fact"
    if uri.startswith("memory://handoffs/"):
        return "handoff"
    if uri.startswith("document://"):
        return "document"
    if uri.startswith("project://"):
        return "project"
    if uri.startswith("tool://"):
        return "tool"
    if uri.startswith("board://"):
        return "board"
    return "entity"


def scope_group_key(scope: str) -> str:
    parts = [part.strip() for part in SCOPE_SEPARATOR_RE.split(str(scope or "")) if part.strip()]
    meaningful = [part for part in parts if part.casefold().strip("[]") not in GENERIC_SCOPE_KEYS and part.casefold().strip("[]") not in GENERIC_SCOPE_PARTS]
    if not meaningful:
        return ""
    # Nested scopes usually end with the concrete project/module name:
    # Products/NeuroCorp -> NeuroCorp, Transform/KnowledgeControl -> KnowledgeControl.
    # Collection-only suffixes keep the parent: RoutePilot/content -> RoutePilot.
    last = parts[-1]
    if last.casefold().strip("[]") not in GENERIC_SCOPE_PARTS and last.casefold().strip("[]") not in GENERIC_SCOPE_KEYS:
        return last[:80]
    return meaningful[0][:80]


def build_graph(connection: sqlite3.Connection, query: str, depth: int, limit: int) -> dict[str, Any]:
    available = table_names(connection)
    version = stable_version(connection, available)
    node_map: dict[str, dict[str, Any]] = {}
    edge_rows: list[dict[str, Any]] = []

    def add_node(uri: str, label: str, kind: str = "entity", scope: str = "", updated: str = "", search: str = "", preview: str = "") -> str:
        node_id = external_id(uri)
        display_label = (label or uri).strip()
        if kind == "project" and display_label.casefold().startswith("project://"):
            display_label = display_label.split("/", 3)[-1] or display_label
        current = node_map.get(node_id)
        if current is None:
            node_map[node_id] = {
                "id": node_id,
                "label": display_label[:180],
                "kind": kind or kind_from_uri(node_id),
                "scope": str(scope or "")[:160],
                "updated": str(updated or "")[:80],
                "preview": str(preview or "").strip()[:700],
                "_search": " ".join([display_label, label, scope, search]).casefold(),
            }
        else:
            current["_search"] += " " + " ".join([display_label, label, scope, search]).casefold()
            if preview and not current.get("preview"):
                current["preview"] = str(preview).strip()[:700]
            if updated and str(updated) > current.get("updated", ""):
                current["updated"] = str(updated)[:80]
        return node_id

    if "memory_items" in available:
        tags_by_item: dict[str, list[str]] = {}
        if "memory_item_tags" in available and "memory_tags" in available:
            tag_names = {int(row["id"]): str(row.get("name") or "") for row in rows(connection, "memory_tags")}
            for row in rows(connection, "memory_item_tags"):
                tags_by_item.setdefault(str(row.get("item_id")), []).append(tag_names.get(int(row.get("tag_id") or 0), ""))
        for row in rows(connection, "memory_items"):
            item_id = str(first(row, "id"))
            title = text(first(row, "title", default=item_id))
            summary = text(first(row, "summary", "content_json"))
            tags = " ".join(tags_by_item.get(item_id, []))
            add_node(
                f"memory://items/{item_id}",
                title,
                "item",
                text(first(row, "scope")),
                text(first(row, "updated_at", "created_at")),
                f"{summary} {tags}",
                summary,
            )

    if "memory_facts" in available:
        for row in rows(connection, "memory_facts"):
            fact_id = str(first(row, "id"))
            subject = text(first(row, "subject", default=fact_id))
            add_node(
                f"memory://facts/{fact_id}",
                subject,
                "fact",
                text(first(row, "scope")),
                text(first(row, "updated_at", "created_at")),
                text(first(row, "predicate", "value_text", "value_json")),
                text(first(row, "value_text", "value_json", "predicate")),
            )

    if "memory_handoffs" in available:
        for row in rows(connection, "memory_handoffs"):
            handoff_id = str(first(row, "id"))
            title = text(first(row, "title", "result", default=handoff_id))
            add_node(
                f"memory://handoffs/{handoff_id}",
                title,
                "handoff",
                text(first(row, "project", "scope")),
                text(first(row, "updated_at", "created_at")),
                text(first(row, "result", "decisions", "next")),
                text(first(row, "result", "decisions", "next")),
            )

    if "documents" in available:
        for row in rows(connection, "documents"):
            document_id = str(first(row, "id"))
            title = text(first(row, "title", "path", default=document_id))
            add_node(
                f"document://{document_id}",
                title,
                "document",
                text(first(row, "scope", "project")),
                text(first(row, "modified_at", "updated_at")),
                text(first(row, "content"))[:2500],
                text(first(row, "content")),
            )

    if "memory_relations" in available:
        for row in rows(connection, "memory_relations"):
            source_uri = str(first(row, "source_uri"))
            target_uri = str(first(row, "target_uri"))
            if not source_uri or not target_uri:
                continue
            source_id = add_node(source_uri, source_uri, kind_from_uri(source_uri))
            target_id = add_node(target_uri, target_uri, kind_from_uri(target_uri))
            edge_rows.append({
                "id": str(first(row, "id", default=f"{source_id}:{target_id}")),
                "source": source_id,
                "target": target_id,
                "type": str(first(row, "relation_type", default="related-to"))[:80],
                "label": str(first(row, "label", default="Связь"))[:120],
                "inferred": False,
            })

    # Keep the source database read-only, but make the global view useful even
    # when older material has no explicit relation yet. These are intentionally
    # marked as inferred so the UI can show them as softer, dashed links.
    edge_keys = {
        tuple(sorted((edge["source"], edge["target"])))
        for edge in edge_rows
    }

    def add_inferred_edge(source: str, target: str, relation_type: str, label: str) -> None:
        if source == target:
            return
        key = tuple(sorted((source, target)))
        if key in edge_keys:
            return
        edge_keys.add(key)
        edge_rows.append({
            "id": "derived://" + hashlib.sha256((relation_type + "|" + "|".join(key)).encode("utf-8")).hexdigest()[:24],
            "source": source,
            "target": target,
            "type": relation_type,
            "label": label,
            "inferred": True,
        })

    scope_groups: dict[str, list[str]] = {}
    title_groups: dict[str, list[str]] = {}
    scope_display_by_key: dict[str, str] = {}
    for node in list(node_map.values()):
        scope_key = scope_group_key(node.get("scope", ""))
        if scope_key:
            normalized_scope_key = scope_key.casefold()
            scope_groups.setdefault(normalized_scope_key, []).append(node["id"])
            scope_display_by_key.setdefault(normalized_scope_key, scope_key)
        title_key = re.sub(r"[_-]+", " ", str(node.get("label", "")).strip())
        title_key = re.sub(r"\s+", " ", title_key).casefold()
        if len(title_key) >= 4 and title_key not in GENERIC_DUPLICATE_TITLES:
            title_groups.setdefault(title_key, []).append(node["id"])

    # A scope hub gives a coherent place for old RoutePilot/KnowledgeControl
    # materials without inventing a semantic relation between every pair.
    project_by_label = {
        str(node.get("label", "")).strip().casefold(): node["id"]
        for node in node_map.values()
        if node.get("kind") == "project"
    }
    for node in node_map.values():
        if node.get("kind") != "project":
            continue
        label = str(node.get("label", "")).strip().casefold()
        if label.startswith("project://"):
            project_by_label.setdefault(label.split("/", 3)[-1], node["id"])
    for scope_key, member_ids in sorted(scope_groups.items()):
        if len(member_ids) < 2:
            continue
        display_scope = scope_display_by_key.get(scope_key, scope_key)
        hub_id = project_by_label.get(scope_key)
        if not hub_id:
            slug = re.sub(r"[^\w\-]+", "-", scope_key, flags=re.UNICODE).strip("-") or hashlib.sha256(scope_key.encode("utf-8")).hexdigest()[:12]
            hub_id = add_node(f"project://derived/{slug}", display_scope, "project", display_scope)
            node_map[hub_id]["derived"] = True
            project_by_label[scope_key] = hub_id
        for member_id in member_ids:
            add_inferred_edge(member_id, hub_id, "inferred-scope", "Общая область")

    # Exact title matches are the safe way to join the common item/document/
    # handoff copies of one material. Similar-but-not-equal titles stay apart.
    for member_ids in title_groups.values():
        if len(member_ids) < 2 or len({node_map[node_id].get("kind") for node_id in member_ids}) < 2:
            continue
        anchor, *duplicates = sorted(set(member_ids))
        for duplicate in duplicates:
            add_inferred_edge(anchor, duplicate, "same-material", "Один материал")

    # Node aliases and relation targets are allowed to be unresolved external entities.
    tokens = [token.casefold() for token in TOKEN_RE.findall(query)]
    if not tokens:
        ordered_ids = [node["id"] for node in sorted(node_map.values(), key=lambda item: (item.get("updated", ""), item["id"]), reverse=True)]
        nodes = []
        for node_id in ordered_ids[:limit]:
            node = dict(node_map[node_id])
            node.pop("_search", None)
            node.pop("preview", None)
            node["selected"] = False
            nodes.append(node)
        visible = {node["id"] for node in nodes}
        edges = [edge for edge in edge_rows if edge["source"] in visible and edge["target"] in visible]
        return {"version": version, "nodes": nodes, "edges": edges[: max(limit * 2, 20)]}
    ranked = []
    for node in node_map.values():
        score = sum(3 if token in node["label"].casefold() else 1 for token in tokens if token in node["_search"])
        if score:
            ranked.append((score, node.get("updated", ""), node["id"]))
    ranked.sort(reverse=True)
    seeds = [node_id for _, _, node_id in ranked[:12]]
    if not seeds:
        seeds = [node["id"] for node in sorted(node_map.values(), key=lambda item: item.get("updated", ""), reverse=True)[:12]]

    adjacency: dict[str, list[str]] = {}
    for edge in edge_rows:
        adjacency.setdefault(edge["source"], []).append(edge["target"])
        adjacency.setdefault(edge["target"], []).append(edge["source"])
    selected = set(seeds)
    queue = deque((seed, 0) for seed in seeds)
    while queue:
        node_id, distance = queue.popleft()
        if distance >= depth:
            continue
        for neighbour in adjacency.get(node_id, []):
            if neighbour in selected:
                continue
            selected.add(neighbour)
            queue.append((neighbour, distance + 1))

    nodes = []
    ordered_ids = list(seeds) + [node_id for node_id in selected if node_id not in seeds]
    for node_id in ordered_ids:
        if node_id not in node_map or len(nodes) >= limit:
            continue
        node = dict(node_map[node_id])
        node.pop("_search", None)
        node.pop("preview", None)
        node["selected"] = node_id in seeds
        nodes.append(node)
    visible = {node["id"] for node in nodes}
    edges = [edge for edge in edge_rows if edge["source"] in visible and edge["target"] in visible]
    return {"version": version, "nodes": nodes, "edges": edges[: max(limit * 2, 20)]}


def read_only_connection(database: str) -> sqlite3.Connection:
    database_uri = Path(database).expanduser().resolve().as_uri() + "?mode=ro"
    return sqlite3.connect(database_uri, uri=True)


def main() -> int:
    payload = json.load(sys.stdin)
    database = str(payload.get("database") or "").strip()
    if not database:
        raise ValueError("database is required")
    connection = read_only_connection(database)
    connection.row_factory = sqlite3.Row
    try:
        available = table_names(connection)
        version = stable_version(connection, available)
        if payload.get("op") == "version":
            print(json.dumps({"version": version}, ensure_ascii=False, separators=(",", ":")))
        else:
            mode = payload.get("mode") or ("all" if not payload.get("query") else "search")
            if mode not in {"all", "search"}:
                raise ValueError("mode must be all or search")
            raw_query = payload.get("query", "")
            if not isinstance(raw_query, str):
                raise ValueError("query must be a string")
            query = raw_query.strip()
            if mode == "search" and not query or len(query) > 2000:
                raise ValueError("query must contain 1 to 2000 characters")
            raw_depth = payload.get("depth", 1)
            raw_limit = payload.get("limit", 800 if mode == "all" else 60)
            if not isinstance(raw_depth, int) or isinstance(raw_depth, bool) or not 1 <= raw_depth <= 2:
                raise ValueError("depth must be an integer from 1 to 2")
            if not isinstance(raw_limit, int) or isinstance(raw_limit, bool) or not 10 <= raw_limit <= 800:
                raise ValueError("limit must be an integer from 10 to 800")
            graph = build_graph(
                connection,
                query,
                raw_depth,
                raw_limit,
            )
            print(json.dumps(graph, ensure_ascii=False, separators=(",", ":")))
        return 0
    finally:
        connection.close()


if __name__ == "__main__":
    raise SystemExit(main())
