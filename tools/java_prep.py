from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import sqlite3
import sys
import uuid
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent.parent
CONFIG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
DEFAULT_DB_PATH = Path(
    os.environ.get("AI_KNOWLEDGE_DATABASE", ROOT / CONFIG["database"])
).expanduser().resolve()
SCOPE = "Senior Java Preparation"
SENSITIVITY = "personal"

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from memory_store import (  # noqa: E402
    ConfirmationRequired,
    SecretRejected,
    ensure_schema as ensure_memory_schema,
    secret_like,
    save_handoff,
)


GAP_TYPES = {
    "THEORY_GAP",
    "EXECUTION_MODEL_GAP",
    "API_SYNTAX_GAP",
    "CODING_GAP",
    "PRODUCTION_GAP",
    "INTERVIEW_EXPRESSION_GAP",
    "FRAMEWORK_CORE_GAP",
}
QUESTION_TYPES = {
    "THEORY",
    "INTERNALS",
    "TRICKY",
    "CODE_OUTPUT",
    "COMPILE",
    "RUNTIME",
    "EDGE_CASE",
    "PRODUCTION",
    "CODING",
    "SYSTEM_DESIGN",
}
MODES = {"TRAIN", "RETEST", "MOCK", "BANK", "STATUS"}


SCHEMA = """
CREATE TABLE IF NOT EXISTS prep_schema_migrations (
    version INTEGER PRIMARY KEY,
    description TEXT NOT NULL,
    applied_at TEXT NOT NULL
) STRICT;

CREATE TABLE IF NOT EXISTS prep_topics (
    id TEXT PRIMARY KEY,
    scope TEXT NOT NULL,
    name TEXT NOT NULL,
    slug TEXT NOT NULL,
    path TEXT NOT NULL,
    parent_id TEXT,
    source TEXT NOT NULL,
    sensitivity TEXT NOT NULL CHECK(sensitivity IN ('public', 'internal', 'personal', 'sensitive')),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(scope, path),
    FOREIGN KEY(parent_id) REFERENCES prep_topics(id)
) STRICT;
CREATE INDEX IF NOT EXISTS idx_prep_topics_parent ON prep_topics(parent_id, path);

CREATE TABLE IF NOT EXISTS prep_topic_prerequisites (
    topic_id TEXT NOT NULL,
    prerequisite_topic_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY(topic_id, prerequisite_topic_id),
    CHECK(topic_id <> prerequisite_topic_id),
    FOREIGN KEY(topic_id) REFERENCES prep_topics(id) ON DELETE CASCADE,
    FOREIGN KEY(prerequisite_topic_id) REFERENCES prep_topics(id) ON DELETE CASCADE
) STRICT;

CREATE TABLE IF NOT EXISTS prep_topic_state (
    topic_id TEXT PRIMARY KEY,
    current_score REAL CHECK(current_score >= 0 AND current_score <= 5),
    confidence REAL NOT NULL DEFAULT 0 CHECK(confidence >= 0 AND confidence <= 1),
    status TEXT NOT NULL DEFAULT 'UNTESTED' CHECK(status IN ('UNTESTED', 'OPEN', 'LEARNING', 'RETEST', 'STABLE')),
    last_tested TEXT,
    attempts_count INTEGER NOT NULL DEFAULT 0 CHECK(attempts_count >= 0),
    weaknesses_json TEXT NOT NULL DEFAULT '[]',
    gap_types_json TEXT NOT NULL DEFAULT '[]',
    next_review TEXT,
    stability REAL NOT NULL DEFAULT 0 CHECK(stability >= 0),
    updated_at TEXT NOT NULL,
    FOREIGN KEY(topic_id) REFERENCES prep_topics(id) ON DELETE CASCADE
) STRICT;
CREATE INDEX IF NOT EXISTS idx_prep_topic_state_due ON prep_topic_state(next_review, current_score);

CREATE TABLE IF NOT EXISTS prep_questions (
    id TEXT PRIMARY KEY,
    topic_id TEXT NOT NULL,
    question_text TEXT NOT NULL,
    compact_text TEXT NOT NULL,
    question_type TEXT NOT NULL CHECK(question_type IN ('THEORY', 'INTERNALS', 'TRICKY', 'CODE_OUTPUT', 'COMPILE', 'RUNTIME', 'EDGE_CASE', 'PRODUCTION', 'CODING', 'SYSTEM_DESIGN')),
    difficulty INTEGER NOT NULL CHECK(difficulty BETWEEN 1 AND 5),
    source TEXT NOT NULL,
    signature TEXT NOT NULL,
    created_at TEXT NOT NULL,
    last_asked_at TEXT,
    UNIQUE(topic_id, signature),
    FOREIGN KEY(topic_id) REFERENCES prep_topics(id) ON DELETE CASCADE
) STRICT;
CREATE INDEX IF NOT EXISTS idx_prep_questions_topic_type ON prep_questions(topic_id, question_type, difficulty);

CREATE TABLE IF NOT EXISTS question_bank (
    id TEXT PRIMARY KEY,
    topic_id TEXT NOT NULL,
    question_text TEXT NOT NULL,
    compact_text TEXT NOT NULL,
    question_type TEXT NOT NULL CHECK(question_type IN ('THEORY', 'INTERNALS', 'TRICKY', 'CODE_OUTPUT', 'COMPILE', 'RUNTIME', 'EDGE_CASE', 'PRODUCTION', 'CODING', 'SYSTEM_DESIGN')),
    difficulty INTEGER NOT NULL CHECK(difficulty BETWEEN 1 AND 5),
    source TEXT NOT NULL,
    signature TEXT NOT NULL,
    created_at TEXT NOT NULL,
    last_asked_at TEXT,
    UNIQUE(topic_id, signature),
    FOREIGN KEY(topic_id) REFERENCES prep_topics(id) ON DELETE CASCADE
) STRICT;
CREATE INDEX IF NOT EXISTS idx_question_bank_topic_type ON question_bank(topic_id, question_type, difficulty);

CREATE TABLE IF NOT EXISTS prep_sessions (
    id TEXT PRIMARY KEY,
    mode TEXT NOT NULL CHECK(mode IN ('TRAIN', 'RETEST', 'MOCK', 'BANK', 'STATUS', 'LEGACY')),
    scope TEXT NOT NULL,
    source TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('ACTIVE', 'COMPLETED')),
    summary TEXT NOT NULL DEFAULT '',
    started_at TEXT NOT NULL,
    completed_at TEXT,
    handoff_id TEXT
) STRICT;
CREATE INDEX IF NOT EXISTS idx_prep_sessions_started ON prep_sessions(started_at DESC);

CREATE TABLE IF NOT EXISTS prep_attempts (
    id TEXT PRIMARY KEY,
    session_id TEXT,
    question_id TEXT NOT NULL,
    topic_id TEXT NOT NULL,
    score INTEGER NOT NULL CHECK(score BETWEEN 0 AND 5),
    strengths_json TEXT NOT NULL,
    mistakes_json TEXT NOT NULL,
    gap_types_json TEXT NOT NULL,
    answer_summary TEXT NOT NULL,
    hint_used INTEGER NOT NULL CHECK(hint_used IN (0, 1)),
    source_ref TEXT UNIQUE,
    attempted_at TEXT NOT NULL,
    FOREIGN KEY(session_id) REFERENCES prep_sessions(id),
    FOREIGN KEY(question_id) REFERENCES question_bank(id),
    FOREIGN KEY(topic_id) REFERENCES prep_topics(id)
) STRICT;
CREATE INDEX IF NOT EXISTS idx_prep_attempts_topic_date ON prep_attempts(topic_id, attempted_at DESC);
CREATE INDEX IF NOT EXISTS idx_prep_attempts_question_date ON prep_attempts(question_id, attempted_at DESC);

CREATE TABLE IF NOT EXISTS prep_imports (
    source_path TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL,
    imported_at TEXT NOT NULL,
    result_json TEXT NOT NULL
) STRICT;

CREATE TABLE IF NOT EXISTS prep_question_rejections (
    id TEXT PRIMARY KEY,
    question_text TEXT NOT NULL,
    topic_path TEXT NOT NULL,
    source TEXT NOT NULL,
    reasons_json TEXT NOT NULL,
    created_at TEXT NOT NULL
) STRICT;
CREATE INDEX IF NOT EXISTS idx_prep_question_rejections_created
ON prep_question_rejections(created_at DESC);
"""


TOPIC_PATHS = (
    "Java",
    "Java/Java Core",
    "Java/Collections",
    "Java/Collections/HashMap",
    "Java/Collections/HashMap/Collisions",
    "Java/Collections/HashMap/Resize",
    "Java/Collections/HashMap/Treeification",
    "Java/Generics",
    "Java/Generics/Type Erasure",
    "Java/Generics/Bridge Methods",
    "Java/Generics/Wildcard Capture",
    "Java/Exceptions",
    "Java/Streams",
    "Java/JVM",
    "Java/GC",
    "Java/Concurrency",
    "Java/Concurrency/JMM",
    "Java/Concurrency/JMM/Happens-Before",
    "Java/Concurrency/ExecutorService",
    "Java/Concurrency/CompletableFuture",
    "Spring",
    "Spring/Spring Core",
    "Spring/Spring Boot",
    "Spring/Spring MVC",
    "Spring/Transactions",
    "Spring/Transactions/Propagation",
    "Spring/Transactions/REQUIRES_NEW",
    "Persistence",
    "Persistence/JPA-Hibernate",
    "Persistence/SQL-PostgreSQL",
    "Messaging",
    "Messaging/Kafka",
    "Backend",
    "Backend/HTTP-REST",
    "Backend/Testing",
    "Backend/Distributed Systems",
    "Backend/Microservices",
    "Backend/Docker-Kubernetes",
    "Backend/System Design",
    "Coding",
    "Coding/Collections",
    "Coding/Generics",
    "Coding/Streams",
    "Coding/Concurrency",
    "Coding/Transactions",
    "Coding/PostgreSQL",
    "Coding/Kafka",
    "Coding/Caching",
    "Coding/Retries",
    "Coding/Idempotency",
    "Coding/Batch Processing",
    "Coding/Deduplication",
    "Algorithms",
    "Algorithms/HashMap-HashSet Lookup",
    "Algorithms/Frequency Counting",
    "Algorithms/Two Pointers",
    "Algorithms/Slow-Fast Pointers",
    "Algorithms/Fixed Sliding Window",
    "Algorithms/Variable Sliding Window",
    "Algorithms/Prefix Sum",
    "Algorithms/Binary Search",
    "Algorithms/Binary Search on Answer",
    "Algorithms/Stack",
    "Algorithms/Monotonic Stack",
    "Algorithms/Heap",
    "Algorithms/Intervals",
    "Algorithms/BFS",
    "Algorithms/DFS",
    "Algorithms/Trees",
    "Algorithms/Backtracking",
    "Algorithms/Dynamic Programming",
)


LEGACY_TOPIC_MAP = {
    "java core": "Java/Java Core",
    "collections": "Java/Collections",
    "generics": "Java/Generics",
    "exceptions": "Java/Exceptions",
    "streams": "Java/Streams",
    "jvm": "Java/JVM",
    "gc": "Java/GC",
    "jmm": "Java/Concurrency/JMM",
    "concurrency": "Java/Concurrency",
    "spring core": "Spring/Spring Core",
    "spring boot": "Spring/Spring Boot",
    "spring mvc": "Spring/Spring MVC",
    "transactions": "Spring/Transactions",
    "jpa/hibernate": "Persistence/JPA-Hibernate",
    "sql/postgresql": "Persistence/SQL-PostgreSQL",
    "kafka": "Messaging/Kafka",
    "http/rest": "Backend/HTTP-REST",
    "testing": "Backend/Testing",
    "distributed systems": "Backend/Distributed Systems",
    "microservices": "Backend/Microservices",
    "docker/kubernetes": "Backend/Docker-Kubernetes",
    "system design": "Backend/System Design",
    "coding": "Coding",
}


JAVA_KEYWORDS = {
    "abstract", "assert", "boolean", "break", "byte", "case", "catch", "char",
    "class", "const", "continue", "default", "do", "double", "else", "enum",
    "extends", "final", "finally", "float", "for", "goto", "if", "implements",
    "import", "instanceof", "int", "interface", "long", "native", "new", "package",
    "private", "protected", "public", "return", "short", "static", "strictfp",
    "super", "switch", "synchronized", "this", "throw", "throws", "transient",
    "try", "void", "volatile", "while", "var", "record", "sealed", "permits",
    "true", "false", "null", "String", "Integer", "Long", "Double", "List",
    "Map", "Set", "System", "Optional", "Stream", "Object", "Thread",
}


def now_iso() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def connect(path: str | Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    connection = sqlite3.connect(Path(path))
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    ensure_memory_schema(connection)
    ensure_schema(connection)
    return connection


def ensure_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(SCHEMA)
    connection.execute(
        "INSERT OR IGNORE INTO prep_schema_migrations(version, description, applied_at) VALUES (1, ?, ?)",
        ("Senior Java preparation core schema", now_iso()),
    )
    connection.commit()


def _slug(value: str) -> str:
    value = re.sub(r"[^\w]+", "-", value.strip().lower(), flags=re.UNICODE)
    return value.strip("-")


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _loads(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def add_topic(
    connection: sqlite3.Connection,
    path: str,
    *,
    source: str = "system:java-preparation",
) -> dict[str, Any]:
    clean = "/".join(part.strip() for part in path.split("/") if part.strip())
    if not clean:
        raise ValueError("topic path is required")
    parent_path = clean.rsplit("/", 1)[0] if "/" in clean else None
    parent_id = None
    if parent_path:
        parent_id = add_topic(connection, parent_path, source=source)["id"]
    existing = connection.execute(
        "SELECT id FROM prep_topics WHERE scope = ? AND path = ?", (SCOPE, clean)
    ).fetchone()
    timestamp = now_iso()
    if existing:
        return get_topic(connection, clean)
    topic_id = uuid.uuid4().hex
    connection.execute(
        """
        INSERT INTO prep_topics(
            id, scope, name, slug, path, parent_id, source, sensitivity, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            topic_id,
            SCOPE,
            clean.rsplit("/", 1)[-1],
            _slug(clean.rsplit("/", 1)[-1]),
            clean,
            parent_id,
            source,
            SENSITIVITY,
            timestamp,
            timestamp,
        ),
    )
    connection.execute(
        "INSERT INTO prep_topic_state(topic_id, updated_at) VALUES (?, ?)",
        (topic_id, timestamp),
    )
    connection.commit()
    return get_topic(connection, clean)


def seed_default_topics(connection: sqlite3.Connection) -> int:
    before = connection.execute("SELECT COUNT(*) FROM prep_topics").fetchone()[0]
    for path in TOPIC_PATHS:
        add_topic(connection, path)
    return connection.execute("SELECT COUNT(*) FROM prep_topics").fetchone()[0] - before


def _topic_row(row: sqlite3.Row) -> dict[str, Any]:
    result = dict(row)
    return result


def get_topic(connection: sqlite3.Connection, path_or_name: str) -> dict[str, Any]:
    exact = connection.execute(
        """
        SELECT t.*, p.path AS parent_path
        FROM prep_topics t LEFT JOIN prep_topics p ON p.id = t.parent_id
        WHERE t.scope = ? AND lower(t.path) = lower(?)
        """,
        (SCOPE, path_or_name.strip()),
    ).fetchone()
    if exact:
        return _topic_row(exact)
    matches = connection.execute(
        """
        SELECT t.*, p.path AS parent_path
        FROM prep_topics t LEFT JOIN prep_topics p ON p.id = t.parent_id
        WHERE t.scope = ? AND (lower(t.name) = lower(?) OR lower(t.path) LIKE ?)
        ORDER BY length(t.path), t.path LIMIT 2
        """,
        (SCOPE, path_or_name.strip(), f"%/{path_or_name.strip().lower()}"),
    ).fetchall()
    if len(matches) == 1:
        return _topic_row(matches[0])
    if not matches:
        raise KeyError(f"unknown topic: {path_or_name}")
    return _topic_row(matches[0])


def add_prerequisite(
    connection: sqlite3.Connection, topic_path: str, prerequisite_path: str
) -> None:
    topic = get_topic(connection, topic_path)
    prerequisite = get_topic(connection, prerequisite_path)
    connection.execute(
        "INSERT OR IGNORE INTO prep_topic_prerequisites(topic_id, prerequisite_topic_id, created_at) VALUES (?, ?, ?)",
        (topic["id"], prerequisite["id"], now_iso()),
    )
    connection.commit()


def topic_prerequisites(
    connection: sqlite3.Connection, topic_path: str
) -> list[dict[str, Any]]:
    topic = get_topic(connection, topic_path)
    return [
        dict(row)
        for row in connection.execute(
            """
            SELECT p.id, p.path, p.name
            FROM prep_topic_prerequisites r
            JOIN prep_topics p ON p.id = r.prerequisite_topic_id
            WHERE r.topic_id = ? ORDER BY p.path
            """,
            (topic["id"],),
        ).fetchall()
    ]


def validate_gap_types(gap_types: Iterable[str]) -> list[str]:
    normalized = sorted({str(value).upper() for value in gap_types})
    invalid = [value for value in normalized if value not in GAP_TYPES]
    if invalid:
        raise ValueError(f"invalid gap type: {', '.join(invalid)}")
    return normalized


def _canonicalize_code(code: str) -> str:
    identifiers: dict[str, str] = {}
    declared = set(
        re.findall(
            r"\b(?:byte|short|int|long|float|double|boolean|char|String|var|[A-Z]\w*(?:<[^;=()]+>)?)\s+([a-zA-Z_$][\w$]*)",
            code,
        )
    )
    token_pattern = re.compile(
        r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|[A-Za-z_$][\w$]*|\d+(?:\.\d+)?|\S'
    )
    tokens = token_pattern.findall(code)
    output: list[str] = []
    for token in tokens:
        if token in declared:
            if token not in identifiers:
                identifiers[token] = f"$v{len(identifiers) + 1}"
            output.append(identifiers[token])
        else:
            output.append(token)
    return " ".join(output)


def question_signature(text: str) -> str:
    code_blocks: list[str] = []

    def replace_code(match: re.Match[str]) -> str:
        code = match.group(1)
        code = re.sub(r"^\s*[A-Za-z0-9_+.-]+\s*\n", "", code, count=1)
        code_blocks.append(_canonicalize_code(code))
        return f" __CODE_{len(code_blocks) - 1}__ "

    prose = re.sub(r"```([\s\S]*?)```", replace_code, text)
    prose = re.sub(r"\s+", " ", prose.lower()).strip()
    for index, code in enumerate(code_blocks):
        prose = prose.replace(f"__code_{index}__", code)
    normalized = re.sub(r"\s+", " ", prose).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _compact(value: str, limit: int = 600) -> str:
    clean = re.sub(r"\s+", " ", value).strip()
    return clean if len(clean) <= limit else clean[: limit - 1].rstrip() + "…"


def add_question(
    connection: sqlite3.Connection,
    *,
    topic_path: str,
    text: str,
    question_type: str,
    difficulty: int,
    source: str,
    created_at: str | None = None,
    commit: bool = True,
) -> dict[str, Any]:
    topic = get_topic(connection, topic_path)
    question_type = question_type.upper()
    if any(secret_like(value) for value in (topic_path, text, source)):
        raise SecretRejected(
            "Java question rejected: secret-like content detected; store only safe aliases."
        )
    if question_type not in QUESTION_TYPES:
        raise ValueError(f"invalid question type: {question_type}")
    if not 1 <= int(difficulty) <= 5:
        raise ValueError("difficulty must be between 1 and 5")
    signature = question_signature(text)
    existing = connection.execute(
        "SELECT * FROM question_bank WHERE topic_id = ? AND signature = ?",
        (topic["id"], signature),
    ).fetchone()
    if existing:
        return dict(existing)
    question_id = uuid.uuid4().hex
    connection.execute(
        """
        INSERT INTO question_bank(
            id, topic_id, question_text, compact_text, question_type,
            difficulty, source, signature, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            question_id,
            topic["id"],
            text.strip(),
            _compact(text),
            question_type,
            int(difficulty),
            source,
            signature,
            created_at or now_iso(),
        ),
    )
    if commit:
        connection.commit()
    return dict(
        connection.execute("SELECT * FROM question_bank WHERE id = ?", (question_id,)).fetchone()
    )


def _parse_time(value: str | None) -> dt.datetime:
    if value:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=dt.timezone.utc)
    return dt.datetime.now().astimezone()


def review_interval_days(
    score: int,
    *,
    attempts_count: int,
    hint_used: bool,
    stability: float,
    gap_types: Iterable[str],
) -> int:
    validate_gap_types(gap_types)
    base = {0: 1, 1: 1, 2: 4, 3: 9, 4: 28, 5: 60}[int(score)]
    factor = 1.0
    if score >= 3:
        factor += min(max(stability, 0.0), 4.0) * 0.18
        factor += min(max(attempts_count - 1, 0), 4) * 0.06
    if hint_used:
        factor *= 0.55
    severe = {"EXECUTION_MODEL_GAP", "CODING_GAP", "PRODUCTION_GAP", "FRAMEWORK_CORE_GAP"}
    if severe.intersection(gap_types):
        factor *= 0.75
    return max(1, round(base * factor))


def _state_from_row(row: sqlite3.Row) -> dict[str, Any]:
    result = dict(row)
    result["weaknesses"] = _loads(result.pop("weaknesses_json"), [])
    result["gap_types"] = _loads(result.pop("gap_types_json"), [])
    return result


def get_topic_state(
    connection: sqlite3.Connection, topic_path: str
) -> dict[str, Any]:
    topic = get_topic(connection, topic_path)
    row = connection.execute(
        """
        SELECT s.*, t.path, t.name
        FROM prep_topic_state s JOIN prep_topics t ON t.id = s.topic_id
        WHERE s.topic_id = ?
        """,
        (topic["id"],),
    ).fetchone()
    return _state_from_row(row)


def _status_for(score: int, stability: float, hint_used: bool) -> str:
    if score <= 1:
        return "OPEN"
    if score <= 3:
        return "LEARNING"
    if score >= 4 and stability >= 2 and not hint_used:
        return "STABLE"
    return "RETEST"


def record_attempt(
    connection: sqlite3.Connection,
    *,
    question_id: str,
    score: int,
    strengths: Iterable[str],
    mistakes: Iterable[str],
    gap_types: Iterable[str],
    answer_summary: str,
    hint_used: bool,
    attempted_at: str | None = None,
    session_id: str | None = None,
    source_ref: str | None = None,
    update_state: bool = True,
    commit: bool = True,
) -> dict[str, Any]:
    if not 0 <= int(score) <= 5:
        raise ValueError("score must be between 0 and 5")
    strengths = [str(value) for value in strengths if str(value).strip()]
    mistakes = [str(value) for value in mistakes if str(value).strip()]
    gap_types = [str(value) for value in gap_types]
    gaps = validate_gap_types(gap_types)
    if any(
        secret_like(value)
        for value in [answer_summary, source_ref or "", *strengths, *mistakes]
    ):
        raise SecretRejected(
            "Java attempt rejected: secret-like content detected; store only safe aliases."
        )
    strengths_list = [_compact(str(value), 500) for value in strengths if str(value).strip()]
    mistakes_list = [_compact(str(value), 500) for value in mistakes if str(value).strip()]
    question = connection.execute(
        "SELECT * FROM question_bank WHERE id = ?", (question_id,)
    ).fetchone()
    if not question:
        raise KeyError(f"unknown question: {question_id}")
    if source_ref:
        existing = connection.execute(
            "SELECT * FROM prep_attempts WHERE source_ref = ?", (source_ref,)
        ).fetchone()
        if existing:
            return dict(existing)
    timestamp = _parse_time(attempted_at).isoformat(timespec="seconds")
    state = connection.execute(
        "SELECT * FROM prep_topic_state WHERE topic_id = ?", (question["topic_id"],)
    ).fetchone()
    previous_stability = float(state["stability"])
    stability = (
        previous_stability + 1.0
        if score >= 4 and not hint_used
        else max(0.0, previous_stability - 0.75)
    )
    attempts_count = int(state["attempts_count"]) + 1
    days = review_interval_days(
        int(score),
        attempts_count=attempts_count,
        hint_used=bool(hint_used),
        stability=stability,
        gap_types=gaps,
    )
    next_review = (_parse_time(timestamp) + dt.timedelta(days=days)).isoformat(timespec="seconds")
    attempt_id = uuid.uuid4().hex
    connection.execute(
            """
            INSERT INTO prep_attempts(
                id, session_id, question_id, topic_id, score, strengths_json,
                mistakes_json, gap_types_json, answer_summary, hint_used,
                source_ref, attempted_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                attempt_id,
                session_id,
                question_id,
                question["topic_id"],
                int(score),
                _json(strengths_list),
                _json(mistakes_list),
                _json(gaps),
                _compact(answer_summary, 1000),
                1 if hint_used else 0,
                source_ref,
                timestamp,
            ),
        )
    connection.execute(
            "UPDATE question_bank SET last_asked_at = ? WHERE id = ?",
            (timestamp, question_id),
        )
    if update_state:
        confidence = min(1.0, 0.2 + attempts_count * 0.18) * (0.55 + score * 0.09)
        connection.execute(
                """
                UPDATE prep_topic_state SET
                    current_score = ?, confidence = ?, status = ?, last_tested = ?,
                    attempts_count = ?, weaknesses_json = ?, gap_types_json = ?,
                    next_review = ?, stability = ?, updated_at = ?
                WHERE topic_id = ?
                """,
                (
                    float(score),
                    min(1.0, round(confidence, 3)),
                    _status_for(int(score), stability, bool(hint_used)),
                    timestamp,
                    attempts_count,
                    _json(mistakes_list[:10]),
                    _json(gaps),
                    next_review,
                    stability,
                    timestamp,
                    question["topic_id"],
                ),
            )
    if commit:
        connection.commit()
    result = dict(
        connection.execute("SELECT * FROM prep_attempts WHERE id = ?", (attempt_id,)).fetchone()
    )
    result["next_review"] = next_review
    return result


def start_session(
    connection: sqlite3.Connection,
    *,
    mode: str,
    source: str,
    started_at: str | None = None,
    commit: bool = True,
) -> dict[str, Any]:
    mode = mode.upper()
    if mode not in MODES and mode != "LEGACY":
        raise ValueError(f"invalid mode: {mode}")
    session_id = uuid.uuid4().hex
    connection.execute(
        "INSERT INTO prep_sessions(id, mode, scope, source, status, started_at) VALUES (?, ?, ?, ?, 'ACTIVE', ?)",
        (session_id, mode, SCOPE, source, started_at or now_iso()),
    )
    if commit:
        connection.commit()
    return dict(
        connection.execute("SELECT * FROM prep_sessions WHERE id = ?", (session_id,)).fetchone()
    )


def complete_session(
    connection: sqlite3.Connection,
    session_id: str,
    *,
    summary: str,
    completed_at: str | None = None,
    commit: bool = True,
) -> dict[str, Any]:
    session = connection.execute(
        "SELECT * FROM prep_sessions WHERE id = ?", (session_id,)
    ).fetchone()
    if not session:
        raise KeyError(f"unknown session: {session_id}")
    if session["status"] != "ACTIVE":
        raise ValueError(f"session is not active: {session_id}")
    attempts = connection.execute(
        """
        SELECT a.score, a.hint_used, a.gap_types_json, q.signature, t.path
        FROM prep_attempts a
        JOIN question_bank q ON q.id = a.question_id
        JOIN prep_topics t ON t.id = a.topic_id
        WHERE a.session_id = ? ORDER BY a.attempted_at
        """,
        (session_id,),
    ).fetchall()
    gaps = sorted({gap for row in attempts for gap in _loads(row["gap_types_json"], [])})
    topics = sorted({row["path"] for row in attempts})
    signatures = [row["signature"] for row in attempts][-10:]
    hints = any(row["hint_used"] for row in attempts)
    result_text = (
        f"{summary} Topics: {', '.join(topics) or 'not recorded'}. "
        f"Remaining gaps: {', '.join(gaps) or 'insufficient data'}. "
        f"Hints used: {'yes' if hints else 'no'}."
    )
    payload = {
        "title": f"Java preparation {session['mode']} session",
        "project": SCOPE,
        "result": result_text,
        "decisions": [],
        "changed": ["knowledge.sqlite3 preparation state"],
        "verified": [f"Recorded {len(attempts)} compact attempts"],
        "next_steps": ["Select an overdue or weakest topic for the next session"],
        "open_questions": [
            f"Avoid immediate repeat of signatures: {', '.join(signatures)}"
        ] if signatures else [],
        "source_session": f"java-prep:{session_id}",
    }
    handoff = save_handoff(connection, **payload, commit=commit)
    finished = completed_at or now_iso()
    connection.execute(
        "UPDATE prep_sessions SET status = 'COMPLETED', summary = ?, completed_at = ?, handoff_id = ? WHERE id = ?",
        (_compact(summary, 2000), finished, handoff["id"], session_id),
    )
    if commit:
        connection.commit()
    payload["handoff_uri"] = handoff["uri"]
    return payload


def _record_result(
    connection: sqlite3.Connection,
    *,
    mode: str,
    topic_path: str,
    question_text: str,
    question_type: str,
    difficulty: int,
    source: str,
    score: int,
    strengths: Iterable[str],
    mistakes: Iterable[str],
    gap_types: Iterable[str],
    answer_summary: str,
    hint_used: bool,
    session_id: str | None = None,
    complete_session: bool = False,
    session_summary: str = "",
    attempted_at: str | None = None,
) -> dict[str, Any]:
    """Persist one compact evaluated answer and optionally close its session.

    This is the transaction boundary used by the MCP adapter. It deliberately
    accepts an answer summary, not a transcript or raw reasoning.
    """
    mode = mode.upper()
    if mode not in MODES:
        raise ValueError(f"invalid mode: {mode}")
    strengths = [str(value) for value in strengths if str(value).strip()]
    mistakes = [str(value) for value in mistakes if str(value).strip()]
    gap_types = [str(value) for value in gap_types]
    gaps = validate_gap_types(gap_types)
    text_fields = [
        question_text,
        answer_summary,
        session_summary,
        source,
        topic_path,
        *strengths,
        *mistakes,
    ]
    if any(secret_like(value) for value in text_fields):
        raise SecretRejected(
            "Java result rejected: secret-like content detected; store only safe aliases."
        )
    question = add_question(
        connection,
        topic_path=topic_path,
        text=question_text,
        question_type=question_type,
        difficulty=difficulty,
        source=source,
        created_at=attempted_at,
        commit=False,
    )
    if session_id:
        session = connection.execute(
            "SELECT * FROM prep_sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if not session:
            raise KeyError(f"unknown session: {session_id}")
        if session["status"] != "ACTIVE":
            raise ValueError(f"session is not active: {session_id}")
        if session["mode"] != mode:
            raise ValueError(
                f"session mode mismatch: expected {session['mode']}, got {mode}"
            )
        session_data = dict(session)
    else:
        session_data = start_session(
            connection,
            mode=mode,
            source=source,
            started_at=attempted_at,
            commit=False,
        )
        session_id = session_data["id"]
    attempt = record_attempt(
        connection,
        question_id=question["id"],
        score=score,
        strengths=strengths,
        mistakes=mistakes,
        gap_types=gap_types,
        answer_summary=answer_summary,
        hint_used=hint_used,
        attempted_at=attempted_at,
        session_id=session_id,
        commit=False,
    )
    handoff = None
    if complete_session:
        handoff = globals()["complete_session"](
            connection,
            session_id,
            summary=session_summary or f"{mode} result recorded for {topic_path}.",
            completed_at=attempted_at,
            commit=False,
        )
        session_data = dict(
            connection.execute(
                "SELECT * FROM prep_sessions WHERE id = ?", (session_id,)
            ).fetchone()
        )
    connection.commit()
    return {
        "session": session_data,
        "question": question,
        "attempt": attempt,
        "topic_state": get_topic_state(connection, topic_path),
        "handoff": handoff,
    }


def record_result(
    connection: sqlite3.Connection,
    **kwargs: Any,
) -> dict[str, Any]:
    """Persist one result atomically, rolling back every partial write on error."""
    try:
        return _record_result(connection, **kwargs)
    except Exception:
        connection.rollback()
        raise


def _attempt_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "score": row["score"],
        "strengths": _loads(row["strengths_json"], []),
        "mistakes": _loads(row["mistakes_json"], []),
        "gap_types": _loads(row["gap_types_json"], []),
        "answer_summary": row["answer_summary"],
        "hint_used": bool(row["hint_used"]),
        "attempted_at": row["attempted_at"],
        "question": row["compact_text"],
        "signature": row["signature"],
    }


def due_topics(
    connection: sqlite3.Connection, now: str | None = None, limit: int = 10
) -> list[dict[str, Any]]:
    timestamp = _parse_time(now).isoformat(timespec="seconds")
    return [
        _state_from_row(row)
        for row in connection.execute(
            """
            SELECT s.*, t.path, t.name
            FROM prep_topic_state s JOIN prep_topics t ON t.id = s.topic_id
            WHERE s.next_review IS NOT NULL AND s.next_review <= ?
            ORDER BY s.next_review, COALESCE(s.current_score, -1), t.path LIMIT ?
            """,
            (timestamp, int(limit)),
        ).fetchall()
    ]


def _choose_topic(connection: sqlite3.Connection, mode: str, now: str | None) -> dict[str, Any]:
    due = due_topics(connection, now, 1)
    if due:
        return get_topic(connection, due[0]["path"])
    condition = "s.attempts_count > 0" if mode == "RETEST" else "1 = 1"
    row = connection.execute(
        f"""
        SELECT t.path FROM prep_topics t JOIN prep_topic_state s ON s.topic_id = t.id
        WHERE t.parent_id IS NOT NULL AND {condition}
        ORDER BY CASE WHEN s.current_score IS NULL THEN 1 ELSE 0 END,
                 COALESCE(s.current_score, 9), s.confidence, t.path LIMIT 1
        """
    ).fetchone()
    return get_topic(connection, row["path"] if row else "Java/Generics")


def _infer_topic(text: str) -> str:
    lowered = text.lower()
    ordered = [
        ("type erasure", "Java/Generics/Type Erasure"),
        ("bridge method", "Java/Generics/Bridge Methods"),
        ("generics", "Java/Generics"),
        ("concurrenthashmap", "Java/Concurrency"),
        ("happens-before", "Java/Concurrency/JMM/Happens-Before"),
        ("jmm", "Java/Concurrency/JMM"),
        ("concurrency", "Java/Concurrency"),
        ("transaction", "Spring/Transactions"),
        ("postgres", "Persistence/SQL-PostgreSQL"),
        ("sql", "Persistence/SQL-PostgreSQL"),
        ("kafka", "Messaging/Kafka"),
        ("spring mvc", "Spring/Spring MVC"),
        ("spring boot", "Spring/Spring Boot"),
        ("spring", "Spring/Spring Core"),
        ("hashmap", "Java/Collections/HashMap"),
        ("collection", "Java/Collections"),
        ("stream", "Java/Streams"),
        ("jvm", "Java/JVM"),
        ("gc", "Java/GC"),
        ("system design", "Backend/System Design"),
        ("coding", "Coding"),
    ]
    for marker, path in ordered:
        if marker in lowered:
            return path
    return "Java/Java Core"


def _infer_question_type(text: str) -> str:
    lowered = text.lower()
    if "код" in lowered or "code" in lowered:
        return "CODE_OUTPUT"
    if "production" in lowered or "продак" in lowered:
        return "PRODUCTION"
    if "compile" in lowered or "скомпил" in lowered:
        return "COMPILE"
    if "edge" in lowered or "tricky" in lowered:
        return "EDGE_CASE"
    return "THEORY"


JAVA_QUESTION_MARKERS = (
    "java", "jvm", "spring", "kafka", "hibernate", "jpa", "executor",
    "thread", "volatile", "happens-before", "stream", "generic", "hashmap",
    "hashset", "gc", "garbage collector", "transaction", "erasure", "bridge method",
)


def question_quality_issues(
    connection: sqlite3.Connection,
    *,
    topic_path: str,
    text: str,
) -> list[str]:
    """Return deterministic reasons why a BANK candidate must not be stored."""
    normalized = re.sub(r"\s+", " ", str(text)).strip()
    lowered = normalized.casefold()
    issues: list[str] = []
    if len(normalized) < 20:
        issues.append("too_short")
    if len(normalized) > 700:
        issues.append("too_long")
    if not any(marker in lowered for marker in JAVA_QUESTION_MARKERS):
        issues.append("not_java_or_java_backend")
    numbered_parts = len(re.findall(r"(?:^|\s)\d+[.)]", normalized))
    compound_markers = numbered_parts + normalized.count("→") + normalized.count(";")
    if compound_markers >= 2 or "проверьте:" in lowered or "check:" in lowered:
        issues.append("compound_question")

    candidate_signature = question_signature(normalized)
    existing = connection.execute(
        "SELECT question_text, signature FROM question_bank q "
        "JOIN prep_topics t ON t.id = q.topic_id WHERE t.path = ?",
        (topic_path,),
    ).fetchall()
    candidate_tokens = set(re.findall(r"[\wа-яё@]+", lowered, flags=re.IGNORECASE))
    for row in existing:
        if row["signature"] == candidate_signature:
            issues.append("duplicate_question")
            break
        existing_text = re.sub(r"\s+", " ", row["question_text"]).strip().casefold()
        existing_tokens = set(re.findall(r"[\wа-яё@]+", existing_text, flags=re.IGNORECASE))
        union = candidate_tokens | existing_tokens
        overlap = len(candidate_tokens & existing_tokens) / max(1, len(union))
        sequence = __import__("difflib").SequenceMatcher(None, lowered, existing_text).ratio()
        if overlap >= 0.55 or sequence >= 0.68:
            issues.append("near_duplicate_question")
            break
    return sorted(set(issues))


def record_question_rejection(
    connection: sqlite3.Connection,
    *,
    topic_path: str,
    text: str,
    source: str,
    reasons: list[str],
) -> None:
    connection.execute(
        """INSERT INTO prep_question_rejections
           (id, question_text, topic_path, source, reasons_json, created_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (
            uuid.uuid4().hex,
            str(text).strip(),
            topic_path,
            source,
            json.dumps(reasons, ensure_ascii=False),
            now_iso(),
        ),
    )


def ingest_bank_questions(
    connection: sqlite3.Connection, questions: Iterable[str], source: str = "user:bank"
) -> list[dict[str, Any]]:
    saved = []
    for text in questions:
        if not str(text).strip():
            continue
        topic_path = _infer_topic(str(text))
        issues = question_quality_issues(
            connection,
            topic_path=topic_path,
            text=str(text),
        )
        if issues:
            record_question_rejection(
                connection,
                topic_path=topic_path,
                text=str(text),
                source=source,
                reasons=issues,
            )
            connection.commit()
            continue
        saved.append(
            add_question(
                connection,
                topic_path=topic_path,
                text=str(text),
                question_type=_infer_question_type(str(text)),
                difficulty=4,
                source=source,
            )
        )
    return saved


def _token_estimate(value: Any) -> int:
    rendered = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return max(1, (len(rendered.encode("utf-8")) + 3) // 4)


def _trim_context(packet: dict[str, Any], max_tokens: int) -> dict[str, Any]:
    for key in ("question_candidates", "recent_attempts", "due_for_review", "avoid_signatures", "coverage"):
        while packet.get(key) and _token_estimate(packet) > max_tokens:
            packet[key].pop()
    if _token_estimate(packet) > max_tokens and packet.get("state"):
        packet["state"]["weaknesses"] = packet["state"].get("weaknesses", [])[:2]
        packet["state"]["gap_types"] = packet["state"].get("gap_types", [])[:3]
    packet["estimated_tokens"] = _token_estimate(packet)
    if packet["estimated_tokens"] > max_tokens:
        packet["teacher_contract"] = _compact(packet.get("teacher_contract", ""), 160)
        packet["estimated_tokens"] = _token_estimate(packet)
    return packet


def build_mode_context(
    connection: sqlite3.Connection,
    *,
    mode: str,
    topic: str | None = None,
    now: str | None = None,
    max_tokens: int = 5000,
    bank_questions: Iterable[str] | None = None,
    bank_source: str = "user:bank",
) -> dict[str, Any]:
    mode = mode.upper().strip()
    if mode not in MODES:
        raise ValueError(f"invalid mode: {mode}")
    seed_default_topics(connection)
    bank_saved: list[dict[str, Any]] = []
    bank_rejections: list[dict[str, Any]] = []
    if mode == "BANK" and bank_questions:
        incoming_bank_questions = list(bank_questions)
        bank_saved = ingest_bank_questions(connection, incoming_bank_questions, source=bank_source)
        bank_rejections = [
            dict(row)
            for row in connection.execute(
                "SELECT topic_path, reasons_json FROM prep_question_rejections "
                "WHERE source = ? ORDER BY created_at DESC LIMIT ?",
                (bank_source, max(1, len(incoming_bank_questions))),
            ).fetchall()
        ]
    selected = None
    if mode not in {"MOCK", "STATUS"} or topic:
        if topic:
            selected = get_topic(connection, topic)
        elif bank_saved:
            selected_path = connection.execute(
                "SELECT path FROM prep_topics WHERE id = ?", (bank_saved[0]["topic_id"],)
            ).fetchone()["path"]
            selected = get_topic(connection, selected_path)
        else:
            selected = _choose_topic(connection, mode, now)
    state = get_topic_state(connection, selected["path"]) if selected else None
    attempts: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    avoid: list[str] = []
    prerequisites: list[dict[str, Any]] = []
    if selected:
        attempts = [
            _attempt_dict(row)
            for row in connection.execute(
                """
                SELECT a.*, q.compact_text, q.signature
                FROM prep_attempts a JOIN question_bank q ON q.id = a.question_id
                WHERE a.topic_id = ? ORDER BY a.attempted_at DESC LIMIT 5
                """,
                (selected["id"],),
            ).fetchall()
        ]
        avoid = [
            row["signature"]
            for row in connection.execute(
                """
                SELECT DISTINCT q.signature FROM prep_attempts a
                JOIN question_bank q ON q.id = a.question_id
                WHERE a.topic_id = ? ORDER BY a.attempted_at DESC LIMIT 20
                """,
                (selected["id"],),
            ).fetchall()
        ]
        used_clause = (
            "AND NOT EXISTS (SELECT 1 FROM prep_attempts a WHERE a.question_id = q.id)"
            if mode == "RETEST"
            else ""
        )
        candidates = [
            dict(row)
            for row in connection.execute(
                f"""
                SELECT q.id, q.compact_text AS question, q.question_type, q.difficulty, q.signature, q.source
                FROM question_bank q WHERE q.topic_id = ? {used_clause}
                ORDER BY q.difficulty DESC, q.created_at LIMIT 8
                """,
                (selected["id"],),
            ).fetchall()
        ]
        prerequisites = topic_prerequisites(connection, selected["path"])
    coverage = list(LEGACY_TOPIC_MAP.values()) if mode == "MOCK" else []
    contracts = {
        "TRAIN": "One narrow topic. Minimal theory, internals, syntax/API, 3-5 tricky or practical questions, coding, review, then RETEST. Ask one question at a time.",
        "RETEST": "No theory before the answer. Use new code, edge cases, production scenarios and formulations. Ask one question at a time.",
        "MOCK": "Run a strict full interview. Do not teach until the mock ends or the user asks. Ask one question at a time.",
        "BANK": "Classify external questions, hide answers, add Senior follow-ups, avoid used signatures and ask one at a time.",
        "STATUS": "Return current tracker state only. Keep the historical 2026-08-16 diagnostic baseline separate and cite its source when comparing.",
    }
    packet = {
        "mode": mode,
        "scope": SCOPE,
        "topic": {key: selected[key] for key in ("id", "path", "name", "parent_path")} if selected else None,
        "state": state,
        "prerequisites": prerequisites,
        "recent_attempts": attempts,
        "question_candidates": candidates,
        "question_rejections": bank_rejections,
        "avoid_signatures": avoid,
        "due_for_review": due_topics(connection, now, 10),
        "coverage": coverage,
        "status_summary": compact_status_report(connection, now) if mode == "STATUS" else None,
        "lifecycle": {
            "record_result_required": mode in {"TRAIN", "RETEST", "MOCK"},
            "complete_session_after_significant_block": mode in {"TRAIN", "RETEST", "MOCK"},
            "record_contract": "After each evaluated attempt call java_prep_record_result with score, strengths, mistakes, one primary gap type, compact answer summary and hint_used.",
            "close_contract": "After a meaningful block or completed mock, call java_prep_record_result with completeSession=true and a short sessionSummary.",
        },
        "teacher_contract": contracts[mode],
    }
    return _trim_context(packet, max(300, int(max_tokens)))


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _already_imported(connection: sqlite3.Connection, path: Path, digest: str) -> bool:
    row = connection.execute(
        "SELECT content_hash FROM prep_imports WHERE source_path = ?", (str(path.resolve()),)
    ).fetchone()
    return bool(row and row["content_hash"] == digest)


def _mark_imported(connection: sqlite3.Connection, path: Path, digest: str, result: dict[str, Any]) -> None:
    connection.execute(
        "INSERT OR REPLACE INTO prep_imports(source_path, content_hash, imported_at, result_json) VALUES (?, ?, ?, ?)",
        (str(path.resolve()), digest, now_iso(), _json(result)),
    )
    connection.commit()


def _legacy_gap_types(text: str) -> list[str]:
    lowered = text.lower()
    result = set()
    if any(marker in lowered for marker in ("execution", "runtime", "пошаг", "internals", "внутр")):
        result.add("EXECUTION_MODEL_GAP")
    if any(marker in lowered for marker in ("syntax", "api", "метод неизвест", "не предложен корректный код")):
        result.add("API_SYNTAX_GAP")
    if any(marker in lowered for marker in ("coding", "код без подсказ", "задача не выполн")):
        result.add("CODING_GAP")
    if any(marker in lowered for marker in ("production", "продак", "trade-off", "инцидент")):
        result.add("PRODUCTION_GAP")
    if any(marker in lowered for marker in ("формулиров", "противореч", "объясн")):
        result.add("INTERVIEW_EXPRESSION_GAP")
    if not result:
        result.add("THEORY_GAP")
    return sorted(result)


def _legacy_topic(value: str) -> str:
    lowered = value.strip().lower()
    if lowered in LEGACY_TOPIC_MAP:
        return LEGACY_TOPIC_MAP[lowered]
    for key, path in LEGACY_TOPIC_MAP.items():
        if key in lowered:
            return path
    return _infer_topic(value)


def _import_session_file(connection: sqlite3.Connection, path: Path) -> dict[str, int]:
    text = path.read_text(encoding="utf-8-sig")
    sections = re.findall(r"(?ms)^##\s+(.+?)\n(.*?)(?=^##\s+|\Z)", text)
    attempts = 0
    questions = 0
    session = start_session(
        connection,
        mode="LEGACY",
        source=f"legacy:{path}",
        started_at=f"{path.stem[:10]}T00:00:00+03:00" if re.match(r"\d{4}-\d{2}-\d{2}", path.stem) else now_iso(),
    )
    for index, (heading, body) in enumerate(sections):
        score_match = re.search(r"Оценка:\s*`?(\d(?:\.\d)?)\s*/\s*5", body, re.IGNORECASE)
        if not score_match:
            continue
        score = max(0, min(5, round(float(score_match.group(1)))))
        topic_path = _legacy_topic(heading)
        question = add_question(
            connection,
            topic_path=topic_path,
            text=heading.strip(),
            question_type=_infer_question_type(heading + " " + body[:500]),
            difficulty=max(1, min(5, score + 1)),
            source=f"legacy:{path}",
            created_at=session["started_at"],
        )
        questions += 1
        mistakes_block = re.search(
            r"(?ms)Ошибки и пробелы:\s*(.*?)(?=\n###|\nЧто распознано:|\Z)", body
        )
        mistakes = re.findall(r"(?m)^-\s+(.+)", mistakes_block.group(1)) if mistakes_block else []
        strengths_block = re.search(
            r"(?ms)Что распознано:\s*(.*?)(?=\nОшибки и пробелы:|\n###|\Z)", body
        )
        strengths = re.findall(r"(?m)^-\s+(.+)", strengths_block.group(1)) if strengths_block else []
        gaps = _legacy_gap_types(" ".join(mistakes) + " " + body)
        record_attempt(
            connection,
            question_id=question["id"],
            score=score,
            strengths=strengths[:8],
            mistakes=mistakes[:10] or ["Legacy diagnostic recorded an unresolved gap"],
            gap_types=gaps,
            answer_summary=f"Legacy diagnostic result: {score}/5.",
            hint_used="подсказ" in body.lower(),
            attempted_at=session["started_at"],
            session_id=session["id"],
            source_ref=f"{path.resolve()}#section-{index}",
        )
        attempts += 1
    connection.execute(
        "UPDATE prep_sessions SET status = 'COMPLETED', completed_at = started_at, summary = ? WHERE id = ?",
        (f"Imported {attempts} legacy attempts", session["id"]),
    )
    connection.commit()
    return {"attempts": attempts, "questions": questions}


def _import_progress_file(connection: sqlite3.Connection, path: Path) -> int:
    text = path.read_text(encoding="utf-8-sig")
    updated = 0
    for name, latest, _average, count, status in re.findall(
        r"(?m)^\|\s*([^|]+?)\s*\|\s*([0-5](?:\.\d)?)\s*\|\s*([0-5](?:\.\d)?)\s*\|\s*(\d+)\s*\|\s*([^|]+?)\s*\|$",
        text,
    ):
        if name.strip().lower() == "тема":
            continue
        topic = get_topic(connection, _legacy_topic(name))
        score = float(latest)
        existing = connection.execute(
            "SELECT attempts_count, stability FROM prep_topic_state WHERE topic_id = ?", (topic["id"],)
        ).fetchone()
        state_status = "OPEN" if score <= 1 else "LEARNING" if score <= 3 else "RETEST"
        connection.execute(
            """
            UPDATE prep_topic_state SET current_score = ?, confidence = ?, status = ?,
                attempts_count = MAX(attempts_count, ?), updated_at = ?
            WHERE topic_id = ?
            """,
            (
                score,
                min(0.85, 0.2 + int(count) * 0.12),
                state_status,
                int(count),
                now_iso(),
                topic["id"],
            ),
        )
        updated += 1
    connection.commit()
    return updated


def _import_weak_topics_file(connection: sqlite3.Connection, path: Path) -> int:
    text = path.read_text(encoding="utf-8-sig")
    updated = 0
    for line in re.findall(r"(?m)^\d+\.\s+\*\*(.+?)\*\*\s*(.*)$", text):
        label, detail = line
        topic_path = _legacy_topic(label)
        topic = get_topic(connection, topic_path)
        state = connection.execute(
            "SELECT weaknesses_json, gap_types_json FROM prep_topic_state WHERE topic_id = ?",
            (topic["id"],),
        ).fetchone()
        weaknesses = _loads(state["weaknesses_json"], [])
        clean = _compact(detail.strip(" :—-"), 500)
        if clean and clean not in weaknesses:
            weaknesses.append(clean)
        gaps = sorted(set(_loads(state["gap_types_json"], [])) | set(_legacy_gap_types(label + " " + detail)))
        connection.execute(
            "UPDATE prep_topic_state SET weaknesses_json = ?, gap_types_json = ?, updated_at = ? WHERE topic_id = ?",
            (_json(weaknesses[:10]), _json(gaps), now_iso(), topic["id"]),
        )
        updated += 1
    connection.commit()
    return updated


def import_legacy_diagnostic(
    connection: sqlite3.Connection, legacy_root: str | Path
) -> dict[str, Any]:
    root = Path(legacy_root).resolve()
    if not root.exists():
        raise FileNotFoundError(root)
    seed_default_topics(connection)
    result = {"files_imported": 0, "topics_updated": 0, "questions_imported": 0, "attempts_imported": 0, "legacy_root": str(root), "sources_preserved": True}
    files = sorted((root / "sessions").glob("*.md")) if (root / "sessions").exists() else []
    for path in files:
        digest = _file_hash(path)
        if _already_imported(connection, path, digest):
            continue
        imported = _import_session_file(connection, path)
        result["files_imported"] += 1
        result["questions_imported"] += imported["questions"]
        result["attempts_imported"] += imported["attempts"]
        _mark_imported(connection, path, digest, imported)
    weak = root / "weak-topics.md"
    if weak.exists():
        digest = _file_hash(weak)
        if not _already_imported(connection, weak, digest):
            count = _import_weak_topics_file(connection, weak)
            result["files_imported"] += 1
            result["topics_updated"] += count
            _mark_imported(connection, weak, digest, {"topics_updated": count})
    progress = root / "progress.md"
    if progress.exists():
        digest = _file_hash(progress)
        if not _already_imported(connection, progress, digest):
            count = _import_progress_file(connection, progress)
            result["files_imported"] += 1
            result["topics_updated"] += count
            _mark_imported(connection, progress, digest, {"topics_updated": count})
    return result


def status_report(connection: sqlite3.Connection, now: str | None = None) -> dict[str, Any]:
    rows = [
        _state_from_row(row)
        for row in connection.execute(
            """
            SELECT s.*, t.path, t.name FROM prep_topic_state s
            JOIN prep_topics t ON t.id = s.topic_id
            WHERE t.parent_id IS NOT NULL OR s.current_score IS NOT NULL
            ORDER BY COALESCE(s.current_score, -1), t.path
            """
        ).fetchall()
    ]
    tested = [row for row in rows if row["current_score"] is not None]
    average = round(sum(row["current_score"] for row in tested) / len(tested), 2) if tested else 0.0
    if average >= 4:
        level = "Senior signal"
    elif average >= 3:
        level = "Middle signal"
    elif average >= 2:
        level = "Junior+ / unstable Middle signal"
    else:
        level = "Junior+ / unstable Middle- signal" if tested else "Insufficient data"
    gaps: dict[str, int] = {}
    for row in tested:
        for gap in row["gap_types"]:
            gaps[gap] = gaps.get(gap, 0) + 1
    trend_rows = connection.execute(
        """
        SELECT t.path, a.score, a.attempted_at
        FROM prep_attempts a JOIN prep_topics t ON t.id = a.topic_id
        ORDER BY t.path, a.attempted_at DESC
        """
    ).fetchall()
    scores_by_topic: dict[str, list[int]] = {}
    for row in trend_rows:
        scores_by_topic.setdefault(row["path"], []).append(int(row["score"]))
    trend = [
        {"path": path, "change": scores[0] - scores[1], "latest": scores[0]}
        for path, scores in scores_by_topic.items()
        if len(scores) >= 2 and scores[0] != scores[1]
    ]
    overdue = due_topics(connection, now, 10)
    weak = [row for row in tested if row["current_score"] < 3][:10]
    next_topic = overdue[0] if overdue else (weak[0] if weak else None)
    return {
        "scope": SCOPE,
        "level": level,
        "average_score": average,
        "tested_topics": len(tested),
        "strong_topics": [row for row in sorted(tested, key=lambda item: (-item["current_score"], item["path"])) if row["current_score"] >= 3.5][:8],
        "weak_topics": weak,
        "insufficient_data": [row["path"] for row in rows if row["current_score"] is None][:15],
        "top_gaps": sorted(gaps.items(), key=lambda item: (-item[1], item[0]))[:6],
        "overdue": overdue,
        "trend": sorted(trend, key=lambda item: (-abs(item["change"]), item["path"]))[:8],
        "best_next_step": (
            f"TRAIN {next_topic['path']} with focus on {', '.join(next_topic['gap_types']) or 'diagnostic evidence'}"
            if next_topic
            else "Start MODE: TRAIN to collect evidence"
        ),
    }


def compact_status_report(connection: sqlite3.Connection, now: str | None = None) -> dict[str, Any]:
    report = status_report(connection, now)
    return {
        "level": report["level"],
        "average_score": report["average_score"],
        "tested_topics": report["tested_topics"],
        "strong_topics": [
            {"path": item["path"], "score": item["current_score"]}
            for item in report["strong_topics"][:6]
        ],
        "weak_topics": [
            {"path": item["path"], "score": item["current_score"], "gaps": item["gap_types"][:3]}
            for item in report["weak_topics"][:8]
        ],
        "top_gaps": report["top_gaps"],
        "insufficient_data": report["insufficient_data"][:10],
        "overdue": [item["path"] for item in report["overdue"][:8]],
        "trend": report["trend"][:6],
        "best_next_step": report["best_next_step"],
    }


def render_dashboard(connection: sqlite3.Connection, now: str | None = None) -> str:
    report = status_report(connection, now)
    focus = report["overdue"][0] if report["overdue"] else (report["weak_topics"][0] if report["weak_topics"] else None)
    last_session = connection.execute(
        "SELECT * FROM prep_sessions WHERE status = 'COMPLETED' ORDER BY completed_at DESC LIMIT 1"
    ).fetchone()

    def topic_lines(items: list[dict[str, Any]], empty: str) -> str:
        return "\n".join(f"- {item['path']}: {item['current_score']}/5" for item in items) or f"- {empty}"

    weakness_lines = []
    for item in report["weak_topics"][:10]:
        detail = "; ".join(item["weaknesses"][:2]) or ", ".join(item["gap_types"]) or "needs retest"
        weakness_lines.append(f"- {item['path']} ({item['current_score']}/5): {detail}")
    due_lines = [f"- {item['path']} — due {item['next_review']}" for item in report["overdue"][:8]]
    next_items = []
    for item in (report["overdue"] + report["weak_topics"]):
        action = f"{item['path']}: RETEST with a new code or production scenario"
        if action not in next_items:
            next_items.append(action)
        if len(next_items) == 5:
            break
    last = "- No completed training session yet; legacy diagnostic may be imported separately."
    if last_session:
        session_attempts = connection.execute(
            """
            SELECT a.topic_id, a.score, a.gap_types_json, a.attempted_at, t.path
            FROM prep_attempts a JOIN prep_topics t ON t.id = a.topic_id
            WHERE a.session_id = ? ORDER BY a.attempted_at
            """,
            (last_session["id"],),
        ).fetchall()
        details = []
        for item in session_attempts[-3:]:
            previous = connection.execute(
                """
                SELECT score FROM prep_attempts
                WHERE topic_id = ? AND attempted_at < ?
                ORDER BY attempted_at DESC LIMIT 1
                """,
                (item["topic_id"], last_session["started_at"]),
            ).fetchone()
            before = str(previous["score"]) if previous else "new"
            gaps = ", ".join(_loads(item["gap_types_json"], [])) or "none recorded"
            details.append(
                f"- {item['path']}: {before} → {item['score']}; improved: {last_session['summary'] or 'attempt recorded'}; residual gap: {gaps}."
            )
        last = "\n".join(details) or (
            f"- {last_session['mode']} — {last_session['summary']} ({last_session['completed_at']})"
        )
    focus_text = (
        f"{focus['path']}. Gaps: {', '.join(focus['gap_types']) or 'insufficient data'}."
        if focus
        else "Import the initial diagnostic or start MODE: TRAIN."
    )
    return f"""# Senior Java Preparation

> Generated from AI Knowledge. The SQLite database is the source of truth.

## Current level

{report['level']}; average {report['average_score']}/5 across {report['tested_topics']} tested topics.

## Current focus

{focus_text}

## Strong topics

{topic_lines(report['strong_topics'], 'No stable 4+/5 evidence yet.')}

## Main weaknesses

{chr(10).join(weakness_lines) or '- Insufficient data.'}

## Due for review

{chr(10).join(due_lines) or '- Nothing is overdue.'}

## Last session

{last}

## Next

{chr(10).join(f'- {item}' for item in next_items) or '- Start `MODE: TRAIN`.'}
"""


def export_dashboard(
    connection: sqlite3.Connection,
    destination: str | Path,
    *,
    confirmed: bool,
    now: str | None = None,
) -> dict[str, Any]:
    if not confirmed:
        raise ConfirmationRequired("dashboard export requires explicit confirmation")
    path = Path(destination).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    content = render_dashboard(connection, now)
    path.write_text(content, encoding="utf-8")
    return {"path": str(path), "bytes": len(content.encode("utf-8")), "source_of_truth": str(DEFAULT_DB_PATH)}


def preparation_documents(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    if "prep_topics" not in tables:
        return []
    report = status_report(connection)
    modified = connection.execute(
        "SELECT MAX(updated_at) FROM prep_topic_state"
    ).fetchone()[0] or now_iso()
    documents = [
        {
            "path": "memory://preparation/status",
            "collection": "sqlite-preparation",
            "scope": SCOPE,
            "doc_type": "preparation_state",
            "mode": "content",
            "title": "Senior Java Preparation current status",
            "modified_at": modified,
            "content": render_dashboard(connection),
        }
    ]
    states = connection.execute(
        """
        SELECT s.*, t.path, t.name FROM prep_topic_state s
        JOIN prep_topics t ON t.id = s.topic_id WHERE s.current_score IS NOT NULL
        ORDER BY s.updated_at DESC
        """
    ).fetchall()
    for state_row in states:
        state = _state_from_row(state_row)
        recent = connection.execute(
            """
            SELECT a.score, a.mistakes_json, a.gap_types_json, a.attempted_at, q.signature
            FROM prep_attempts a JOIN question_bank q ON q.id = a.question_id
            WHERE a.topic_id = ? ORDER BY a.attempted_at DESC LIMIT 3
            """,
            (state["topic_id"],),
        ).fetchall()
        lines = [
            f"# {state['path']}",
            "",
            f"Current score: {state['current_score']}/5",
            f"Confidence: {state['confidence']}",
            f"Status: {state['status']}",
            f"Gap types: {', '.join(state['gap_types']) or 'none recorded'}",
            f"Weaknesses: {'; '.join(state['weaknesses']) or 'none recorded'}",
            f"Next review: {state['next_review'] or 'not scheduled'}",
            "",
            "Recent attempts:",
        ]
        for item in recent:
            lines.append(
                f"- {item['attempted_at']}: {item['score']}/5; gaps {', '.join(_loads(item['gap_types_json'], []))}; mistakes {'; '.join(_loads(item['mistakes_json'], [])[:2])}; signature {item['signature']}"
            )
        documents.append(
            {
                "path": f"memory://preparation/topics/{state['topic_id']}",
                "collection": "sqlite-preparation",
                "scope": SCOPE,
                "doc_type": "preparation_topic",
                "mode": "content",
                "title": f"Java preparation: {state['path']}",
                "modified_at": state["updated_at"],
                "content": "\n".join(lines),
            }
        )
    return documents


def smoke_modes(connection: sqlite3.Connection) -> dict[str, Any]:
    results = {}
    for mode in sorted(MODES):
        kwargs = {"bank_questions": ["What does type erasure change at runtime?"]} if mode == "BANK" else {}
        packet = build_mode_context(connection, mode=mode, max_tokens=1000, **kwargs)
        results[mode] = {"ok": packet["mode"] == mode, "tokens": packet["estimated_tokens"]}
    return {"ok": all(item["ok"] for item in results.values()), "modes": results}


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Senior Java Preparation inside AI Knowledge")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("migrate")
    sub.add_parser("seed")
    status = sub.add_parser("status")
    status.add_argument("--now")
    mode = sub.add_parser("mode")
    mode.add_argument("mode", choices=sorted(MODES))
    mode.add_argument("--topic")
    mode.add_argument("--now")
    mode.add_argument("--max-tokens", type=int, default=5000)
    mode.add_argument("--bank-json", default="[]")
    mode.add_argument("--bank-source", default="user:bank")
    legacy = sub.add_parser("import-legacy")
    legacy.add_argument("path")
    dashboard = sub.add_parser("dashboard")
    dashboard.add_argument("path")
    dashboard.add_argument("--confirm", action="store_true")
    sub.add_parser("smoke")
    question = sub.add_parser("add-question")
    question.add_argument("--topic", required=True)
    question.add_argument("--text", required=True)
    question.add_argument("--type", required=True, choices=sorted(QUESTION_TYPES))
    question.add_argument("--difficulty", type=int, default=4)
    question.add_argument("--source", default="user:explicit")
    attempt = sub.add_parser("record-attempt")
    attempt.add_argument("--payload-json", required=True)
    result = sub.add_parser("record-result")
    result.add_argument("--payload-json", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    connection = connect(args.db)
    try:
        if args.command == "migrate":
            seed_default_topics(connection)
            _print({"database": str(Path(args.db).resolve()), "schema_version": 1})
        elif args.command == "seed":
            _print({"topics_added": seed_default_topics(connection)})
        elif args.command == "status":
            _print(status_report(connection, args.now))
        elif args.command == "mode":
            _print(
                build_mode_context(
                    connection,
                    mode=args.mode,
                    topic=args.topic,
                    now=args.now,
                    max_tokens=args.max_tokens,
                    bank_questions=json.loads(args.bank_json),
                    bank_source=args.bank_source,
                )
            )
        elif args.command == "import-legacy":
            _print(import_legacy_diagnostic(connection, args.path))
        elif args.command == "dashboard":
            _print(export_dashboard(connection, args.path, confirmed=args.confirm))
        elif args.command == "smoke":
            _print(smoke_modes(connection))
        elif args.command == "add-question":
            _print(
                add_question(
                    connection,
                    topic_path=args.topic,
                    text=args.text,
                    question_type=args.type,
                    difficulty=args.difficulty,
                    source=args.source,
                )
            )
        elif args.command == "record-attempt":
            _print(record_attempt(connection, **json.loads(args.payload_json)))
        elif args.command == "record-result":
            _print(record_result(connection, **json.loads(args.payload_json)))
        return 0
    finally:
        connection.close()


if __name__ == "__main__":
    raise SystemExit(main())
