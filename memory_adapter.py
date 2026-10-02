from __future__ import annotations

import re
from datetime import datetime
from typing import Any


# Locale-independent parser for strings like: 2023/05/30 (Tue) 11:46
# The weekday label is ignored because it is redundant and can depend on OS locale.
_LONGMEMEVAL_DATE_RE = re.compile(
    r"^\s*(\d{4})/(\d{1,2})/(\d{1,2})"
    r"(?:\s+\([^)]+\))?\s+"
    r"(\d{1,2}):(\d{2})\s*$"
)


def parse_longmemeval_date(value: str) -> datetime:
    """Parse a LongMemEval timestamp without relying on the system locale."""
    if not isinstance(value, str):
        raise ValueError(f"Invalid LongMemEval timestamp: {value!r}")

    match = _LONGMEMEVAL_DATE_RE.match(value)
    if not match:
        raise ValueError(f"Invalid LongMemEval timestamp: {value!r}")

    year, month, day, hour, minute = map(int, match.groups())
    try:
        return datetime(year, month, day, hour, minute)
    except ValueError as exc:
        raise ValueError(f"Invalid LongMemEval timestamp: {value!r}") from exc


def _is_future_calendar_day(session_dt: datetime, question_dt: datetime) -> bool:
    """
    Availability filtering is day-level, not minute-level.

    LongMemEval's data generation can assign random times within the same calendar
    day. Therefore a session a few minutes later than question_date on the same day
    is retained; only sessions on a strictly later calendar date are excluded.
    """
    return session_dt.date() > question_dt.date()


def _render_turn(turn: dict[str, Any]) -> str:
    role = str(turn.get("role", "unknown")).strip()
    content = turn.get("content", "")
    text = content if isinstance(content, str) else str(content)
    return f"{role.capitalize()}: {text}".strip()


def _split_session_into_rounds(
    session: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    """Split one session into dialogue rounds, normally user + assistant."""
    rounds: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []

    for turn in session:
        role = turn.get("role")

        if role == "user" and current:
            rounds.append(current)
            current = []

        current.append(turn)

        if role == "assistant":
            rounds.append(current)
            current = []

    if current:
        rounds.append(current)

    return rounds


def temporal_diagnostics(record: dict[str, Any]) -> dict[str, Any]:
    """
    Inspect timestamps without assuming the input list is sorted.

    Two notions are deliberately separated:
    - future calendar day: session date is strictly after the question date;
      these are excluded by default.
    - same-day later time: session is on the same day but has a later HH:MM;
      these are retained because LongMemEval can randomize times within a day.
    """
    dates = record.get("haystack_dates", [])
    session_ids = [str(x) for x in record.get("haystack_session_ids", [])]
    answer_session_ids = {
        str(x) for x in record.get("answer_session_ids", [])
    }
    question_date_raw = record.get("question_date")

    parsed_dates = [parse_longmemeval_date(x) for x in dates]
    sorted_by_date = all(
        parsed_dates[i] <= parsed_dates[i + 1]
        for i in range(len(parsed_dates) - 1)
    )

    future_session_ids: list[str] = []
    future_gold_session_ids: list[str] = []
    same_day_later_session_ids: list[str] = []
    same_day_later_gold_session_ids: list[str] = []

    if question_date_raw:
        question_dt = parse_longmemeval_date(question_date_raw)

        for session_id, session_dt in zip(session_ids, parsed_dates):
            if _is_future_calendar_day(session_dt, question_dt):
                future_session_ids.append(session_id)
                if session_id in answer_session_ids:
                    future_gold_session_ids.append(session_id)
            elif (
                session_dt.date() == question_dt.date()
                and session_dt.time() > question_dt.time()
            ):
                same_day_later_session_ids.append(session_id)
                if session_id in answer_session_ids:
                    same_day_later_gold_session_ids.append(session_id)

    return {
        "sorted_by_date": sorted_by_date,
        "num_future_sessions": len(future_session_ids),
        "future_session_ids": future_session_ids,
        "num_future_gold_sessions": len(future_gold_session_ids),
        "future_gold_session_ids": future_gold_session_ids,
        "num_same_day_later_sessions": len(same_day_later_session_ids),
        "same_day_later_session_ids": same_day_later_session_ids,
        "num_same_day_later_gold_sessions": len(
            same_day_later_gold_session_ids
        ),
        "same_day_later_gold_session_ids": same_day_later_gold_session_ids,
    }


def record_to_memories(
    record: dict[str, Any],
    *,
    exclude_future: bool = True,
) -> list[dict[str, Any]]:
    """
    Convert one classic LongMemEval record into dialogue-round memory items.

    Methodological choices:
    - One dialogue round is one memory item, matching the Research Plan.
    - Recency is computed from parsed haystack_dates, NOT list position.
    - By default, only sessions on a calendar date strictly later than the
      question date are excluded. Same-day sessions are retained even when their
      HH:MM is later than question_date, because LongMemEval may assign random
      within-day times.
    - Full timestamps are still used for recency ranking.
    - Ties are broken by original session position and then round index so rounds
      from two same-minute sessions do not interleave.
    - Round relevance uses turn-level has_answer=True labels. Session-level
      evidence is kept separately via answer_session_ids.
    """
    sessions = record.get("haystack_sessions", [])
    raw_session_ids = record.get("haystack_session_ids", [])
    session_ids = [str(x) for x in raw_session_ids]
    dates = record.get("haystack_dates", [])
    answer_session_ids = {
        str(x) for x in record.get("answer_session_ids", [])
    }
    question_date_raw = record.get("question_date")
    question_dt = (
        parse_longmemeval_date(question_date_raw)
        if question_date_raw
        else None
    )

    if not (len(sessions) == len(session_ids) == len(dates)):
        raise ValueError(
            "haystack_sessions, haystack_session_ids, and haystack_dates "
            "must have the same length."
        )

    memories: list[dict[str, Any]] = []

    for original_session_index, (session, session_id, date_raw) in enumerate(
        zip(sessions, session_ids, dates)
    ):
        session_dt = parse_longmemeval_date(date_raw)
        is_future_calendar_day = (
            question_dt is not None
            and _is_future_calendar_day(session_dt, question_dt)
        )
        is_same_day_later = (
            question_dt is not None
            and session_dt.date() == question_dt.date()
            and session_dt.time() > question_dt.time()
        )

        if exclude_future and is_future_calendar_day:
            continue

        rounds = _split_session_into_rounds(session)
        has_any_turn_level_label = any(
            "has_answer" in turn for turn in session
        )

        for round_index, round_turns in enumerate(rounds):
            turn_level_relevant = any(
                bool(turn.get("has_answer", False))
                for turn in round_turns
            )

            # If this session contains turn-level labels, use them exactly.
            # If it has no turn-level labels at all, use session-level evidence
            # only as a fallback rather than silently treating all rounds as
            # relevant when explicit False labels are present.
            if has_any_turn_level_label:
                is_round_relevant = turn_level_relevant
            else:
                is_round_relevant = session_id in answer_session_ids

            text = "\n".join(_render_turn(t) for t in round_turns)

            # Tuple ordering prevents interleaving rounds across sessions that
            # share the same minute-level timestamp. Full timestamp remains the
            # primary recency signal; original list position is only a tie-break.
            recency_order = (
                session_dt.timestamp(),
                original_session_index,
                round_index,
            )

            memories.append(
                {
                    "id": f"{session_id}::round_{round_index}",
                    "session_id": session_id,
                    "original_session_index": original_session_index,
                    "round_index": round_index,
                    "date": date_raw,
                    "question_date": question_date_raw,
                    "recency_order": recency_order,
                    "text": text,
                    "is_round_relevant": is_round_relevant,
                    "is_gold_session": session_id in answer_session_ids,
                    "is_future": is_future_calendar_day,
                    "is_same_day_later": is_same_day_later,
                }
            )

    return memories


def relevant_memory_ids(memories: list[dict[str, Any]]) -> list[str]:
    """Gold dialogue-round IDs according to has_answer=True."""
    return [
        str(memory["id"])
        for memory in memories
        if memory.get("is_round_relevant")
    ]


def gold_session_ids(record: dict[str, Any]) -> list[str]:
    """Gold evidence sessions exactly as supplied by LongMemEval."""
    return [str(x) for x in record.get("answer_session_ids", [])]


def render_memory_for_generation(memory: dict[str, Any]) -> str:
    """Render one retrieved memory with its date for answer generation."""
    return f"[Session date: {memory.get('date', '')}]\n{memory.get('text', '')}"


def build_generation_context(
    retrieved_memories: list[dict[str, Any]],
    *,
    question_date: str | None,
    chronological: bool = True,
) -> str:
    """
    Build a generation context with dates.

    All retrieval strategies use the same ordering policy before the answer LLM
    sees the selected memories. Chronological order is the default so temporal
    relations are explicit and strategy-independent.
    """
    items = list(retrieved_memories)
    if chronological:
        items.sort(key=lambda x: x["recency_order"])

    parts = []
    if question_date:
        parts.append(f"[Question date: {question_date}]")
    parts.extend(render_memory_for_generation(x) for x in items)
    return "\n\n".join(parts)
