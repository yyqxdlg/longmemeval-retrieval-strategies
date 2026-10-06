from __future__ import annotations

import hashlib
from typing import Any

from memory_adapter import build_generation_context, record_to_memories


DEFAULT_SYSTEM_PROMPT = (
    "Answer the question using only the supplied memories. Use the dates when "
    "temporal reasoning is required. If the memories do not contain enough "
    "information, say that the information is insufficient. Give a concise answer."
)


def parse_retrieved_memory_ids(value: str) -> list[str]:
    return [item for item in str(value or "").split("|") if item]


def reconstruct_retrieved_memories(
    record: dict[str, Any],
    retrieved_memory_ids: str,
    *,
    exclude_future: bool = True,
) -> list[dict[str, Any]]:
    """Rebuild exactly the memory items named in one retrieval-result row."""
    requested = parse_retrieved_memory_ids(retrieved_memory_ids)
    if not requested:
        raise ValueError("retrieved_memory_ids is empty")

    all_memories = record_to_memories(record, exclude_future=exclude_future)
    by_id = {str(memory["id"]): memory for memory in all_memories}
    missing = [memory_id for memory_id in requested if memory_id not in by_id]
    if missing:
        raise ValueError(
            "Could not reconstruct retrieved memories: " + ", ".join(missing)
        )
    return [by_id[memory_id] for memory_id in requested]


def build_answer_messages(
    *,
    question: str,
    question_date: str | None,
    retrieved_memories: list[dict[str, Any]],
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> tuple[list[dict[str, str]], str]:
    """
    Build the only payload visible to the answer model.

    The function accepts no gold answer, relevance label, answer-session ID, or
    task label. This boundary prevents evaluation labels from entering prompts.
    """
    context = build_generation_context(
        retrieved_memories,
        question_date=question_date,
        chronological=True,
    )
    user_prompt = f"Retrieved memories:\n\n{context}\n\nQuestion: {question.strip()}"
    messages = [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": user_prompt},
    ]
    return messages, context


def context_sha256(context: str) -> str:
    return hashlib.sha256(context.encode("utf-8")).hexdigest()
