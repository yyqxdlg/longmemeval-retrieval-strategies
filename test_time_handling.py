"""Small local sanity checks for calendar-day filtering and tie handling."""

from memory_adapter import record_to_memories, temporal_diagnostics


def _turns(label):
    return [
        {"role": "user", "content": f"u-{label}"},
        {"role": "assistant", "content": f"a-{label}"},
        {"role": "user", "content": f"u2-{label}"},
        {"role": "assistant", "content": f"a2-{label}"},
    ]


def main():
    record = {
        "question_date": "2023/05/29 (Mon) 10:00",
        "haystack_dates": [
            "2023/05/29 (Mon) 10:30",  # same day: must stay
            "2023/05/29 (Mon) 10:30",  # same minute: must stay grouped
            "2023/05/30 (Tue) 09:00",  # later day: must be excluded
        ],
        "haystack_session_ids": ["s1", "s2", "s3"],
        "haystack_sessions": [_turns("1"), _turns("2"), _turns("3")],
        "answer_session_ids": ["s1"],
    }

    diag = temporal_diagnostics(record)
    assert diag["num_future_sessions"] == 1
    assert diag["num_same_day_later_sessions"] == 2

    memories = record_to_memories(record, exclude_future=True)
    assert {m["session_id"] for m in memories} == {"s1", "s2"}

    # Same-minute session tie-break should keep each session's rounds together
    # when ordered by recency rather than interleaving s1/s2 round indices.
    newest = sorted(memories, key=lambda x: x["recency_order"], reverse=True)
    ordered_sessions = [m["session_id"] for m in newest]
    assert ordered_sessions == ["s2", "s2", "s1", "s1"]

    print("Time-handling sanity checks passed.")


if __name__ == "__main__":
    main()
