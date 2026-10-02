import argparse
import json
from collections import Counter
from pathlib import Path

from memory_adapter import temporal_diagnostics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-file", required=True)
    args = parser.parse_args()

    records = json.loads(Path(args.pilot_file).read_text(encoding="utf-8"))

    task_counts = Counter()
    unsorted = []
    future = []
    future_gold = []
    same_day_later = []
    same_day_later_gold = []

    for record in records:
        qid = record.get("question_id")
        task = record.get("_pilot_task")
        task_counts[task] += 1
        diag = temporal_diagnostics(record)

        if not diag["sorted_by_date"]:
            unsorted.append((qid, task))
        if diag["num_future_sessions"]:
            future.append((qid, task, diag["num_future_sessions"]))
        if diag["num_future_gold_sessions"]:
            future_gold.append(
                (qid, task, diag["future_gold_session_ids"])
            )
        if diag["num_same_day_later_sessions"]:
            same_day_later.append(
                (qid, task, diag["num_same_day_later_sessions"])
            )
        if diag["num_same_day_later_gold_sessions"]:
            same_day_later_gold.append(
                (
                    qid,
                    task,
                    diag["same_day_later_gold_session_ids"],
                )
            )

    print("Task counts:", dict(task_counts))
    print(f"Unsorted histories: {len(unsorted)}/{len(records)}")
    for row in unsorted:
        print("  UNSORTED", row)

    print(
        "Questions with sessions on a STRICTLY LATER CALENDAR DAY: "
        f"{len(future)}/{len(records)}"
    )
    for row in future:
        print("  FUTURE_DAY", row)

    print(
        "Questions with GOLD sessions on a strictly later calendar day: "
        f"{len(future_gold)}/{len(records)}"
    )
    for row in future_gold:
        print("  FUTURE_DAY_GOLD", row)

    print(
        "Questions with same-day sessions later by HH:MM (retained): "
        f"{len(same_day_later)}/{len(records)}"
    )
    for row in same_day_later:
        print("  SAME_DAY_LATER", row)

    print(
        "Questions with same-day GOLD sessions later by HH:MM (retained): "
        f"{len(same_day_later_gold)}/{len(records)}"
    )
    for row in same_day_later_gold:
        print("  SAME_DAY_LATER_GOLD", row)


if __name__ == "__main__":
    main()
