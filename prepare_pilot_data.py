import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path


TASK_MAP = {
    "single-session-user": "IE",
    "single-session-assistant": "IE",
    "single-session-preference": "IE",
    "multi-session": "MR",
    "knowledge-update": "KU",
    "temporal-reasoning": "TR",
}

TASK_ORDER = ["IE", "MR", "KU", "TR"]

CLASSIC_CANDIDATES = [
    "longmemeval_s_cleaned.json",
    "longmemeval_s.json",
    "longmemeval_m_cleaned.json",
    "longmemeval_m.json",
]


def detect_v2(data_root: Path) -> bool:
    return (
        (data_root / "questions.jsonl").exists()
        and (data_root / "trajectories.jsonl").exists()
    )


def find_classic_input(data_root: Path, explicit_input: str | None) -> Path:
    if explicit_input:
        path = Path(explicit_input)
        if not path.is_absolute():
            candidate = data_root / path
            if candidate.exists():
                path = candidate
        if not path.exists():
            raise FileNotFoundError(f"Input file does not exist: {path}")
        return path

    for name in CLASSIC_CANDIDATES:
        path = data_root / name
        if path.exists():
            return path

    if detect_v2(data_root):
        raise RuntimeError(
            "\nYou supplied LongMemEval-V2, but your current Research Plan "
            "requires the classic LongMemEval task categories IE/MR/KU/TR.\n"
            "LongMemEval-V2 has a different benchmark design and different "
            "question types.\n\n"
            "Download the classic cleaned dataset instead:\n"
            "  git clone https://huggingface.co/datasets/"
            "xiaowu0162/longmemeval-cleaned\n\n"
            "Then run this script with that directory."
        )

    raise FileNotFoundError(
        "No classic LongMemEval data file found. Expected one of:\n  "
        + "\n  ".join(CLASSIC_CANDIDATES)
    )


def normalize_task(record: dict) -> str | None:
    question_id = str(record.get("question_id", ""))
    if question_id.endswith("_abs"):
        return None

    qtype = record.get("question_type")
    return TASK_MAP.get(qtype)


def _sample_ie_stratified(
    records: list[dict],
    n: int,
    rng: random.Random,
) -> list[dict]:
    """Balance IE across user / assistant / preference as evenly as possible."""
    subtypes = [
        "single-session-user",
        "single-session-assistant",
        "single-session-preference",
    ]
    groups = {qtype: [] for qtype in subtypes}
    for record in records:
        qtype = record.get("question_type")
        if qtype in groups:
            groups[qtype].append(record)

    base = n // len(subtypes)
    remainder = n % len(subtypes)
    order = list(subtypes)
    rng.shuffle(order)
    allocation = {qtype: base for qtype in subtypes}
    for qtype in order[:remainder]:
        allocation[qtype] += 1

    chosen = []
    for qtype in subtypes:
        candidates = sorted(
            groups[qtype], key=lambda x: str(x.get("question_id", ""))
        )
        need = allocation[qtype]
        if len(candidates) < need:
            raise RuntimeError(
                f"Not enough {qtype} IE questions: need {need}, "
                f"found {len(candidates)}"
            )
        chosen.extend(rng.sample(candidates, need))
    return chosen


def sample_records(
    data: list[dict],
    per_task: int,
    seed: int,
    *,
    stratify_ie: bool = False,
) -> list[dict]:
    grouped = {task: [] for task in TASK_ORDER}

    for record in data:
        task = normalize_task(record)
        if task is not None:
            grouped[task].append(record)

    rng = random.Random(seed)
    selected = []

    for task in TASK_ORDER:
        candidates = sorted(
            grouped[task],
            key=lambda x: str(x.get("question_id", "")),
        )

        if len(candidates) < per_task:
            raise RuntimeError(
                f"Not enough {task} questions: "
                f"need {per_task}, found {len(candidates)}"
            )

        if task == "IE" and stratify_ie:
            chosen = _sample_ie_stratified(candidates, per_task, rng)
        else:
            chosen = rng.sample(candidates, per_task)

        for record in chosen:
            copied = dict(record)
            copied["_pilot_task"] = task
            selected.append(copied)

    return selected


def write_manifest(records: list[dict], path: Path) -> None:
    fields = [
        "question_id",
        "pilot_task",
        "question_type",
        "question",
        "answer",
        "question_date",
        "num_sessions",
        "num_answer_sessions",
    ]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for x in records:
            writer.writerow(
                {
                    "question_id": x.get("question_id", ""),
                    "pilot_task": x.get("_pilot_task", ""),
                    "question_type": x.get("question_type", ""),
                    "question": x.get("question", ""),
                    "answer": x.get("answer", ""),
                    "question_date": x.get("question_date", ""),
                    "num_sessions": len(x.get("haystack_sessions", [])),
                    "num_answer_sessions": len(
                        x.get("answer_session_ids", [])
                    ),
                }
            )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Select a reproducible 20-question pilot from classic LongMemEval."
        )
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument(
        "--input",
        default=None,
        help=(
            "Optional filename/path. If omitted, prefers "
            "longmemeval_s_cleaned.json."
        ),
    )
    parser.add_argument("--per-task", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--stratify-ie",
        action="store_true",
        help=(
            "Balance IE across single-session-user / assistant / preference. "
            "Recommended for the larger final sample; leave off if you want to "
            "preserve the already-checked 20-question pilot selected earlier."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/pilot",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    input_path = find_classic_input(data_root, args.input)

    print(f"Loading: {input_path}")
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise RuntimeError(
            "Expected classic LongMemEval JSON to contain a list of records."
        )

    counts = Counter(
        task
        for task in (normalize_task(x) for x in data)
        if task is not None
    )
    print("Available non-abstention questions:")
    for task in TASK_ORDER:
        print(f"  {task}: {counts[task]}")

    selected = sample_records(
        data,
        per_task=args.per_task,
        seed=args.seed,
        stratify_ie=args.stratify_ie,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    full_path = out_dir / "pilot_questions.json"
    ids_path = out_dir / "pilot_ids.json"
    manifest_path = out_dir / "pilot_manifest.csv"

    with full_path.open("w", encoding="utf-8") as f:
        json.dump(selected, f, ensure_ascii=False, indent=2)

    id_payload = {
        "seed": args.seed,
        "per_task": args.per_task,
        "stratify_ie": args.stratify_ie,
        "source_file": str(input_path),
        "questions": [
            {
                "question_id": x.get("question_id"),
                "task_type": x.get("_pilot_task"),
                "question_type": x.get("question_type"),
            }
            for x in selected
        ],
    }

    with ids_path.open("w", encoding="utf-8") as f:
        json.dump(id_payload, f, ensure_ascii=False, indent=2)

    write_manifest(selected, manifest_path)

    selected_counts = Counter(x["_pilot_task"] for x in selected)

    print("\nSelected pilot:")
    for task in TASK_ORDER:
        print(f"  {task}: {selected_counts[task]}")

    print(f"\nSaved full records: {full_path}")
    print(f"Saved IDs:          {ids_path}")
    print(f"Saved manifest:     {manifest_path}")


if __name__ == "__main__":
    main()
