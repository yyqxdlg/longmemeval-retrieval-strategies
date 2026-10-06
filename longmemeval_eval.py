from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


KEY_FIELDS = ("question_id", "strategy", "k")


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def export_hypotheses(input_path: Path, output_path: Path) -> int:
    _, rows = read_csv(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            answer = str(row.get("generated_answer", "")).strip()
            if not answer:
                continue
            payload = {
                "question_id": str(row["question_id"]),
                "hypothesis": answer,
                "strategy": str(row["strategy"]),
                "k": int(row["k"]),
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            written += 1
    return written


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc
    return rows


def score_key(row: dict[str, Any]) -> tuple[str, str, int]:
    missing = [field for field in KEY_FIELDS if field not in row]
    if missing:
        raise ValueError(
            "Evaluation log must preserve question_id, strategy, and k; missing "
            + ", ".join(missing)
        )
    return str(row["question_id"]), str(row["strategy"]), int(row["k"])


def parse_label(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1", "correct"}:
            return True
        if normalized in {"false", "no", "0", "incorrect"}:
            return False
    raise ValueError(f"Unsupported answer-correctness label: {value!r}")


def merge_scores(
    generation_path: Path,
    evaluation_path: Path,
    output_path: Path,
    *,
    require_complete: bool = True,
) -> tuple[int, int, int]:
    fields, generation_rows = read_csv(generation_path)
    scores: dict[tuple[str, str, int], tuple[bool, str]] = {}
    for entry in read_jsonl(evaluation_path):
        key = score_key(entry)
        label_data = entry.get("autoeval_label")
        if not isinstance(label_data, dict) or "label" not in label_data:
            raise ValueError(f"Missing autoeval_label for {key}")
        if key in scores:
            raise ValueError(f"Duplicate evaluation result for {key}")
        scores[key] = (
            parse_label(label_data["label"]),
            str(label_data.get("model", "")),
        )

    expected_keys = {
        (str(row["question_id"]), str(row["strategy"]), int(row["k"]))
        for row in generation_rows
        if str(row.get("generated_answer", "")).strip()
    }
    missing_keys = expected_keys - set(scores)
    if missing_keys and require_complete:
        preview = ", ".join(str(key) for key in sorted(missing_keys)[:5])
        raise ValueError(
            f"Missing judge labels for {len(missing_keys)} generated answers: {preview}"
        )

    output_fields = list(fields)
    for field in ("answer_correct", "judge_model"):
        if field not in output_fields:
            output_fields.append(field)

    matched = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields)
        writer.writeheader()
        for row in generation_rows:
            key = (str(row["question_id"]), str(row["strategy"]), int(row["k"]))
            if key in scores:
                label, judge_model = scores[key]
                row["answer_correct"] = int(label)
                row["judge_model"] = judge_model
                matched += 1
            writer.writerow(row)

    unused = len(scores) - matched
    return matched, unused, len(missing_keys)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export hypotheses for and merge scores from LongMemEval's official judge."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export")
    export_parser.add_argument("--input", required=True)
    export_parser.add_argument("--output", required=True)

    merge_parser = subparsers.add_parser("merge")
    merge_parser.add_argument("--generation-results", required=True)
    merge_parser.add_argument("--evaluation-log", required=True)
    merge_parser.add_argument("--output", required=True)
    merge_parser.add_argument("--allow-partial", action="store_true")

    args = parser.parse_args()
    if args.command == "export":
        count = export_hypotheses(Path(args.input), Path(args.output))
        print(f"Exported {count} hypotheses to {args.output}")
    else:
        matched, unused, missing = merge_scores(
            Path(args.generation_results),
            Path(args.evaluation_log),
            Path(args.output),
            require_complete=not args.allow_partial,
        )
        print(f"Merged {matched} judged answers into {args.output}")
        if unused:
            print(f"WARNING: {unused} evaluation rows did not match generation rows")
        if missing:
            print(f"WARNING: {missing} generated answers have no judge label")


if __name__ == "__main__":
    main()
