import argparse
import json
from collections import Counter
from pathlib import Path


CLASSIC_FILES = [
    "longmemeval_s_cleaned.json",
    "longmemeval_s.json",
    "longmemeval_m_cleaned.json",
    "longmemeval_m.json",
    "longmemeval_oracle.json",
]


def inspect_v2(root: Path) -> bool:
    qfile = root / "questions.jsonl"
    tfile = root / "trajectories.jsonl"
    if not (qfile.exists() and tfile.exists()):
        return False

    counter = Counter()
    count = 0
    with qfile.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            counter[obj.get("question_type", "<missing>")] += 1
            count += 1

    print("Dataset detected: LongMemEval-V2")
    print(f"Questions: {count}")
    print("question_type distribution:")
    for key, value in sorted(counter.items()):
        print(f"  {key}: {value}")

    print(
        "\nIMPORTANT: LongMemEval-V2 does not use the classic "
        "IE/MR/KU/TR categories from your current Research Plan."
    )
    return True


def inspect_classic(root: Path) -> bool:
    for name in CLASSIC_FILES:
        path = root / name
        if not path.exists():
            continue

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        counter = Counter(x.get("question_type", "<missing>") for x in data)
        abstention = sum(
            str(x.get("question_id", "")).endswith("_abs") for x in data
        )

        print("Dataset detected: classic LongMemEval")
        print(f"File: {path}")
        print(f"Questions: {len(data)}")
        print(f"Abstention (_abs): {abstention}")
        print("question_type distribution:")
        for key, value in sorted(counter.items()):
            print(f"  {key}: {value}")
        return True

    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    args = parser.parse_args()

    root = Path(args.data_root)

    if not root.exists():
        raise SystemExit(f"Directory does not exist: {root}")

    if inspect_v2(root):
        return

    if inspect_classic(root):
        return

    print("Could not recognize the dataset.")
    print("Files in directory:")
    for path in sorted(root.iterdir()):
        print(f"  {path.name}")


if __name__ == "__main__":
    main()
