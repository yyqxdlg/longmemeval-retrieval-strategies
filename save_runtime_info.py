from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

from experiment_io import runtime_snapshot


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Save UTF-8 runtime provenance for one experiment run."
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    snapshot = runtime_snapshot()
    (output_dir / "runtime.json").write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "python_version.txt").write_text(
        sys.version.replace("\n", " ") + "\n",
        encoding="utf-8",
    )

    distributions = sorted(
        {
            f"{dist.metadata['Name']}=={dist.version}"
            for dist in importlib.metadata.distributions()
            if dist.metadata.get("Name")
        },
        key=str.lower,
    )
    (output_dir / "environment.txt").write_text(
        "\n".join(distributions) + "\n",
        encoding="utf-8",
    )

    try:
        result = subprocess.run(
            ["nvidia-smi"],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        gpu_text = result.stdout or result.stderr
    except FileNotFoundError:
        gpu_text = "nvidia-smi is not available on this machine.\n"
    (output_dir / "gpu_info.txt").write_text(gpu_text, encoding="utf-8")

    print(f"Runtime information written to: {output_dir}")


if __name__ == "__main__":
    main()
