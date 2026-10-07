from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def runtime_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "packages": {
            name: package_version(name)
            for name in (
                "torch",
                "transformers",
                "sentence-transformers",
                "accelerate",
                "bitsandbytes",
                "numpy",
                "pandas",
                "scipy",
            )
        },
    }

    try:
        import torch

        snapshot["cuda_available"] = bool(torch.cuda.is_available())
        snapshot["cuda_version"] = torch.version.cuda
        snapshot["cudnn_version"] = (
            torch.backends.cudnn.version() if torch.cuda.is_available() else None
        )
        snapshot["gpu_names"] = [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ]
    except Exception as exc:  # Runtime capture must not block an experiment.
        snapshot["torch_runtime_error"] = repr(exc)

    repo_root = Path(__file__).resolve().parent
    git_prefix = [
        "git",
        "-c",
        f"safe.directory={repo_root.as_posix()}",
        "-C",
        str(repo_root),
    ]
    try:
        commit = subprocess.run(
            [*git_prefix, "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            [*git_prefix, "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        snapshot["git_commit"] = commit
        snapshot["git_dirty"] = bool(dirty)
    except (FileNotFoundError, subprocess.SubprocessError):
        snapshot["git_commit"] = None
        snapshot["git_dirty"] = None

    return snapshot


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def merge_resume_metadata(
    existing: dict[str, Any],
    current: dict[str, Any],
    *,
    setting_keys: Iterable[str],
) -> dict[str, Any]:
    """Validate fixed settings and merge k values/run history for a resume."""
    mismatches = []
    for key in setting_keys:
        if key in existing and existing.get(key) != current.get(key):
            mismatches.append(
                f"{key}: existing={existing.get(key)!r}, current={current.get(key)!r}"
            )

    for key in ("input_sha256", "retrieval_results_sha256"):
        if key in existing and key in current and existing[key] != current[key]:
            mismatches.append(
                f"{key}: existing={existing[key]!r}, current={current[key]!r}"
            )

    if mismatches:
        details = "\n  - ".join(mismatches)
        raise ValueError(
            "Refusing to mix incompatible experiment settings while resuming:\n"
            f"  - {details}\nUse a new output file for the new condition."
        )

    merged = dict(existing)
    merged.update({k: v for k, v in current.items() if k != "run_history"})
    merged["k_values"] = sorted(
        {
            int(value)
            for value in existing.get("k_values", []) + current.get("k_values", [])
        }
    )
    merged["run_history"] = list(existing.get("run_history", [])) + list(
        current.get("run_history", [])
    )
    return merged
