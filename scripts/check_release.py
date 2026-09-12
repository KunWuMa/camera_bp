"""Fail closed when a GitHub release tree contains likely research data."""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

BANNED_SUFFIXES = {
    ".avi", ".ckpt", ".csv", ".h5", ".hdf5", ".joblib", ".mat", ".mkv",
    ".mov", ".mp4", ".npy", ".npz", ".onnx", ".pickle", ".pkl", ".pt",
    ".pth", ".tsv", ".wav", ".xls", ".xlsx",
}
BANNED_NAMES = {
    ".env", "id_mapping.txt", "identifier_lookup.csv", "subject_lookup.csv",
}
TEXT_SUFFIXES = {"", ".bat", ".gitignore", ".json", ".md", ".py", ".txt", ".yml", ".yaml"}
ABSOLUTE_WINDOWS_PATH = re.compile(r"(?i)(?:^|[\"'=(\s])[a-z]:[\\/]")
MAX_PUBLIC_FILE_BYTES = 5 * 1024 * 1024


def main() -> None:
    failures: list[str] = []
    checked = 0
    for path in ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.parts or "__pycache__" in path.parts:
            continue
        checked += 1
        relative = path.relative_to(ROOT)
        lower_name = path.name.lower()

        if path.suffix.lower() in BANNED_SUFFIXES:
            failures.append(f"restricted file type: {relative}")
        if lower_name in BANNED_NAMES or lower_name.startswith("local."):
            failures.append(f"secret/local file name: {relative}")
        if relative.parts and relative.parts[0] == "data" and path.name != "README.md":
            failures.append(f"data directory may contain README.md only: {relative}")
        if path.stat().st_size > MAX_PUBLIC_FILE_BYTES:
            failures.append(f"file exceeds 5 MiB review threshold: {relative}")

        if path.suffix.lower() in TEXT_SUFFIXES:
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                failures.append(f"unreadable text file: {relative}")
                continue
            if ABSOLUTE_WINDOWS_PATH.search(text):
                failures.append(f"likely Windows absolute path in text: {relative}")

    if failures:
        print("RELEASE CHECK FAILED")
        for failure in sorted(set(failures)):
            print(f"- {failure}")
        raise SystemExit(1)
    print(f"RELEASE CHECK PASSED: {checked} files inspected; no data files detected.")


if __name__ == "__main__":
    main()
