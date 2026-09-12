"""Run documented BSPC pipeline stages in a fixed dependency order."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable

STAGES: dict[str, list[list[str]]] = {
    "preflight": [
        [PYTHON, "src/env_check.py"],
        [PYTHON, "scripts/smoke_models.py"],
    ],
    "private-preprocessing": [
        [PYTHON, "src/build_manifest.py"],
        [PYTHON, "src/extract_signals.py", "--source", "all", "--workers", "4"],
        [PYTHON, "src/prepare_dataset.py"],
        [PYTHON, "src/validate_artifacts.py"],
    ],
    "primary": [
        [
            PYTHON,
            "src/run_bspc_extension_experiments.py",
            "--seeds",
            "20260907,20260908,20260909",
            "--bootstrap-repeats",
            "5000",
        ],
        [
            PYTHON,
            "src/run_model_suite_extension.py",
            "--seeds",
            "20260907,20260908,20260909",
            "--bootstrap-repeats",
            "5000",
        ],
    ],
    "analysis": [
        [PYTHON, "src/revision_audit.py"],
        [PYTHON, "src/audit_model_suite_outputs.py"],
        [PYTHON, "src/profile_models.py"],
    ],
    "figures": [
        [PYTHON, "figures/make_journal_figures.py"],
        [PYTHON, "figures/make_five_model_split_figures.py"],
    ],
    "exploratory": [
        [PYTHON, "src/train_classical.py"],
        [PYTHON, "src/train_deep.py"],
        [PYTHON, "src/summarize_results.py"],
        [PYTHON, "src/train_snn_fusion.py"],
        [PYTHON, "src/train_snn_fusion_nested.py"],
        [PYTHON, "src/train_public_luh.py"],
        [PYTHON, "src/summarize_public_luh.py"],
        [PYTHON, "src/train_public_luh_optimized.py"],
        [PYTHON, "src/train_public_luh_v2.py"],
        [PYTHON, "src/summarize_public_luh_optimization.py"],
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=[*STAGES, "all"],
        required=True,
        help="Pipeline stage to run; all excludes post hoc exploratory experiments.",
    )
    parser.add_argument(
        "--config",
        default="config/default.json",
        help="Configuration path relative to the repository root, or an absolute path.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = Path(args.config)
    if not config.is_absolute():
        config = ROOT / config
    env = os.environ.copy()
    env["BP_CONFIG"] = str(config.resolve())

    selected = (
        ["preflight", "private-preprocessing", "primary", "analysis", "figures"]
        if args.stage == "all"
        else [args.stage]
    )
    for stage in selected:
        print(f"\n[{stage}]", flush=True)
        for command in STAGES[stage]:
            shown = subprocess.list2cmdline(command)
            print(f"$ {shown}", flush=True)
            if not args.dry_run:
                subprocess.run(command, cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
