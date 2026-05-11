"""Aggregate per-fold scores from ``script_evaluate*.py`` into one CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from configs.path_configs import path_configs  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile per-fold scores into one CSV.")
    parser.add_argument(
        "run_name",
        help=(
            "Path under <STRABLE_ROOT>/results/, of the form "
            "<ABLATION>/<save_dir> (e.g. 'default/benchmark_main', "
            "'no-pca/qwen_runs', 'ct30-ohe/main'). The ABLATION segment "
            "is hardcoded by each evaluate script; see its top-of-file "
            "ABLATION constant."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Output CSV path. Defaults to "
            "<STRABLE_ROOT>/results/compiled_results/"
            "result_<run_name with / -> _>.csv."
        ),
    )
    args = parser.parse_args()

    score_dir = Path(path_configs["results"]) / args.run_name
    if not score_dir.is_dir():
        raise SystemExit(f"Results directory not found: {score_dir}")

    score_files = list(score_dir.glob("**/score/*.csv"))
    if not score_files:
        raise SystemExit(f"No score CSVs found under {score_dir}")
    print(f"Found {len(score_files)} score files under {score_dir}")

    rows = Parallel(n_jobs=-1)(delayed(pd.read_csv)(p) for p in score_files)
    df = pd.concat(rows, axis=0).reset_index(drop=True)

    flat_name = args.run_name.replace("/", "_")
    out = args.out or (
        Path(path_configs["compiled_results"]) / f"result_{flat_name}.csv"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows to {out}")


if __name__ == "__main__":
    main()
