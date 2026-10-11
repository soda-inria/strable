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

from configs.exp_configs import estim_configs  # noqa: E402
from configs.path_configs import compiled_results_path, path_configs  # noqa: E402

TUNE_INDICATORS = ("default", "tune")


def tune_indicator_of(score_file, method):
    """``-ti`` value of the run that wrote ``score_file``.

    The evaluate scripts save each score under
    ``<dataset>/<method>_<tune_indicator>[_<flags>]/score/``.
    """
    method_marker = Path(score_file).parent.parent.name
    tune_indicator = method_marker[len(method) + 1:].split("_")[0]
    if not method_marker.startswith(f"{method}_") or tune_indicator not in TUNE_INDICATORS:
        raise ValueError(
            f"{score_file}: folder {method_marker!r} is not "
            f"'{method}_<{'|'.join(TUNE_INDICATORS)}>'"
        )
    return tune_indicator


def label_method(method, tune_indicator):
    """Method label used in the compiled CSV and by the figures.

    Learners with an ``estim_configs`` entry get ``_<tune_indicator>``
    (``num-str_tabvec_xgb_default``, ``..._tune``); the others (ContextTab,
    TabSTAR, TabICLv2, TabM, RealMLP, Mambular) keep the bare name, as in the
    paper's compiled CSVs. A name that already ends in a tune indicator is
    returned unchanged.
    """
    last = method.rsplit("_", 1)[-1]
    if last in TUNE_INDICATORS or last not in estim_configs:
        return method
    return f"{method}_{tune_indicator}"


def read_score(score_file):
    """One score CSV with its ``method`` column labelled by ``label_method``."""
    df = pd.read_csv(score_file)
    df["method"] = [
        label_method(m, tune_indicator_of(score_file, m)) for m in df["method"]
    ]
    return df


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

    rows = Parallel(n_jobs=-1)(delayed(read_score)(p) for p in score_files)
    df = pd.concat(rows, axis=0).reset_index(drop=True)

    out = args.out or compiled_results_path(args.run_name)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows to {out}")


if __name__ == "__main__":
    main()
