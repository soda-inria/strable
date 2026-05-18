# AGENTS.md — STRABLE for coding agents

This file is the entry point for coding agents (Claude Code, Codex, Cursor, Gemini CLI, Aider, …) working in this repo. It complements [README.md](README.md): the README is the human walkthrough; this file is the operational reference an agent can lift commands and rules from verbatim.

> **What is STRABLE?** A benchmark of 108 real-world tables with strings, plus pipelines (modular encoder → tabular learner, and end-to-end) for evaluating them. See README.md and the paper ([arXiv:2605.12292](https://arxiv.org/abs/2605.12292)) for context.

## Typical agent-handled requests

| If the user asks… | Go to |
|---|---|
| "Set up the repo / install" | [Setup](#setup) |
| "Reproduce paper figure / table N" | [Reproducing paper artifacts](#reproducing-paper-artifacts) |
| "Run STRABLE on my own model / encoder" | [Adding a new pipeline](#adding-a-new-pipeline-component) |
| "Just compare my model on the same tables" | [Use as a dataset only](#use-strable-as-a-dataset-only) |
| "Why is my run not producing output / where do results go?" | [Output layout](#output-layout) and [Common pitfalls](#common-pitfalls) |
| "Run on a single dataset / fold to test" | [Smoke test](#smoke-test) |

---

## Guardrails — do not violate

1. **Never edit `data/data_processed/`** (or `data/data_processed_FULL/`, `data/data_processed_feature_eng/`, `data/data_processed_skewness_inverse_transformation/`). These are downloaded immutable inputs. To regenerate, re-run the download script — do not hand-patch.
2. **Never commit `results/`, `results_compiled/`, `results_pics/`, `results_tables/`, `__cache__/`, `data/llm_embeding/`, `data/data_raw/`, `data/data_processed*/`.** They are runtime outputs/inputs and gitignored. If a path appears tracked, check `.gitignore` rather than `git rm`.
3. **Never hardcode absolute paths.** All paths flow through [`configs/path_configs.py`](configs/path_configs.py), which derives from `$STRABLE_ROOT` (env var) or the repo root. To override, export `STRABLE_ROOT=/your/path` — do not edit `path_configs.py`.
4. **Don't append `_default` / `_tune` to method names.** The suffix is added automatically from `--tune_indicator`. Pass `-m num-str_tabvec_xgb -ti tune`, not `-m num-str_tabvec_xgb_tune`.
5. **Don't skip the embedding-extraction step for LLM pipelines.** `script_evaluate*.py` will fail (or silently fall back) if `data/llm_embeding/<model>/…` is missing for the requested LLM encoder. See [The two-stage rule](#the-two-stage-rule-llm-pipelines).
6. **Don't introduce a new ablation folder.** The eight scripts in `scripts/evaluate_scripts/` already cover every ablation in the paper (each has a hard-coded `ABLATION` tag that determines `results/<ABLATION>/…`).

---

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export STRABLE_ROOT=$(pwd)
export HF_TOKEN=hf_<user-token>           # required for HuggingFace download + gated LLMs
export HF_HOME=${HF_HOME:-~/.cache/huggingface}
python data/download_datasets.py          # downloads the 108-table default variant
```

Python 3.12 required. ContextTab needs a separate install (see README §Install). GPU (`cuda`) is recommended for any LLM encoder and for TabPFN-2.5 / TabICLv2 / TabSTAR / ContextTab; everything else runs on `cpu`.

---

## The two-stage rule (LLM pipelines)

LLM-encoder pipelines run in **two stages** that share a cache. Skip this if the user only wants Tf-Idf / TargetEncoder / Tarte or an end-to-end learner (CatBoost / ContextTab / TabSTAR / Mambular).

**Stage A — pre-compute embeddings** (writes to `data/llm_embeding/<encoder>/<dataset>/`):

```bash
python scripts/embedding_extraction_scripts/script_extract_llm_embeddings.py \
    -m llm-qwen3-8b -dn all-wide -ns 3 -fi all -dv cuda -cf False -oc False
```

For the ablation variants there are sibling scripts: `..._FULL.py` and `..._feat_eng.py` (see [Ablation matrix](#ablation-matrix)).

**Stage B — run the learner** (reads the cache, runs PCA + learner, writes `results/<ABLATION>/…`):

```bash
python scripts/evaluate_scripts/script_evaluate.py \
    --save_dir benchmark_main -m num-str_llm-qwen3-8b_tabpfn \
    -dn all-wide -ns 3 -fi all -dv cuda -cf False -oc False -ti default
```

If Stage A is missing the cache, Stage B will recompute it on-the-fly — fine for one dataset but wasteful at full scale.

---

## CLI flag grammar (`scripts/evaluate_scripts/script_evaluate*.py`)

All eight `script_evaluate*.py` share this core signature. Only `script_evaluate_impute_missing_val.py` adds one extra flag (`-mvp`).

| Flag | Long | Type | Meaning |
|---|---|---|---|
| `-dn` | `--data_name` | `str+` | Dataset name(s). Use the literal `all-wide` to expand to the full 108-table list (`data_list_wide` in [`configs/exp_configs.py`](configs/exp_configs.py)). |
| `-ns` | `--n_split` | `str+` | Outer CV folds. Paper uses `3`. |
| `-m`  | `--method` | `str+` | Method token `{dtype}_{encoder}_{learner}` — see [`scripts/evaluate_scripts/METHOD_NAMES.md`](scripts/evaluate_scripts/METHOD_NAMES.md). |
| `-fi` | `--fold_index` | `str+` | `all` for every fold, or e.g. `0` for fold 0 only. |
| `-dv` | `--device` | `str` | `cpu` or `cuda`. |
| `-cf` | `--check_result_flag` | `str` | `True` skips work if the target CSV already exists. Use `False` to overwrite. |
| `-oc` | `--override_cache` | `str` | `True` recomputes embeddings/encodings even if cached. Default `False`. |
| `-ti` | `--tune_indicator` | `str+` | `default` (no HP search), `tune` (random search, 100 iters), or `all` (both). Appended to method name in the output path. |
| `-sd` | `--save_dir` | `str` | Sub-folder under `results/<ABLATION>/`. **Required.** |
| `-norm` | `--normalization` | `bool` | StandardScaler before PCA. Default `False`. Paper's StandScal+PCA variants use `True`. |
| `-nopca` | `--no_pca` | `bool` | Skip PCA, keep first `--n_dimensions` raw dims. Switches `ABLATION` from `default` to `no-pca` in `script_evaluate.py`. |
| `-ndim` | `--n_dimensions` | `int` | Number of components / sliced dims. Default `30`. |
| `-mvp` | `--missing_val_policy` | `str` | **Only `script_evaluate_impute_missing_val.py`.** One of `none`, `drop_all`, `uniform_impute`. |
|       | `--dry-run` | flag | Limit to the first dataset and fold 0 only. Use for smoke testing. |

`-dn`, `-ns`, `-m`, `-fi`, `-ti` accept multiple values and the script Cartesian-products them via `sklearn.ParameterGrid`.

### Method-name slots (`-m`)

`{dtype}_{encoder}_{learner}` — full enumeration in [`scripts/evaluate_scripts/METHOD_NAMES.md`](scripts/evaluate_scripts/METHOD_NAMES.md). Quick reference:

- **dtype** — `num-str` (both), `num-only` (drop strings), `str-only` (drop numbers).
- **encoder** — `tabvec` (Tf-Idf+SVD), `tarenc` (TargetEncoder), `tarte`, or `llm-<name>` (41 LM keys in [`configs/exp_configs.py`](configs/exp_configs.py), e.g. `llm-qwen3-8b`, `llm-llama-3.1-8b`, `llm-all-MiniLM-L6-v2`, `llm-fasttext`, `llm-e5-small-v2`, `llm-jasper-0.6b`).
- **learner** — modular: `ridge`, `xgb`, `extrees`, `tabpfn`, `tabicl`, `tabm`, `realmlp`. E2E (encoder field equals learner): `catboost`, `contexttab`, `tabstar`, `mambular`.

For E2E learners the method is `num-str_<name>_<name>`, e.g. `num-str_contexttab_contexttab`.

---

## Ablation matrix

Each script writes to a hard-coded `ABLATION` folder. Match the user's intent to a script before running.

| Paper figure / claim | Script | `ABLATION` tag | Data folder it reads | Notes |
|---|---|---|---|---|
| Main results (Fig. 3, 4a, 6) | `script_evaluate.py` | `default` | `data/data_processed/` | The workhorse. |
| No-PCA / StandScal+PCA (Fig. 2, E.4) | `script_evaluate.py` with `--no_pca True` or `--normalization True` | `no-pca` (when `--no_pca True`), else `default` | same | Re-uses cached embeddings; only post-processing differs. |
| CT=30, OHE branch (Fig. E.11) | `script_evaluate_30_thres_OHE.py` | `ct30-ohe` | same | Low-card columns → OHE, high-card → LLM. Ridge family. |
| CT=30, passthrough branch (Fig. E.11) | `script_evaluate_30_thres_lowcard_passthrough.py` | `ct30-passthrough` | same | XGBoost / TabPFN family. |
| Full-data ablation (Fig. E.10) | `script_evaluate_FULL.py` | `full-data` | `data/data_processed_FULL/` | Needs FULL-variant embeddings via `script_extract_llm_embeddings_FULL.py`. |
| Feature-engineering ablation (Fig. E.7) | `script_evaluate_feature_eng.py` | `feature-eng` | `data/data_processed_feature_eng/` | Needs feat-eng-variant embeddings via `script_extract_llm_embeddings_feat_eng.py`. |
| Raw-target ablation (Fig. E.8) | `script_evaluate_skewness.py` | `no-target-transform` | `data/data_processed_skewness_inverse_transformation/` | |
| Imputation ablation (Fig. E.9) | `script_evaluate_impute_missing_val.py` | `imputed` | `data/data_processed/` | Adds `-mvp`. |

Ablation data variants (`full`, `feature-eng`, `raw-targets`) are downloaded via:

```bash
python scripts/download_datasets.py --variants default,full,feature-eng,raw-targets
```

---

## Output layout

```
results/
  <ABLATION>/                 # default, no-pca, ct30-ohe, ct30-passthrough,
                              # full-data, feature-eng, no-target-transform, imputed
    <save_dir>/               # whatever you passed to --save_dir
      <dataset>/
        <method>_<ti><suffix>/   # e.g. num-str_tabvec_xgb_default, _tune, _norm, _pc60
          score/
            <dataset>|<method>_<ti><suffix>|<n>-cv|idx-<fold>.csv
  compiled_results/
    result_<ABLATION>_<save_dir>.csv     # produced by scripts/compile_results.py
```

Compilation step (always runs after evaluation):

```bash
python scripts/compile_results.py <ABLATION>/<save_dir>
# e.g.
python scripts/compile_results.py default/benchmark_main
```

Figures consume `results/compiled_results/*.csv` and write to `results_pics/<YYYY-MM-DD>/`. Each figure script is independent:

```bash
python figures/main_paper/figure_3.py
python figures/appendix/figure_E10.py
```

---

## Reproducing paper artifacts

Recipe for one figure (Fig. 3, CD diagram of encoder–learner pipelines):

```bash
# 1. Data
python data/download_datasets.py

# 2. Embeddings for each LLM encoder you care about (one call per encoder)
python scripts/embedding_extraction_scripts/script_extract_llm_embeddings.py \
    -m llm-all-MiniLM-L6-v2 -dn all-wide -ns 3 -fi all -dv cuda -cf False -oc False

# 3. Pipelines for each (encoder × learner × ti) combo. Tf-Idf example, tuned:
python scripts/evaluate_scripts/script_evaluate.py \
    --save_dir benchmark_main -m num-str_tabvec_xgb \
    -dn all-wide -ns 3 -fi all -dv cpu -cf False -oc False -ti tune

# 4. Compile
python scripts/compile_results.py default/benchmark_main

# 5. Figure
python figures/main_paper/figure_3.py
```

The full set of method-encoder combinations evaluated in the paper is enumerated in `METHOD_NAMES.md` (§ "Valid combinations"). The paper used **~445 pipelines × 108 datasets × 3 folds** — total ~842 CPU+GPU days. For reproduction at scale, parallelize across (dataset, method, fold) on a cluster; the scripts are idempotent when `-cf True` is set.

---

## Use STRABLE as a dataset only

If the user just wants the 108 tables for their own evaluation (no STRABLE pipelines):

```bash
python data/download_datasets.py
# tables land in data/data_processed/<dataset>/
# each dataset has: data.parquet, info.json, and CV split indices
```

Then they can read them directly with pandas / pyarrow. Don't pull in `src/encoding.py` or the evaluate scripts unless they want to compare against STRABLE's published pipeline numbers — in which case follow [The two-stage rule](#the-two-stage-rule-llm-pipelines).

For a fair comparison with published numbers, match: 3-fold CV (`-ns 3`), the per-dataset score (R² for regression on the skewness-transformed target, AUROC for classification), and the same train/test indices stored in `info.json`.

---

## Adding a new pipeline component

These are the seams.

- **New tabular learner.** Add an entry to `estim_configs` in [`configs/exp_configs.py`](configs/exp_configs.py) with `search_method` (`'no-search'` or `'random-search'`) and `fit_with_val` (bool). Wire the call in `src/inference.py` and `src/param_search.py`. Register the token in [`METHOD_NAMES.md`](scripts/evaluate_scripts/METHOD_NAMES.md).
- **New string encoder (non-LLM).** Add a branch in `src/encoding.py::embed_table` keyed on the encoder token; register in `METHOD_NAMES.md`.
- **New LLM encoder.** Add an `llm_configs["llm-<name>"]` entry in [`configs/exp_configs.py`](configs/exp_configs.py) (mirrors existing entries — they specify the HF model id and pooling) and you can immediately call `script_extract_llm_embeddings.py -m llm-<name>`.
- **New dataset.** Place a preprocess script under [`scripts/script_preprocess_data/`](scripts/script_preprocess_data/) that writes to `data/data_processed/<dataset>/`. Append the name to `data_list_wide` in `configs/exp_configs.py`. Re-running an experiment with `-dn all-wide` will then include it.

---

## Smoke test

Before running any non-trivial experiment, validate with `--dry-run` (limits to first dataset, fold 0):

```bash
python scripts/evaluate_scripts/script_evaluate.py --dry-run \
    --save_dir smoke -m num-str_tabvec_ridge \
    -dn yelp_business -ns 3 -fi 0 \
    -dv cpu -cf False -oc False -ti default
```

Expected: one CSV at `results/default/smoke/yelp_business/num-str_tabvec_ridge_default/score/…`. If that works, drop `--dry-run` and scale up.

---

## Common pitfalls

- **`FileNotFoundError` on `data/llm_embeding/...`** → Stage A (embedding extraction) wasn't run for that encoder; see [The two-stage rule](#the-two-stage-rule-llm-pipelines).
- **`KeyError` on a method token** → check `METHOD_NAMES.md` for valid `{dtype}_{encoder}_{learner}` combinations; remember not to append `_default` / `_tune`.
- **Results not appearing** → check `-cf True` isn't silently skipping because the CSV already exists. Either delete the target file or pass `-cf False`.
- **Wrong ablation folder in output** → you ran the wrong script. The ablation tag is hard-coded in each script's `ABLATION = "..."` line; pick the script that matches the user's intent (see [Ablation matrix](#ablation-matrix)).
- **`STRABLE_ROOT` not set** → paths fall back to the repo root inferred from `configs/path_configs.py`, which works for in-repo runs but breaks if the user launches from another cwd. Always `export STRABLE_ROOT=$(pwd)` from the repo root.
- **HuggingFace download fails for gated models** (Llama, Gemma) → user needs to accept the model license on the HF page and have `HF_TOKEN` set.
- **GPU OOM on 8B-class encoders** → reduce batch size in the encoder script, or fall back to `llm-all-MiniLM-L6-v2` for testing.

---

## Cross-reference

- Human walkthrough with badges, TL;DR, figures: [README.md](README.md)
- Full method-name catalogue: [scripts/evaluate_scripts/METHOD_NAMES.md](scripts/evaluate_scripts/METHOD_NAMES.md)
- Path resolution: [configs/path_configs.py](configs/path_configs.py)
- Experiment registries (estimators, LLM encoders, dataset list): [configs/exp_configs.py](configs/exp_configs.py)
- Paper: [arXiv:2605.12292](https://arxiv.org/abs/2605.12292)
- Dataset on Hugging Face: [inria-soda/STRABLE-benchmark](https://huggingface.co/datasets/inria-soda/STRABLE-benchmark)
