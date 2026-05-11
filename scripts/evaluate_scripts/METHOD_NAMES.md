# Method-name catalogue

Every `--method` token passed to an `evaluate_scripts/script_evaluate*.py`
follows the convention `{dtype}_{encoder}_{learner}`.

The display-name maps for each token live in
[`figures/posthoc_analysis.py`](../figures/posthoc_analysis.py)
(`dtype_map`, `encoder_map`, `learner_map`).

The `_default` / `_tune` suffix on the learner field is **not** part of the
input — it is added at output time based on the `--tune_indicator` flag.

---

## Slot vocabulary

### `dtype` (3 values)

| Token | Display | Behaviour |
|---|---|---|
| `num-str` | Num+Str | Pipeline sees both numerical and string columns. |
| `num-only` | Num | String columns dropped before encoding. |
| `str-only` | Str | Numerical columns dropped before encoding. |

### `encoder`

| Token | Display | Use case |
|---|---|---|
| `tabvec` | Tf-Idf | Character-n-gram Tf-Idf (with SVD), the lightweight default. |
| `tarenc` | TargetEncoder | High-cardinality target encoder. |
| `tarte` | Tarte | Tarte's frozen pretrained string embeddings. |
| `llm-<name>` | LM ‹name› | One of the HuggingFace encoders enumerated in [`configs/exp_configs.py`](../configs/exp_configs.py) (e.g. `llm-qwen3-8b`, `llm-llama-3.1-8b`, `llm-all-MiniLM-L6-v2`, `llm-fasttext`, …). |

For E2E architectures the encoder field equals the learner field:
`catboost`, `contexttab`, `tabstar`, `mambular`.

### `learner`

| Token | Display | Modular / E2E |
|---|---|---|
| `ridge` | Ridge | modular |
| `xgb` | XGBoost | modular |
| `extrees` | ExtraTrees | modular |
| `tabpfn` | TabPFN-2.5 | modular |
| `tabicl` | TabICLv2 | modular |
| `tabm` | TabM | modular |
| `realmlp` | RealMLP | modular |
| `catboost` | CatBoost | **E2E** (encoder = learner) |
| `contexttab` | ContextTab | **E2E** |
| `tabstar` | TabSTAR | **E2E** |
| `mambular` | Mambular | **E2E** |

---

## Valid `{dtype}_{encoder}_{learner}` combinations

### `num-str` (Num + Str — main results)

#### Modular pipelines

For each modular learner below, any of these encoders is valid:
`tabvec`, `tarenc`, `tarte`, `llm-<name>`.

| Learner | Example tokens |
|---|---|
| `ridge` | `num-str_tabvec_ridge`, `num-str_llm-qwen3-8b_ridge`, … |
| `xgb` | `num-str_tabvec_xgb`, `num-str_llm-llama-3.1-8b_xgb`, … |
| `extrees` | `num-str_tabvec_extrees`, `num-str_llm-bge-base_extrees`, … |
| `tabpfn` | `num-str_tabvec_tabpfn`, `num-str_llm-qwen3-8b_tabpfn`, … |
| `tabicl` | `num-str_tabvec_tabicl`, … |
| `tabm` | `num-str_tabvec_tabm`, … |
| `realmlp` | `num-str_tabvec_realmlp`, … |

#### E2E pipelines (encoder = learner)

| Architecture | Token |
|---|---|
| CatBoost | `num-str_catboost_catboost` |
| ContextTab | `num-str_contexttab_contexttab` |
| TabSTAR | `num-str_tabstar_tabstar` |
| Mambular | `num-str_mambular_mambular` |

### `num-only` (numeric-only baseline)

No string encoder is needed. The default encoder here is set as `tabvec`.
One row per learner.

| Learner | Token |
|---|---|
| `ridge` | `num-only_tabvec_ridge` |
| `xgb` | `num-only_tabvec_xgb` |
| `extrees` | `num-only_tabvec_extrees` |
| `tabpfn` | `num-only_tabvec_tabpfn` |
| `tabicl` | `num-only_tabvec_tabicl` |
| `tabm` | `num-only_tabvec_tabm` |
| `realmlp` | `num-only_tabvec_realmlp` |
| `catboost` | `num-only_catboost_catboost` (E2E — handles numerics natively) |
| `contexttab` | `num-only_contexttab_contexttab` |
| `tabstar` | `num-only_tabstar_tabstar` |
| `mambular` | `num-only_mambular_mambular` |

`num-only` does not pair with `tarte` (Tarte's purpose is string semantics).

### `str-only` (strings-only baseline)

Same encoder × learner space as `num-str`. Numerical columns are dropped
before encoding. Examples:

| Learner | Example tokens |
|---|---|
| `ridge` | `str-only_tabvec_ridge`, `str-only_llm-qwen3-8b_ridge`, … |
| `xgb` | `str-only_llm-llama-3.1-8b_xgb`, … |
| `tabpfn` | `str-only_llm-qwen3-8b_tabpfn`, … |
| `catboost` | `str-only_catboost_catboost` |
| `contexttab` | `str-only_contexttab_contexttab` |
| `tabstar` | `str-only_tabstar_tabstar` |
| `mambular` | `str-only_mambular_mambular` |

---

## Tune flag

For tunable learners (`xgb`, `catboost`, `extrees` per
[`configs/exp_configs.py`](../configs/exp_configs.py)), pass
`--tune_indicator default | tune | all` — the value is appended to the
saved `method_marker` automatically (e.g. `num-str_tabvec_xgb_tune`). The
input `--method` token never carries the suffix.

For learners with no search space (`ridge`, `tabpfn`, `tabicl`, `tabm`,
`realmlp`, `tabstar`, `contexttab`, `mambular`), `--tune_indicator tune` is silently skipped (the
script prints "The model requires no tuning; skipping" and exits).