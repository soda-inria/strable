## STRABLE

**STRABLE** is a tabular learning benchmark built from the **SALTS** codebase and structured following the **CARTE** repository conventions.

This repository collects the preprocessing, configuration, and evaluation code used for large-scale STRABLE experiments.

### 01. Install

Create a virtual environment and install the STRABLE dependencies:

```bash
pip install -r requirements.txt
```

or install it as a package:

```bash
pip install .
```

### 02. Layout

- **`src/`**: core benchmarking logic (encoding, inference, evaluation utilities).
- **`configs/`**: experiment, model, and path configuration files.
- **`script_preprocess/`** and **`script_preprocess_jk/`**: dataset-specific preprocessing scripts.
- **`scripts/`**: additional data cleaning and consistency checks.
- **`data/`**: processed data and embeddings required by the benchmark (not version-controlled in full in all setups).

Auxiliary analysis and aggregation scripts (e.g. `compile_results*.py`, `posthoc_analysis.py`, `datasets_representation.py`) live at the repository root when added.

### 03. Usage

Run the main benchmark (using the default STRABLE configuration):

```bash
make run-benchmark
```

which is equivalent to:

```bash
python script_evaluate.py
```

You can modify experiment settings in `configs/exp_configs.py` and dataset paths in `configs/path_configs.py` as needed for your environment.
