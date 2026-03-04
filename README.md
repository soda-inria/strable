# STRABLE

**STRABLE** (**Str**ing T**able**) is a comprehensive benchmarking corpus of 108 tables with strings and numbers.

## Repository layout

```
strable/
├── configs/                 
│   ├── exp_configs.py
│   ├── model_parameters.py
│   └── path_configs.py
├── data/
│   ├── download_datasets.py # Download benchmark data from Hugging Face
│   └── data_processed/      # (created after download)
├── scripts/
│   ├── analysis_setup.py    # Shared setup for all analysis scripts
│   ├── compile_results.py   # Aggregate individual run scores into one CSV
│   ├── datasets_representation.py  # Collect dataset metadata
│   ├── download_fasttext.py # Download the fastText model
│   └── data_preprocessing_scripts/  # Dataset-specific preprocessing
├── src/                     
│   ├── encoding.py          # Table embedding / encoding strategies
│   ├── inference.py         # Model inference
│   ├── param_search.py      # Hyper-parameter search
│   ├── utils_evaluation.py  # Data loading, scoring, estimator assignment
│   ├── utils_preprocess.py  # Data-cleaning helpers
│   └── utils_visualization.py  # Critical-difference diagrams, etc.
├── plots/
│   ├── main/                # Figures for the main paper
│   └── appendix/            # Figures for the appendix
├── tables/
│   ├── main/                # Tables for the main paper
│   └── appendix/            # Tables for the appendix
├── script_evaluate.py       # Main benchmark evaluation entry point
├── script_extract_llm_embeddings.py  # LLM embedding extraction
├── requirements.txt
└── pyproject.toml
```

---

## Step 1 – Install

Install all dependencies:

```bash
pip install -r requirements.txt
```

> **Note:** To install ContextTab follow https://github.com/SAP-samples/sap-rpt-1-oss.

## Step 2 – Prepare data and models

All paths should be configures automatically through `configs/path_configs.py`.

### 2a. Download benchmark datasets

```bash
python data/download_datasets.py
```

This mirrors the [STRABLE-benchmark](https://huggingface.co/datasets/inria-soda/STRABLE-benchmark) Hugging Face dataset repository into `data/data_processed/`.

### 2b. Download the fastText model

```bash
python scripts/download_fasttext.py
```

Downloads the English fastText model (`cc.en.300.bin`) to the path specified in `path_configs`.

## Step 3 – Extract LLM embeddings

```bash
python script_extract_llm_embeddings.py
```

Extracts embeddings of a Language Model for a given dataset. Results are saved under `data/llm_embeding/` and timing information under `data/llm_embed_time/`.

## Step 4 – Run the benchmark

```bash
python script_evaluate.py
```

This runs the full evaluation pipeline for all dataset (Num+Str, Str only, Num only) × encoder × learner combinations.
Individual scores are stored in `results/benchmark/`.

## Step 5 – Compile results

```bash
python scripts/compile_results.py
```

Aggregates every per-run CSV under `results/benchmark/` into a single results file used by all downstream analysis scripts.

## Step 6 – Collect dataset metadata

```bash
python scripts/datasets_representation.py
```

Produces the dataset summary table consumed by the figures and tables.

## Step 7 – Reproduce the figures

Each figure is a self-contained script.
Running it generates a PDF in `results_pics/<today>/`.

**Main paper:**

```bash
python plots/main/figure_1.py
python plots/main/figure_2.py
# … through figure_11
python plots/main/figure_11.py
```

**Appendix:**

```bash
python plots/appendix/figure_C1.py
python plots/appendix/figure_E1.py
# … and so on for all appendix figures
```

## Step 8 – Reproduce the tables

Each table is likewise a self-contained script.
Running it generates a LaTeX file in `results_tables/<today>/`.

**Main paper:**

```bash
python tables/main/table_1.py
```

**Appendix:**

```bash
python tables/appendix/table_B1.py
python tables/appendix/table_C1.py
# … through table_E2
```

---

## Citation

If you use STRABLE in your work, please cite:

```bibtex
@unpublished{strable2026,
  title={STRABLE: Benchmarking Tabular Machine Learning with Strings},
  author={Anonymous Authors},
  year={2026}
}
```

## License

This project is released under the [BSD 3-Clause License](LICENSE).
