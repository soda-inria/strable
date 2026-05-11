"""Recap of preprocessed datasets."""

import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))
del _sys, _Path, _REPO_ROOT


import json
from pathlib import Path
import pandas as pd
from configs.path_configs import path_configs

data_processed_path = Path(path_configs['path_data_processed'])
wide_datasets = []
panel_datasets = []
other_type = []
for dataset_dir in data_processed_path.iterdir():
    if dataset_dir.is_dir():
        metadata_path = dataset_dir / 'config.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                if metadata.get('task_type') == 'wide':
                    wide_datasets.append(dataset_dir.name)
                elif metadata.get('task_type') == 'panel':
                    panel_datasets.append(dataset_dir.name)
                else:
                    other_type.append(dataset_dir.name)


def _report_class_balance():
    """Heavy parquet I/O — only run when this script is __main__."""
    print(f'Number of wide datasets: {len(wide_datasets)}')
    print(f'Number of panel datasets: {len(panel_datasets)}')
    print(f'Number of other type datasets: {len(other_type)}')

    summary_path = path_configs["dataset_summary_wide"]
    df_summary = pd.read_parquet(summary_path)
    df_summary = df_summary[df_summary['task'] == 'b-classification']
    binary_task_stats = []
    balance_threshold = 0.40

    print(f"Analyzing {len(df_summary)} wide datasets for binary classification balance...")

    for _, row in df_summary.iterrows():
        dataset_name = row['data_name']
        target_col = row['target_column']
        dataset_path = data_processed_path / dataset_name / 'data.parquet'

        if not dataset_path.exists():
            print(f"Warning: Data file not found for {dataset_name}")
            continue

        try:
            try:
                df_target = pd.read_parquet(dataset_path, columns=[target_col], engine="pyarrow")
            except Exception:
                df_target = pd.read_parquet(dataset_path, columns=[target_col], engine="fastparquet")

            y = df_target[target_col].dropna()
            unique_classes = y.unique()
            if len(unique_classes) == 2:
                counts = y.value_counts()
                ratios = y.value_counts(normalize=True)
                minority_ratio = ratios.min()
                status_label = "Balanced" if minority_ratio >= balance_threshold else "Imbalanced"
                binary_task_stats.append({
                    'data_name': dataset_name,
                    'is_binary': True,
                    'class_0_ratio': ratios.iloc[0],
                    'class_1_ratio': ratios.iloc[1],
                    'minority_ratio': minority_ratio,
                    'status': status_label,
                    'counts': str(counts.to_dict()),
                })
        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")

    df_binary_task_stats = pd.DataFrame(binary_task_stats)
    if not df_binary_task_stats.empty:
        print(f"\nFound {len(df_binary_task_stats)} binary classification tasks.")
        print("-" * 50)
        print(df_binary_task_stats[['data_name', 'minority_ratio', 'status']].to_string())
        print("-" * 50)
        print(df_binary_task_stats['status'].value_counts())
    else:
        print("No binary classification tasks found.")


if __name__ == "__main__":
    _report_class_balance()