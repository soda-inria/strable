"""Figure E.1 — KDE of per-column uniqueness across STRABLE / VSE / CARTE / TTB."""

import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))
del _sys, _Path, _REPO_ROOT


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from configs.path_configs import path_configs
from figures._main import save_figure


BENCHMARK_COLORS = [
    ('VSE',     '#1f77b4'),
    ('STRABLE', '#ff7f0e'),
    ('CARTE',   '#2ca02c'),
    ('TTB',     '#d62728'),
]


def plot_uniqueness_kde():
    df_comp = pd.read_csv(
        f"{path_configs['base_path']}/df_complexity_VSE_STRABLE_CARTE_TTB.csv"
    )

    fig, ax = plt.subplots(figsize=(5, 3))
    for name, color in BENCHMARK_COLORS:
        sub = df_comp[df_comp['benchmark'] == name]
        if sub.empty:
            continue
        med = sub['uniqueness'].median()
        sns.kdeplot(
            data=sub, x='uniqueness',
            label=f'{name} (med={med:.3f})',
            fill=True, color=color, linewidth=2, ax=ax,
        )
    ax.set_xlabel(
        "Proportion of Unique Values\n"
        "0.0 = highly repetitive category / constants\n"
        "1.0 = every row unique",
        fontsize=12,
    )
    ax.set_ylabel("Density of Columns", fontsize=12)
    ax.set_xlim(0, 1)
    ax.legend()

    save_figure(fig, "VSE_STRABLE_CARTE_TTB_uniqueness")
    plt.close(fig)


if __name__ == "__main__":
    plot_uniqueness_kde()
