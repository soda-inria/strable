import os
import re
import time

import pandas as pd

from strable.configs.path_configs import path_configs
from strable.scripts.analysis_setup import TODAYS_FOLDER

def generate_hyperparameter_table():
    # 1. Define the exact text and LaTeX commands for each cell
    data = [
        ["TabPFN-2.5", "-", "Default parameters"],
        ["TabStar", "-", "Default parameters"],
        ["ContextTab", "-", "Default parameters"],
        ["Ridge Regression", r"Alpha $(\alpha)$", r"$[0.01, 0.1, 1, 10, 100]$"],
        
        # XGBoost section
        [r"\multirow{9}{*}{XGBoost}", "Max depth", r"UniformInt [$2, 6$]"],
        ["", "Min child weight", r"LogUniform [$1, 100$]"],
        ["", "Subsample", r"Uniform [$0.5, 1$]"],
        ["", "Learning rate", r"LogUniform [$10^{-5}, 1$]"],
        ["", "Colsample by level", r"Uniform [$0.5, 1$]"],
        ["", "Colsample by tree", r"Uniform [$0.5, 1$]"],
        ["", "Gamma", r"LogUniform [$10^{-8}, 7$]"],
        ["", r"L2 regularization ($\lambda$)", r"LogUniform [$1, 4$]"],
        ["", r"L1 regularization ($\alpha$)", r"LogUniform [$10^{-8}, 100$]"],
        
        # CatBoost section
        [r"\multirow{7}{*}{CatBoost}", "Max depth", r"UniformInt [$2, 6$]"],
        ["", "Learning rate", r"LogUniform [$10^{-5}, 1$]"],
        ["", "Bagging temperature", r"Uniform [$0, 1$]"],
        ["", r"$l_2$-leaf regularization", r"LogUniform [$1, 10$]"],
        ["", "Random strength", r"UniformInt [$1, 20$]"],
        ["", "One hot max size", r"UniformInt [$0, 25$]"],
        ["", "Leaf estimation iterations", r"UniformInt [$1, 20$]"],
        
        # ExtraTrees section
        [r"\multirow{3}{*}{ExtraTrees}", "Max features", r"\{sqrt, $0.5, 0.75, 1.0$\}"],
        ["", "Min samples split", r"LogUniformInt [$2, 32$]"],
        ["", "Min impurity decrease", r"Choice \{$0, 10^{-5}, 3\cdot 10^{-5}, 10^{-4}, 3\cdot 10^{-4}, 10^{-3}$\}"]
    ]

    # Pre-format the column headers to perfectly match the \multicolumn wrappers
    columns = [
        r"\multicolumn{1}{c}{\textbf{Methods}}", 
        r"\multicolumn{1}{c}{\textbf{Parameters}}", 
        r"\multicolumn{1}{c}{\textbf{Grid}}"
    ]

    df = pd.DataFrame(data, columns=columns)

    # 2. Generate Base LaTeX with Pandas
    styler = df.style.hide(axis="index")
    
    latex_output = styler.to_latex(
        environment="table",
        position="!h",
        caption="Hyperparameter search space for STRABLE learners.",
        label="tab:strable_hyperparameter_space",
        column_format="lll",
        hrules=True
    )

    # 3. Inject structural commands
    # Inject \small and \centering
    latex_output = latex_output.replace(
        r"\begin{table}[!h]",
        r"\begin{table}[!h]" + "\n" + r"\small" + "\n" + r"\centering"
    )

    # Inject \vspace{0.1in} immediately after the label
    latex_output = latex_output.replace(
        r"\label{tab:strable_hyperparameter_space}",
        r"\label{tab:strable_hyperparameter_space}" + "\n" + r"\vspace{0.1in}"
    )

    # 4. Inject \midrule dynamically at specific row boundaries
    # We use regex to find the end of the target rows (\\) and append \midrule underneath
    midrule_targets = [
        r"(.*TabPFN-2.5.*\\\\)",
        r"(.*TabStar.*\\\\)",
        r"(.*ContextTab.*\\\\)",
        r"(.*Ridge Regression.*\\\\)",
        r"(.*L1 regularization.*\\\\)",          # End of XGBoost
        r"(.*Leaf estimation iterations.*\\\\)"   # End of CatBoost
    ]

    for target in midrule_targets:
        latex_output = re.sub(target, r"\1\n\\midrule", latex_output)

    # 5. Save the output
    today_date = time.strftime("%Y-%m-%d")
    filename = f"table_hyperparameters_{today_date}.tex"
    save_path = os.path.join(
        path_configs["base_path"],
        "results_tables",
        TODAYS_FOLDER,
        filename,
    )
    with open(save_path, "w") as f:
        f.write(latex_output)

    print(f"✅ Table successfully generated and saved to {save_path}")

if __name__ == "__main__":
    generate_hyperparameter_table()