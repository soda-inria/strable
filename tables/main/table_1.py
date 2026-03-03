# ==============================================
# TABLE 1.2: Task Distribution across categories
# ==============================================

import os
import time

import pandas as pd

from strable.configs.path_configs import path_configs
from strable.scripts.analysis_setup import TODAYS_FOLDER, results

df_datasets = results.drop_duplicates(subset=['data_name']).copy()

print(f"Analysis performed on {len(df_datasets)} unique datasets across {df_datasets['category'].nunique()} sources.")

table1 = pd.crosstab(df_datasets['category'], df_datasets['task'])

# Add Total column
table1['Total'] = table1.sum(axis=1)

# save as latex
table1_final = table1.copy()

# Add a 'Total' row at the bottom
table1_final.loc['Total'] = table1_final.sum()

# Reset index so 'category' becomes a column for the LaTeX table
table1_latex = table1_final.reset_index()
table1_latex = table1_latex.rename(columns={'category': 'Category'})

df_to_export = table1_latex.copy()

# Shorten names and bold them for the ICML layout and desired LaTeX output
df_to_export = df_to_export.rename(columns={
    'Category': '\\textbf{Category}',
    'b-classification': '\\textbf{b-class}',
    'm-classification': '\\textbf{m-class}',
    'regression': '\\textbf{reg}',
    'Total': '\\textbf{Total}'
})

# 2. Use the Styler with LaTeX-specific bolding
styler = df_to_export.style.hide(axis="index")

# Bold the "Total" column (Note: the column name now includes the \textbf wrapper)
styler.format(subset=['\\textbf{Total}'], formatter="\\textbf{{{}}}")

# Bold the "Total" row (the last row)
styler.format(subset=pd.IndexSlice[df_to_export.index[-1], :], formatter="\\textbf{{{}}}")

# 3. Export to LaTeX using the modern Styler method
today_date = time.strftime("%Y-%m-%d")
filename = f"category_task_contingency_table_{today_date}.tex"
save_path = os.path.join(
    path_configs["base_path"],
    "results_tables",
    TODAYS_FOLDER,
    filename,
)

# Export with exact position, caption, and formatting from the paper
latex_output = styler.to_latex(
    hrules=True,
    caption="Distribution of datasets across categories and task types.",
    label="tab:dataset_distribution_table1.5",
    column_format="l" + "c" * (len(df_to_export.columns) - 1),
    position="b!",
    position_float="centering"
)

# 4. Inject Compactness Commands
compact_latex = latex_output.replace(
    r"\begin{tabular}", 
    r"\setlength{\tabcolsep}{3pt} \small" + "\n" + r"\begin{tabular}"
)

# 5. Save the file
with open(save_path, 'w') as f:
    f.write(compact_latex)