import os
import time

from strable.configs.path_configs import path_configs
from strable.scripts.analysis_setup import (
    TODAYS_FOLDER,
    median_iqr,
    results,
)

df_datasets = results.drop_duplicates(subset=['data_name']).copy()

# Select features to analyze from your columns
features_to_analyze = ['num_rows','num_columns', 'num_text_columns', 'avg_string_length_per_cell', 'avg_cardinality', 'avg_tfidf_cosine_similarity', 'prop_missing_text_cells', 'prop_unique_text_cells']

# drop num_rows_y and rename num_rows_x to num_rows
# df_datasets = df_datasets.rename(columns={'num_rows_x': 'num_rows'}).drop(columns=['num_rows_y'])

# Group by Source and apply the custom formatter
table2 = df_datasets.groupby('category', as_index=False)[features_to_analyze].agg(median_iqr)


print("\n--- Table 2: Aggregated features (Median + IQR) ---")
table2


feature_names_map = {
    'category':'Category',
    'num_rows': 'Number of Rows',
    'num_columns': 'Number of Columns',
    'num_text_columns': 'Number of String Columns',
    'avg_string_length_per_cell': 'Avg. String Length',
    'avg_cardinality': 'Cardinality',
    'avg_tfidf_cosine_similarity': 'Semantic Similarity',
    'prop_missing_text_cells': 'Prop. Missing Values in String Columns',
    'prop_unique_text_cells': 'Prop. Unique Values in String Columns'
}

table2.rename(columns=feature_names_map, inplace=True)


# rest of the columns
table2_part2 = table2.drop(columns=['Category','Semantic Similarity', 'Prop. Missing Values in String Columns', 'Prop. Unique Values in String Columns'])
table2_part2.index = table2['Category']


header_map = {
    'Number of Rows': 'Number of Rows', 
    'Number of Columns': 'Number of Columns', 
    'Number of String Columns': 'Number of String Columns',
    'Avg. String Length': '\\makecell{Avg. String\\\\Length}',
    'Cardinality': 'Cardinality',
}

# 'Semantic Similarity': '\\makecell{Semantic\\\\Similarity}',
# 'Prop. Missing Values in String Columns': '\\makecell{Prop. Missing Values\\\\in String Columns}',
# 'Prop. Unique Values in String Columns': '\\makecell{Prop. Unique Values\\\\in String Columns}'


# 2. Prepare the data
# Assuming table2_part2 is your second dataframe
df_export = table2_part2.copy()
df_export.columns = [header_map.get(col, col) for col in df_export.columns]
df_export = df_export.reset_index().rename(columns={'index': 'Category'})

# 3. Initialize Styler
styler = df_export.style.hide(axis="index")

# 4. Apply LaTeX-specific formatting
# Bold the 'Category' column
styler.format(subset=['Category'], formatter="\\textbf{{{}}}")

# Disable automatic escaping so our \makecell and \textbf commands work
styler.format(escape=None)

# 5. Define file path
today_date = time.strftime("%Y-%m-%d")
filename = f"dataset_features_part2_{today_date}.tex"
save_path = os.path.join(
    path_configs["base_path"],
    "results_tables",
    TODAYS_FOLDER,
    filename,
)

# 6. Export to LaTeX
# We use 'c' for data columns; makecell handles the width automatically
num_data_cols = len(df_export.columns) - 1
col_format = "l" + "c" * num_data_cols

latex_output = styler.to_latex(
    hrules=True,
    caption="Summary statistics of curated datasets by category: Median [IQR]",
    label="tab:dataset_features_part2",
    column_format=col_format,
    position="t",
    position_float="centering"
)

# 7. Inject compactness tweaks manually
# \footnotesize and \tabcolsep reduce the footprint to fit a single column
compact_latex = latex_output.replace(
    r"\begin{tabular}", 
    r"\setlength{\tabcolsep}{3pt} \footnotesize" + "\n" + r"\begin{tabular}"
)

# 8. Save
with open(save_path, 'w') as f:
    f.write(compact_latex)

print(f"Compact Table 2 with multiline headers saved to: {save_path}")