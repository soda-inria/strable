import pandas as pd

# 1. Define the dataset exactly as it appears in the table
data = [
    [r"\textbf{OpenML-CC18} \citep{bischl2021openmlbenchmarkingsuites}", "Mostly numerical", r"High ($N=72$); curated classification tasks."],
    [r"\textbf{PMLB / PMLBmini} \citep{romano2021pmlbv10opensource}", "Numerical / Low-cardinality", r"High ($N \approx 290$); inclusive of simplified datasets."],
    [r"\textbf{Grinsztajn et al. (2022)} \citep{grinsztajn2022tree}", "One-Hot Encoding", r"Moderate ($N=45$); removes high-cardinality features."],
    [r"\textbf{TabReD} \citep{rubachev2024tabredanalyzingpitfallsfilling}", "Removed / Numerical", r"Low ($N=8$); industry-grade but removes string signals."],
    [r"\textbf{TabArena} \citep{erickson2025tabarenalivingbenchmarkmachine}", "Curated / IID", r"Moderate ($N=51$); does not include complex string signals."],
    
    [r"\textbf{McElfresh et al. (2024)} \citep{mcelfresh2024neuralnetsoutperformboosted}", "Standard Vectorization", r"High ($N=176$); relies on OpenML pre-processed formats."],
    [r"\textbf{AMLB} \citep{gijsbers2023amlbautomlbenchmark}", "AutoML System Specific Encoding", r"High ($N \approx 104$); focuses on AutoML framework evaluation."],
    [r"\textbf{TabRepo} \citep{salinas2024tabrepolargescalerepository}", "N-gram and Method Specific Encoding", r"High ($N=211$); large-scale repository of model evaluations."],
    [r"\textbf{TALENT} \citep{liu2024talenttabularanalyticslearning}", "Standard Vectorization", r"High ($N=300$); broad scope but standard numerical focus."],
    [r"\textbf{Zabërgja et al. (2025)} \citep{zabërgja2025tabulardatadeeplearning}", "Standard Vectorization", r"Moderate ($N=68$); formally treats strings as mathematical vectors."],
    
    [r"\textbf{CARTE} \citep{kim2024cartepretrainingtransfertabular}", "Transformed / Curated", r"Moderate ($N=51$); strings manually transformed to numbers."],
    [r"\textbf{TextTabBench} \citep{mráz2025benchmarkingfoundationmodelstabular}", "Raw free-text", r"Low ($N=13$); limited dataset diversity."]
]

# Create the dataframe with pre-bolded column names
columns = [r"\textbf{Benchmark}", r"\textbf{Attention to string features}", r"\textbf{Size and Scope}"]
df = pd.DataFrame(data, columns=columns)

# 2. Use the Pandas Styler to generate the base LaTeX
styler = df.style.hide(axis="index")

latex_output = styler.to_latex(
    environment="table*",
    position="t",
    caption="Comparison of STRABLE against existing tabular benchmarks.",
    label="tab:benchmark_comparison",
    column_format="@{}llp{8cm}@{}",
    hrules=True
)

# 3. Inject custom LaTeX commands

# Add \centering
latex_output = latex_output.replace(
    r"\begin{table*}[t]",
    r"\begin{table*}[t]" + "\n" + r"\centering"
)

# Wrap the tabular environment in \resizebox
latex_output = latex_output.replace(
    r"\begin{tabular}",
    r"\resizebox{\textwidth}{!}{%" + "\n" + r"\begin{tabular}"
)
latex_output = latex_output.replace(
    r"\end{tabular}",
    r"\end{tabular}%" + "\n" + r"}"
)

# Fix Header Midrule: 
# Pandas puts \midrule on a new line. We replace that newline to pull it up, and add the first commented header.
header_orig = r"\textbf{Size and Scope} \\" + "\n" + r"\midrule"
header_new = r"\textbf{Size and Scope} \\ \midrule" + "\n" + r"% \multicolumn{3}{@{}l}{\textit{\textbf{Benchmarks with focused pre-processing and structural simplification}}} \\"
latex_output = latex_output.replace(header_orig, header_new)

# Inject midrules and commented headers for the middle sections
row_1_break = r"Moderate ($N=51$); does not include complex string signals. \\"
latex_output = latex_output.replace(
    row_1_break, 
    row_1_break + r" \midrule" + "\n" + r"% \multicolumn{3}{@{}l}{\textit{\textbf{Benchmarks focused on vectorized representations}}} \\"
)

row_2_break = r"Moderate ($N=68$); formally treats strings as mathematical vectors. \\"
latex_output = latex_output.replace(
    row_2_break, 
    row_2_break + r" \midrule" + "\n" + r"% \multicolumn{3}{@{}l}{\textit{\textbf{Specialized semantic benchmarks}}} \\"
)

# Fix Bottom Section:
# Pandas automatically adds \bottomrule here. We overwrite it with your commented-out rows.
row_3_orig = r"Low ($N=13$); limited dataset diversity. \\" + "\n" + r"\bottomrule"
row_3_new = r"Low ($N=13$); limited dataset diversity. \\ \midrule" + "\n" + r"% \textbf{STRABLE (Ours)} & \textbf{Raw free-text} & \textbf{High ($N=112$); raw, uncurated heterogeneous data.} \\" + "\n" + r"% \bottomrule"
latex_output = latex_output.replace(row_3_orig, row_3_new)

# 4. Save or print the output
save_path = "./benchmarks_comparison.tex"
with open(save_path, "w") as f:
    f.write(latex_output)

print(f"Table successfully generated and saved to {save_path}")