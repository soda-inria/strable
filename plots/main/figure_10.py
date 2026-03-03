'''
LEAVE-ONE-CATEGORY-OUT
'''

# 1. Prepare the Pivot Table
# Ensure we have one score per model per dataset

results_copy = results[(results['dtype']=='Num+Str') & (results['method'] != 'num-str_tabpfn_tabpfn_default')].copy()

pivot_source = results_copy.pivot_table(
    index='data_name', 
    columns='method_polished', 
    values='score', 
    aggfunc='mean'
)

# select only algos that are Num+Str
pivot_source = pivot_source[[col for col in pivot_source.columns if 'Num+Str' in col]].copy()

# Drop any methods/datasets that have NaN values
pivot_source = pivot_source.dropna(axis=1) #drop columns with NaN

# pivot_source = pivot_source.dropna(axis=0) #drop columns with NaN

valid_datasets = pivot_source.index
dataset_to_source = results[results['data_name'].isin(valid_datasets)].groupby('data_name')['category_with_ds_count'].first()

# 2. LOSO Calculation Loop
sources = results['category_with_ds_count'].unique()
loso_correlations = []

for src in sources:
    # Split datasets into current source and all other sources
    datasets_in_source = dataset_to_source[dataset_to_source == src].index
    
    # Filter pivot table
    df_src = pivot_source.loc[datasets_in_source]
    
    # Vector 1: Rank of each model on Source_i (averaged across datasets in that source)
    # Higher score is better -> ascending=False
    ranks_src = df_src.mean().rank(ascending=False)

    print(f"{len(ranks_src)} models ranked on category {src}.")
    
    # Vector 2: Average rank of each model on all other categories
    # We rank per dataset first, then average those ranks
    ranks_others_per_ds = pivot_source.rank(axis=1, ascending=False)
    avg_ranks_others = ranks_others_per_ds.mean()

    print(f"{len(avg_ranks_others)} models ranked on other categories excluding {src}.")

    # check that both vectors have the same models
    print(f"Category: {src}")
    assert set(ranks_src.index) == set(avg_ranks_others.index), "Model mismatch between category and others"
    # Calculate Kendall Tau between the two ranking vectors
    corr, _ = kendalltau(ranks_src, avg_ranks_others)
    
    loso_correlations.append({
        'Category': src,
        'Correlation': corr,
        'N_datasets': len(datasets_in_source)
    })

df_loso = pd.DataFrame(loso_correlations).sort_values('Correlation', ascending=False)
#drop rows with NaN correlation
df_loso = df_loso.dropna(subset=['Correlation'])

# Visualization - Transposed
plt.figure(figsize=(6, 5)) # Increased height to accommodate Y-axis labels
colors = plt.cm.viridis(np.linspace(0, 1, len(df_loso)))

# Use barh for horizontal bars
bars = plt.barh(df_loso['Category'], df_loso['Correlation'], color=colors, alpha=0.8)

# Add value labels for each bar
for bar in bars:
    width = bar.get_width()
    plt.text(
        width + 0.01,                # Position slightly to the right of the bar
        bar.get_y() + bar.get_height() / 2, # Center vertically in the bar
        f'{width:.3f}',             # The Kendall Tau value
        va='center', 
        fontsize=16, 
        fontweight='bold'
    )

# Formatting
# plt.title('Leave-One-Category-Out: $\\tau$(Category_i, Benchmark)', fontsize=16, pad=20)
plt.xlabel('Kendall $\\tau$ Rank Correlation', fontsize=16)
plt.ylabel('Excluded Category ($Category_i$)', fontsize=16)
plt.yticks(fontsize=16)
plt.xlim(0, 1.0) # Extended limit to make room for text labels
plt.grid(axis='x', alpha=0.3)
plt.legend(loc='lower right')

# plt.tight_layout()
today_date = time.strftime("%Y-%m-%d")
format = 'pdf'
PIC_NAME = f'leave_one_category_out_v4_check_transposed_{today_date}.{format}'
plt.savefig('/data/parietal/store4/soda/gblayer/salts/results_pics/'+ f"{TODAYS_FOLDER}/" + PIC_NAME, bbox_inches='tight')
plt.show()