"""Preprocess HIFLD US Schools dataset.
The task is to predict Enrollment/Full Time Teachers ratio
"""


# >>>
if __name__ == "__main__":
    import os
    import sys

    _project_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    os.environ["PROJECT_DIR"] = _project_dir
    sys.path.append(_project_dir)
    del _project_dir
# <<<

import pandas as pd
import numpy as np
import os
import json

from scipy import stats
from skrub import TableVectorizer, column_associations
from src.utils_preprocess import clean_backslash_operations, clean_constant_null
from configs.path_configs import path_configs

## Load data (This may be different for specific cases)
# Change the filename accordingly
# private = f"{path_configs['base_path']}/data/data_raw/HIFLD/Private_Schools_-7285710811296673603.csv"
# public = f"{path_configs['base_path']}/data/data_raw/HIFLD/Public_Schools_-7669544197405643438.csv"

private_path = f"{path_configs['path_data_raw']}/HIFLD/us-private-schools.csv"
public_path = f"{path_configs['path_data_raw']}/HIFLD/us-public-schools.csv"

def load_hifld_csv(path):
    # Use sep=';' because your screenshots show semicolon-delimited data
    # Use engine='python' to safely handle the JSON-like 'Geo Point' strings
    df = pd.read_csv(path, sep=';', engine='python', on_bad_lines='warn')
    
    # Clean column names: remove whitespace and convert to uppercase for consistency
    df.columns = [col.strip().upper() for col in df.columns]
    
    # If the first row is metadata (like in your Section 117 screenshot), 
    # you might need to drop it if it contains NaNs or repeats headers
    if df.iloc[0].isnull().all() or "GEO POINT" in str(df.iloc[0]):
         df = df.iloc[1:].reset_index(drop=True)
         
    return df

# Load both
df_private = load_hifld_csv(private_path)
df_public = load_hifld_csv(public_path)

# Standardize columns before concat
# Since they might have different columns (e.g., DISTRICTID vs NCESID)
# we use 'outer' join to keep everything, or 'inner' to keep only shared ones.
data = pd.concat([df_private, df_public], axis=0, join='outer', ignore_index=True)


# 1. First, merge the columns that have slight name variations from the concat
# This ensures we don't have NaNs in one while the data is in the other
data['FT_TOTAL'] = data['FT_TEACHERS'].fillna(data['FT_TEACHER'])
data['SOURCE_DATE_CLEAN'] = data['SOURCE_DATE'].fillna(data['SOURCEDATE'])
data['x'] = data['LONGITUDE']
data['y'] = data['LATITUDE']
# 2. Define the mapping from RAW to TARGET
rename_map = {
    'NCESID': 'NCES ID',
    'NAME': 'Name',
    'ADDRESS': 'Address',
    'CITY': 'City',
    'STATE': 'State',
    'ZIP': 'Zip',
    'ZIP4': 'Zip4',
    'TELEPHONE': 'Telephone',
    'TYPE': 'Type',
    'STATUS': 'Status',
    'COUNTY': 'County',
    'COUNTYFIPS': 'County FIPS',
    'LATITUDE': 'Latitude',
    'LONGITUDE': 'Longitude',
    'SOURCE': 'Source',
    'VAL_METHOD': 'Validation Method',
    'VAL_DATE': 'Validation Date',
    'WEBSITE': 'Website',
    'LEVEL_': 'Level',
    'ST_GRADE': 'Start Grade',
    'END_GRADE': 'End Grade',
    'SHELTER_ID': 'Shelter ID',
    'DISTRICTID': 'District ID'
}

# Apply renames
data.rename(columns=rename_map, inplace=True)

# 3. Calculate the missing Ratio column
# We use the merged FT_TOTAL we created in step 1
# Replace 0 with NaN to avoid division by zero errors
data = data[data['FT_TOTAL']>0].reset_index(drop=True)
data['Enrollment_FT_Teachers_Ratio'] = data['ENROLLMENT'] / data['FT_TOTAL'].replace(0, np.nan)

# 4. Final selection: Keep only the columns present in your sampled schools_df
# and in the exact order you specified
target_columns = [
    'OBJECTID', 'NCES ID', 'Name', 'Address', 'City', 'State', 'Zip', 
    'Zip4', 'Telephone', 'Type', 'Status', 'County', 'County FIPS', 
    'Latitude', 'Longitude', 'Source', 'Validation Method', 
    'Validation Date', 'Website', 'Level', 'Start Grade', 'End Grade', 
    'Shelter ID', 'x', 'y', 'District ID', 'Enrollment_FT_Teachers_Ratio'
]

# # Use existing columns to avoid 'KeyError' if some were missing from the raw files
existing_targets = [col for col in target_columns if col in data.columns]
data = data[existing_targets]

print(f"Data successfully reshaped. Final shape: {data.shape}")


# check with existing processed data
schools_df = pd.read_parquet(f"{path_configs['path_data_processed']}/schools/data.parquet")
common_cols = set(data.columns).intersection(set(schools_df.columns))
print(f"Common columns: {common_cols}")
print(f"Number of common columns: {len(common_cols)/len(schools_df.columns)*100:.2f}%")

## Clean for backslash operations
data = clean_backslash_operations(data)

## Dataset-level specific cleaning
data = data.replace(-999, np.nan)
data['NCES ID']= data['NCES ID'].astype(str)
data['Level']= data['Level'].astype(str)
data['Start Grade']= data['Start Grade'].astype(str)
data['End Grade']= data['End Grade'].astype(str)


## Clean for specific data formats (dict / list)

## Set metadata
target_name = 'Enrollment_FT_Teachers_Ratio'
task = 'regression'
task_type = 'wide'
source = 'HIFLD'

## Clean for the target column
data.dropna(subset=[target_name], inplace=True)
data.reset_index(drop=True, inplace=True)
if task == 'regression':
    data[target_name] = data[target_name].astype('float32')

## Check skewness and kurtosis if target is numeric
if task == 'regression':
    check_y = data[target_name].copy()
    check_y = np.array(check_y)
    skewness = stats.skew(check_y)
    kurtosis = stats.kurtosis(check_y)
    print(f'Before - skewness: {skewness} | kurtosis: {kurtosis}') # target is highly skewed and kurtotic
    
    # apply transformation if found skewed: np.log, np.log1p, np.cbrt, np.arcsinh, np.sign(check_y) * np.log1p(np.abs(check_y))
    if abs(skewness) > 1:
        check_y = data[target_name].copy()
        check_y = np.sign(check_y) * np.log1p(np.abs(check_y)) #np.arcsinh(check_y)
        skewness = stats.skew(check_y)
        kurtosis = stats.kurtosis(check_y)
        print(f'After - skewness: {skewness} | kurtosis: {kurtosis}')

## Apply appropriate transformation 
## np.log, np.log1p, np.cbrt, np.arcsinh, np.sign(check_y) * np.log1p(np.abs(check_y))
data[target_name] = np.log1p(data[target_name]) # applied

## Clean for columns with constants or only with null values
data = clean_constant_null(data, proportion_null=1.0)

## Drop duplicate columns
data.drop_duplicates(inplace=True)
data.reset_index(drop=True, inplace=True)

## Check the data criterion
if data.shape[0] < 500:
    raise ValueError("Dataset must have at least 500 rows.")
# if data.shape[0] > 75000:
#     print(f"Dataset has {data.shape[0]} rows, subsampling to 75000 rows.")
#     if task == 'regression':
#         data = data.sample(n=75000, random_state=42).reset_index(drop=True)
#     else:
#         data, _ = train_test_split(data, train_size=75000, random_state=42, stratify=data[target_name])
#         data.reset_index(drop=True, inplace=True)

## Check the number of string columns
tabvec = TableVectorizer(high_cardinality='passthrough', cardinality_threshold=0)
tabvec.fit_transform(data.drop(columns=target_name))
str_cols = tabvec.kind_to_columns_['high_cardinality']
if len(str_cols) < 2:
        raise ValueError("Dataset must have at least 2 string-type columns.")

## Column-level cleaning
# Possible leakage columns and high correlations.
# drop_col = []
# # Leakage
# drop_col.append('Population')
# # Highly related
# data.drop(columns=drop_col, inplace=True)

## Check with Cramer's-V
cram_df = column_associations(data)
cram_df1 = cram_df[cram_df.left_column_name==target_name].copy()
cram_df2 = cram_df[cram_df.right_column_name==target_name].copy()
cram_df = pd.concat([cram_df1, cram_df2])
cram_df = cram_df.sort_values(by='cramer_v', ascending=False).reset_index(drop=True)

data.dropna(subset=[target_name], inplace=True)
data.reset_index(drop=True, inplace=True)

if set(data.columns) != set(schools_df.columns):
    print("Columns in new dataset but not in existing dataset:")
    print(set(data.columns) - set(schools_df.columns))
    print("Columns in existing dataset but not in new dataset:")
    print(set(schools_df.columns) - set(data.columns))

## Save data and config
## Change appropriately the save_folder
data_filename = 'schools-FULL'
save_folder = f'{path_configs['path_data_processed']}_FULL/{data_filename}'
save_path_data = f'{save_folder}/data.parquet'
save_path_config = f'{save_folder}/config.json'
if not os.path.exists(save_folder):
    os.makedirs(save_folder, exist_ok=True)

data.to_parquet(save_path_data, index=False)

config = dict()
config["target_name"] = target_name
config["task"] = task
config["task_type"] = task_type
config["source"] = source
with open(save_path_config, "w") as outfile:
    json.dump(config, outfile)

#%%

## Check for other leakages

import numpy as np
from src.encoding import embed_table
from src.utils_evaluation import calculate_output, reshape_pred_output, check_pred_output, return_score, set_score_criterion
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.linear_model import RidgeCV, RidgeClassifierCV
from sklearn.impute import SimpleImputer

data_name = data_filename
n_split = 3
fold_index = 1
embed_method = 'num-str_tarenc'

scoring, result_criterion = set_score_criterion(task)

X_train, X_test, y_train, y_test, duration_embed, cat_features = embed_table(
    data_name,
    n_split,
    fold_index,
    embed_method,
)

if task == 'regression':
    estimator_extrees = ExtraTreesRegressor(n_jobs=24, random_state=1234)
    estimator_ridge = RidgeCV()
else:
    estimator_extrees = ExtraTreesClassifier(n_jobs=24, random_state=1234)
    estimator_ridge = RidgeClassifierCV()

# Extrees
estimator_extrees.fit(X_train, y_train)
y_prob, y_pred = calculate_output(X_test, estimator_extrees, task)

# Reshape prediction
if "classification" in task:
    y_prob = reshape_pred_output(y_prob)

# Check the output
if task == "regression":
    y_pred = check_pred_output(y_train, y_pred)

# obtain scores
score = return_score(y_test, y_prob, y_pred, task)

print(f'ExtraTrees - The {result_criterion[0]} for {data_name} is {np.round(score[0], 3)}')

# Ridge
imputer = SimpleImputer()
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)
estimator_ridge.fit(X_train, y_train)
y_prob, y_pred = calculate_output(X_test, estimator_ridge, task)

# Reshape prediction
if "classification" in task:
    y_prob = reshape_pred_output(y_prob)

# Check the output
if task == "regression":
    y_pred = check_pred_output(y_train, y_pred)

# obtain scores
score = return_score(y_test, y_prob, y_pred, task)

print(f'Ridge - The {result_criterion[0]} for {data_name} is {np.round(score[0], 3)}')

# %%
