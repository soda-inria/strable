"""Preprocess awarded grants.
The task is to predict financial assistance (regression).
"""

#%%

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
from sklearn.model_selection import train_test_split
from skrub import TableVectorizer, column_associations
from src.utils_preprocess import clean_backslash_operations, clean_constant_null, clean_dict_type_col, clean_list_type_col
from configs.path_configs import path_configs

## Load data (This may be different for specific cases)
filename = 'awarded-grants'
data_path = f'{path_configs['path_data_processed']}/{filename}/data.parquet'
data = pd.read_parquet(data_path)

## Clean for backslash operations
# data = clean_backslash_operations(data)

# ## Dataset-level specific cleaning
# data = data.drop_duplicates(subset=['Grant Number', 'Financial Assistance']).reset_index(drop=True)

## String column overview
## | Column                    | Sample values                          | Transformation                 |
## |---------------------------|----------------------------------------|--------------------------------|
## | Grantee Address           | '32033 Beaver Run Dr', '2318 E Cent...| —                             |
## | Grantee City              | 'Salisbury', 'Wichita', 'Sioux City'  | —                             |
## | Grantee County Description| 'County      ', 'County      ', 'Co...| —                             |
## | Grantee County Name       | 'Wicomico', 'Sedgwick', 'Woodbury'    | —                             |
## | Grantee Name              | 'Three Lower Counties Commun...', '...| —                             |
## | HRSA Region               | 'Region 3', 'Region 7', 'Region 7'    | → region_num int              |
## | Grantee State Abbreviation| 'MD', 'KS', 'IA'                      | —                             |
## | State Name                | 'Maryland', 'Kansas', 'Iowa'          | —                             |
## | Grantee ZIP Code          | '21804-1773', '67214-4436', '51105-...| —                             |
## | Grant Activity Code       | 'H80', 'H8C', 'H8C'                   | —                             |
## | Grant Number              | 'H80CS00774', 'H8CCS34768', 'H8CCS3...| —                             |
## | Project Period Start Date | '2002/06/01', '2020/03/15', '2020/0...| —                             |
## | Grant Project Period End D| '2023/05/31', '2021/03/14', '2021/0...| —                             |
## | HRSA Program Area Code    | 'BPHC', 'BPHC', 'BPHC'                | —                             |
## | HRSA Program Area Name    | 'Primary Health Care', 'Primary Hea...| —                             |
## | Complete County Name      | 'Wicomico County', 'Sedgwick County...| —                             |
## | Grant Program Name        | 'Health Center Program (H80)', 'FY ...| —                             |
## | Congressional District Nam| 'Maryland District 01', 'Kansas Dis...| —                             |
## | Congressional District Num| '01', '04', '04'                      | —                             |
## | HHS Region Number         | '3', '7', '7'                         | —                             |
## | U.S. Congressional Represe| 'Andy Harris', 'Ron Estes', 'Randy ...| —                             |
## | State and County Federal I| '24045', '20173', '19193'             | —                             |
## | State FIPS Code           | '24', '20', '19'                      | —                             |
## | U.S. - Mexico Border 100 K| 'N', 'N', 'N'                         | → binary 1/0                  |
## | U.S. - Mexico Border Count| 'N', 'N', 'N'                         | → binary 1/0                  |
## | Name of U.S. Senator Numbe| 'Benjamin L. Cardin', 'Jerry Moran'...| —                             |
## | Name of U.S. Senator Numbe| 'Chris Van Hollen', 'Roger Marshall...| —                             |
## | Uniform Data System Grant | 'This notice announces the o...', '...| —                             |
## | Grantee Type Description  | 'Corporate Entity, Federal T...', '...| —                             |
## | DUNS Number               | '959475484', '181584889', '794550830' | —                             |
## | Unique Entity Identifier  | 'EB81PX23H8F5', 'LP9KLMAQQNB7', 'S4...| —                             |
## | Data Warehouse Record Crea| '2021/03/09', '2021/03/09', '2021/0...| —                             |

## Feature engineering
# HRSA Region: 'Region 3' → 3 (extract region number)
import re as _re
data['hrsa_region_num'] = (
    data['HRSA Region']
    .str.extract(r'Region\s*(\d+)', expand=False)
    .astype(float)
)

# U.S. - Mexico Border flags: 'Y'/'N' → 1/0
_border_cols = [c for c in data.columns if 'Mexico Border' in c]
for _col in _border_cols:
    data[f'{_col}_bin'] = data[_col].str.strip().str.upper().map({'Y': 1, 'N': 0})

## Clean for specific data formats (dict / list)

## Set metadata
target_name = 'Financial Assistance'
task = 'regression'
task_type = 'wide'
source = 'HRSA'

# ## Clean for the target column
# data.dropna(subset=[target_name], inplace=True)
# data.reset_index(drop=True, inplace=True)
# if task == 'regression':
#     data[target_name] = data[target_name].astype('float32')

# ## Check skewness and kurtosis if target is numeric
# if task == 'regression':
#     check_y = data[target_name].copy()
#     check_y = np.array(check_y)
#     skewness = stats.skew(check_y)
#     kurtosis = stats.kurtosis(check_y)
#     print(f'Before - skewness: {skewness} | kurtosis: {kurtosis}') # target is highly skewed and kurtotic
    
#     # apply transformation if found skewed: np.log, np.log1p, np.cbrt, np.arcsinh, np.sign(check_y) * np.log1p(np.abs(check_y))
#     if abs(skewness) > 1:
#         check_y = data[target_name].copy()
#         check_y = np.log(check_y)
#         skewness = stats.skew(check_y)
#         kurtosis = stats.kurtosis(check_y)
#         print(f'After - skewness: {skewness} | kurtosis: {kurtosis}')

# ## Apply appropriate transformation 
# ## np.log, np.log1p, np.cbrt, np.arcsinh, np.sign(check_y) * np.log1p(np.abs(check_y))
# data[target_name] = np.log(data[target_name]) # applied

# ## Clean for columns with constants or only with null values
# data = clean_constant_null(data, proportion_null=1.0)

# ## Drop duplicate columns
# data.drop_duplicates(inplace=True)
# data.reset_index(drop=True, inplace=True)

# ## Check the data criterion
# if data.shape[0] < 500:
#     raise ValueError("Dataset must have at least 500 rows.")
# if data.shape[0] > 75000:
#     print(f"Dataset has {data.shape[0]} rows, subsampling to 75000 rows.")
#     if task == 'regression':
#         data = data.sample(n=75000, random_state=42).reset_index(drop=True)
#     else:
#         data, _ = train_test_split(data, train_size=75000, random_state=42, stratify=data[target_name])
#         data.reset_index(drop=True, inplace=True)

# ## Check the number of string columns
# tabvec = TableVectorizer(high_cardinality='passthrough', cardinality_threshold=0)
# tabvec.fit_transform(data.drop(columns=target_name))
# str_cols = tabvec.kind_to_columns_['high_cardinality']
# if len(str_cols) < 2:
#         raise ValueError("Dataset must have at least 2 string-type columns.")

# ## Column-level cleaning
# # Possible leakage columns and high correlations.
# drop_col = []
# data.drop(columns=drop_col, inplace=True)

# ## Check with Cramer's-V
# cram_df = column_associations(data)
# cram_df1 = cram_df[cram_df.left_column_name==target_name].copy()
# cram_df2 = cram_df[cram_df.right_column_name==target_name].copy()
# cram_df = pd.concat([cram_df1, cram_df2])
# cram_df = cram_df.sort_values(by='cramer_v', ascending=False).reset_index(drop=True)

## Save data and config
## Change appropriately the save_folder
data_filename = 'awarded-grants'
save_folder = f'{path_configs['path_data_processed']}_feature_eng/{data_filename}'
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
