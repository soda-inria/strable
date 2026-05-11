"""Preprocess HRSA Health Professional Shortage Areas.
The task is to predict HPSA Score (regression)
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
filename = 'health-professional-shortage-areas'
data_path = f'{path_configs['path_data_processed']}/{filename}/data.parquet'
data = pd.read_parquet(data_path)

## Clean for backslash operations
# data = clean_backslash_operations(data)

# ## Dataset-level specific cleaning
# data = data[data['HPSA Status'].isin(['Designated'])].reset_index(drop=True)
# data.drop_duplicates(subset='HPSA Name', inplace=True)
# data.reset_index(drop=True, inplace=True)

# data = data.dropna(subset='HPSA Status').reset_index(drop=True)
# data['HPSA Name'].value_counts()
# data['HPSA Status'].value_counts()
# data.columns.tolist()

## String column overview
## | Column                    | Sample values                          | Transformation                 |
## |---------------------------|----------------------------------------|--------------------------------|
## | HPSA Name                 | 'Republic of the Marshall Is...', '...| —                             |
## | HPSA ID                   | '6684864122', '6669996602', '666916...| —                             |
## | Designation Type          | 'Geographic HPSA', 'Federally Quali...| —                             |
## | HPSA Discipline Class     | 'Dental Health', 'Dental Health', '...| —                             |
## | Primary State Abbreviation| 'MH', 'GU', 'GU'                      | —                             |
## | HPSA Designation Date     | '2018-12-31', '2003-11-21', '2020-0...| → designation_year int        |
## | HPSA Designation Last Upda| '2020-12-31', '2021-09-11', '2020-0...| —                             |
## | Metropolitan Indicator    | 'Unknown', 'Non-Metropolitan', 'Unk...| —                             |
## | HPSA Geography Identificat| '68090', 'POINT', 'POINT'             | —                             |
## | HPSA Degree of Shortage   | 'Not applicable', '12', 'Not applic...| → shortage_degree float (NaN if N/A)|
## | HPSA Formal Ratio         | '78255:1', '20173:1', '31188:1'       | → formal_ratio_num int (numerator)|
## | HPSA Population Type      | 'Geographic Population', 'Low Incom...| —                             |
## | Rural Status              | 'Rural', 'Rural', 'Rural'             | → rural_bin 1/0               |
## | BHCMIS Organization Identi| '093530', '11E01249', '091920'        | —                             |
## | Common County Name        | 'Enewetak, MH', 'Guam, GU', 'Guam, GU'| —                             |
## | Common Region Name        | 'Region 9', 'Region 9', 'Region 9'    | → region_num int              |
## | Common State Abbreviation | 'MH', 'GU', 'GU'                      | —                             |
## | Common State County FIPS C| '68090', '66010', '66010'             | —                             |
## | Common State Name         | 'Marshall Islands', 'Guam', 'Guam'    | —                             |
## | County Equivalent Name    | 'Enewetak', 'Guam', 'Guam'            | —                             |
## | County or County Equivalen| '090', '010', '010'                   | —                             |
## | HPSA Address              | 'MANUEL F.L. GUERRERO BUILDING', 'C...| —                             |
## | HPSA City                 | 'Hagatna', 'Mangilao', 'Mangilao'     | —                             |
## | HPSA Component Name       | 'Enewetak', 'GOVERNMENT OF GUAM- DE...| —                             |
## | HPSA Component Source Iden| '6669996602', '6669164203', '666201...| —                             |
## | HPSA Component State Abbre| 'MH', 'GU', 'FM'                      | —                             |
## | HPSA Component Type Code  | 'SCTY', 'UNK', 'UNK'                  | —                             |
## | HPSA Component Type Descri| 'Single County', 'Unknown', 'Unknown' | —                             |
## | HPSA Designation Populatio| 'Geographic Population', 'Federally...| —                             |
## | HPSA Metropolitan Indicato| '0', 'N', '0'                         | —                             |
## | HPSA Population Type Code | 'TRC', 'LI', 'TRC'                    | —                             |
## | HPSA Postal Code          | '96932', '96913', '96913'             | —                             |
## | HPSA Provider Ratio Goal  | '5000:1', '1500:0', '4000:1'          | —                             |
## | HPSA Type Code            | 'Hpsa Geo', 'FQHC', 'PRSN'            | —                             |
## | Primary State Name        | 'Marshall Islands', 'Guam', 'Guam'    | —                             |
## | Provider Type             | 'Not Applicable', 'Not Applicable',...| —                             |
## | Rural Status Code         | 'R', 'R', 'R'                         | —                             |
## | State Abbreviation        | 'MH', 'GU', 'GU'                      | —                             |
## | State and County Federal I| '68090', '66010', '66010'             | —                             |
## | State Name                | 'Marshall Islands', 'Guam', 'Guam'    | —                             |
## | U.S. - Mexico Border 100 K| 'N', 'N', 'N'                         | —                             |
## | U.S. - Mexico Border Count| 'N', 'N', 'N'                         | —                             |

## Feature engineering
# HPSA Designation Date: extract year
data['designation_year'] = pd.to_datetime(
    data['HPSA Designation Date'], errors='coerce'
).dt.year

# HPSA Degree of Shortage: '12', 'Not applicable' → float (NaN for non-numeric)
data['shortage_degree'] = pd.to_numeric(
    data['HPSA Degree of Shortage'], errors='coerce'
)

# HPSA Formal Ratio: '78255:1' → extract numerator integer
data['formal_ratio_num'] = (
    data['HPSA Formal Ratio']
    .str.extract(r'^(\d+):', expand=False)
    .astype(float)
)

# Common Region Name: 'Region 9' → 9 (extract region number)
import re as _re
data['region_num'] = (
    data['Common Region Name']
    .str.extract(r'Region\s*(\d+)', expand=False)
    .astype(float)
)

# Rural Status: 'Rural'→1, others→0
data['rural_bin'] = (data['Rural Status'].str.strip().str.lower() == 'rural').astype(int)

## Clean for specific data formats (dict / list)

## Set metadata
target_name = 'HPSA Score'
task = 'regression'
task_type = 'wide'
source = 'HRSA'

## Clean for the target column
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

## Apply appropriate transformation 
## np.log, np.log1p, np.cbrt, np.arcsinh, np.sign(check_y) * np.log1p(np.abs(check_y))
# data[target_name] = np.log(data[target_name]) # applied

## Clean for columns with constants or only with null values
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
data_filename = 'health-professional-shortage-areas'
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
