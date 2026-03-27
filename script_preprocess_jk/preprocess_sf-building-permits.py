"""
Preprocess sf building permits (from textTabBench)
The task is to predict time_to_approve (regression)
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
filename = 'textTabBench/sf_Building_Permits.csv' # Change the filename accordingly
data_path = f'{path_configs['base_path']}/data/data_raw/{filename}'
data = pd.read_csv(data_path, low_memory=False)

## Clean for backslash operations
data = clean_backslash_operations(data)

## Clean for specific data formats (dict / list)

## Dataset-level specific cleaning
filed_date = pd.to_datetime(data['Filed Date'], errors='coerce')
issued_date = pd.to_datetime(data['Issued Date'], errors='coerce')
data['time_to_approve'] = (issued_date - filed_date).dt.days
data.drop(columns=['Filed Date', 'Issued Date'], inplace=True)

## String column overview
## | Column                    | Sample values                          | Transformation                 |
## |---------------------------|----------------------------------------|--------------------------------|
## | Permit Number             | '201412012739', '201504022691', '20...| —                             |
## | Permit Type Definition    | 'otc alterations permit', 'addition...| —                             |
## | Permit Creation Date      | '12/01/2014', '04/02/2015', '09/11/...| → permit_creation_year int    |
## | Block                     | '1410', '0805', '5865'                | —                             |
## | Lot                       | '023', '034', '003'                   | —                             |
## | Street Number Suffix      | 'A', 'B', 'V'                         | —                             |
## | Street Name               | '24th', 'Ivy', 'Princeton'            | —                             |
## | Street Suffix             | 'Av', 'St', 'St'                      | —                             |
## | Unit Suffix               | 'C', 'B', 'LOWER'                     | —                             |
## | Description               | 'install 19 each simonton wi...', '...| —                             |
## | Current Status            | 'complete', 'complete', 'complete'    | —                             |
## | Current Status Date       | '02/02/2015', '11/22/2017', '12/01/...| —                             |
## | Completed Date            | '02/02/2015', '11/22/2017', '12/01/...| —                             |
## | First Construction Documen| '12/01/2014', '06/10/2016', '09/11/...| —                             |
## | Permit Expiration Date    | '11/26/2015', '06/05/2017', '09/05/...| —                             |
## | Existing Use              | 'apartments', '2 family dwelling', ...| —                             |
## | Proposed Use              | 'apartments', 'apartments', '1 fami...| —                             |
## | TIDF Compliance           |                                       | —                             |
## | Existing Construction Type| 'wood frame (5)', 'wood frame (5)',...| → existing_constr_type_num    |
## | Proposed Construction Type| 'wood frame (5)', 'wood frame (5)',...| → proposed_constr_type_num    |
## | Neighborhoods - Analysis B| 'Outer Richmond', 'Hayes Valley', '...| —                             |
## | Location                  | '(37.7825, -122.4812)', ...            | → lat, lon float              |

## Feature engineering
# Permit Creation Date: extract year
data['permit_creation_year'] = pd.to_datetime(
    data['Permit Creation Date'], errors='coerce'
).dt.year

# Existing/Proposed Construction Type: 'wood frame (5)' → extract type number
import re as _re
data['existing_constr_type_num'] = (
    data['Existing Construction Type']
    .str.extract(r'\((\d+)\)', expand=False)
    .astype(float)
)
data['proposed_constr_type_num'] = (
    data['Proposed Construction Type']
    .str.extract(r'\((\d+)\)', expand=False)
    .astype(float)
)

# Location: '(37.7825, -122.4812)' → lat, lon floats
_loc = data['Location'].str.extract(r'\(([+-]?\d+\.?\d*),\s*([+-]?\d+\.?\d*)\)')
data['location_lat'] = _loc[0].astype(float)
data['location_lon'] = _loc[1].astype(float)

## Set metadata
target_name = 'time_to_approve'
task = 'regression'
task_type = 'wide'
source = 'data.sfgov.org'

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
        check_y = np.log1p(check_y)
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
if data.shape[0] > 75000:
    print(f"Dataset has {data.shape[0]} rows, subsampling to 75000 rows.")
    if task == 'regression':
        data = data.sample(n=75000, random_state=42).reset_index(drop=True)
    else:
        data, _ = train_test_split(data, train_size=75000, random_state=42, stratify=data[target_name])
        data.reset_index(drop=True, inplace=True)

## Check the number of string columns
tabvec = TableVectorizer(high_cardinality='passthrough', cardinality_threshold=0)
tabvec.fit_transform(data.drop(columns=target_name))
str_cols = tabvec.kind_to_columns_['high_cardinality']
if len(str_cols) < 2:
        raise ValueError("Dataset must have at least 2 string-type columns.")

## Column-level cleaning
drop_col = []
data.drop(columns=drop_col, inplace=True)

## Check with Cramer's-V
cram_df = column_associations(data)
cram_df1 = cram_df[cram_df.left_column_name==target_name].copy()
cram_df2 = cram_df[cram_df.right_column_name==target_name].copy()
cram_df = pd.concat([cram_df1, cram_df2])
cram_df = cram_df.sort_values(by='cramer_v', ascending=False).reset_index(drop=True)

## Save data and config
## Change appropriately the save_folder
data_filename = 'sf-building-permits'
save_folder = f'{path_configs['path_data_processed']}/{data_filename}'
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
