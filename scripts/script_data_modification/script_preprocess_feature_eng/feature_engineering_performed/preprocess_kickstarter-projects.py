"""Preprocess kickstarter-projects dataset
The task is to predict status (classification) 
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
filename = 'kickstarter-projects'
data_path = f'{path_configs['path_data_processed']}/{filename}/data.parquet'
data = pd.read_parquet(data_path)

## Clean for backslash operations
# data = clean_backslash_operations(data)

## Clean for specific data formats (dict / list)

## Dataset-level specific cleaning

## String column overview
## | Column                    | Sample values                          | Transformation                 |
## |---------------------------|----------------------------------------|--------------------------------|
## | name                      | 'whiddy', 'Still Laughing LIVE Come...| —                             |
## | currency                  | 'USD', 'USD', 'USD'                   | —                             |
## | main_category             | 'journalism', 'theater', 'music'      | —                             |
## | sub_category              | 'Web', 'Theater', 'Rock'              | —                             |
## | launched_at               | '2014-08-28 22:03:39', '2015-03-19 ...| —                             |
## | deadline                  | '2014-09-27 22:03:39', '2015-04-18 ...| —                             |
## | city                      | 'Las Vegas', 'Parsippany', 'Ruleville'| —                             |
## | state                     | 'NV', 'NJ', 'MS'                      | —                             |
## | country                   | 'US', 'US', 'US'                      | —                             |
## | status                    | 'failed', 'successful', 'successful'  | —                             |
## | start_Q                   | 'Q3', 'Q1', 'Q3'                      | → start_quarter_num int       |
## | end_Q                     | 'Q3', 'Q2', 'Q3'                      | → end_quarter_num int         |

## Feature engineering
# start_Q / end_Q: 'Q3' → 3 (quarter number)
data['start_quarter_num'] = data['start_Q'].str.extract(r'Q(\d+)', expand=False).astype(float)
data['end_quarter_num'] = data['end_Q'].str.extract(r'Q(\d+)', expand=False).astype(float)

# launched_at / deadline: extract year and month
data['launched_year'] = pd.to_datetime(data['launched_at'], errors='coerce').dt.year
data['launched_month'] = pd.to_datetime(data['launched_at'], errors='coerce').dt.month
data['deadline_year'] = pd.to_datetime(data['deadline'], errors='coerce').dt.year
data['deadline_month'] = pd.to_datetime(data['deadline'], errors='coerce').dt.month

## Set metadata
target_name = 'status'
task = 'b-classification'
task_type = 'wide'
source = 'webrobots.io'

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
# drop_col = []
# drop_col.append('usd_pledged') # leakage
# data.drop(columns=drop_col, inplace=True)

# ## Check with Cramer's-V
# cram_df = column_associations(data)
# cram_df1 = cram_df[cram_df.left_column_name==target_name].copy()
# cram_df2 = cram_df[cram_df.right_column_name==target_name].copy()
# cram_df = pd.concat([cram_df1, cram_df2])
# cram_df = cram_df.sort_values(by='cramer_v', ascending=False).reset_index(drop=True)

## Save data and config
## Change appropriately the save_folder
data_filename = 'kickstarter-projects'
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
