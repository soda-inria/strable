"""Preprocess drug-ndc."""

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
filename = 'fda/drug-ndc.json' # Change the filename accordingly
data_path = f'{path_configs['base_path']}/data/data_raw/{filename}'
filename = open(data_path)
data = json.load(filename)
filename.close()
data = pd.DataFrame(data['results'])

## Clean for backslash operations
data = clean_backslash_operations(data)

## Dataset-level specific cleaning
data = data.sort_values(by='marketing_start_date')
data.drop_duplicates(subset=['generic_name', 'dea_schedule'], keep='last', inplace=True)
data.reset_index(drop=True, inplace=True)

## String column overview
## | Column                    | Sample values                          | Transformation                 |
## |---------------------------|----------------------------------------|--------------------------------|
## | product_ndc               | '0792-0099', '0792-0415', '0121-1008' | —                             |
## | generic_name              | 'Amobarbital Sodium', 'Phentermine ...| —                             |
## | labeler_name              | 'Siegfried USA, LLC', 'Siegfried US...| —                             |
## | brand_name                | 'Acetaminophen and Codeine P...', '...| —                             |
## | active_ingredients        | 'name: AMOBARBITAL SODIUM, s...', '...| —                             |
## | packaging                 | 'package_ndc: 0792-0099-00, ...', '...| —                             |
## | listing_expiration_date   | '20261231', '20261231', '20261231'    | → year, month int             |
## | openfda                   | '', '', 'manufacturer_name: PAI Hol...| —                             |
## | marketing_category        | 'BULK INGREDIENT', 'BULK INGREDIENT...| —                             |
## | dosage_form               | 'POWDER', 'LIQUID', 'SOLUTION'        | —                             |
## | spl_id                    | '5cedc2f3-18e6-ece6-e053-2a9...', '...| —                             |
## | product_type              | 'BULK INGREDIENT', 'BULK INGREDIENT...| —                             |
## | route                     | 'ORAL', 'ORAL', 'SUBLINGUAL'          | —                             |
## | marketing_start_date      | '19670323', '19790109', '19810821'    | → marketing_start_year int    |
## | product_id                | '0792-0099_5cedc2f3-18e6-ece...', '...| —                             |
## | application_number        | 'ANDA087508', 'ANDA086996', 'ANDA07...| —                             |
## | brand_name_base           | 'Acetaminophen and Codeine P...', '...| —                             |
## | marketing_end_date        | '20260930', '20270630', '20271116'    | → marketing_end_year int      |
## | pharm_class               | 'Full Opioid Agonists MoA, O...', '...| —                             |
## | dea_schedule              | 'CII', 'CIV', 'CV'                    | → dea_schedule_num int (1-5)  |
## | brand_name_suffix         | 'KIDNEY', 'C', 'TEST-MET'             | —                             |

## Feature engineering
# listing_expiration_date / marketing_start_date / marketing_end_date: YYYYMMDD strings → year int
data['listing_expiration_year'] = pd.to_datetime(
    data['listing_expiration_date'], format='%Y%m%d', errors='coerce'
).dt.year
data['marketing_start_year'] = pd.to_datetime(
    data['marketing_start_date'], format='%Y%m%d', errors='coerce'
).dt.year
data['marketing_end_year'] = pd.to_datetime(
    data['marketing_end_date'], format='%Y%m%d', errors='coerce'
).dt.year

# dea_schedule: 'CI'→1, 'CII'→2, 'CIII'→3, 'CIV'→4, 'CV'→5 (Roman numeral suffix)
_roman_map = {'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5}
data['dea_schedule_num'] = data['dea_schedule'].str.strip().str.lstrip('C').map(_roman_map)

## Clean for specific data formats (dict / list)
# Clean for dict-type columns
data = clean_list_type_col(data, 'active_ingredients')
data = clean_dict_type_col(data, 'active_ingredients', clean_type='to_string')
data = clean_list_type_col(data, 'packaging')
data = clean_dict_type_col(data, 'packaging', clean_type='to_string')
data = clean_dict_type_col(data, 'openfda', clean_type='to_string')
data = clean_list_type_col(data, 'openfda')
# Clean for list-type columns
for col in data.columns:
    if data[col].isnull().sum() == data.shape[0]:
        pass
    else:
        if isinstance(data[~data[col].isnull()][col].iloc[0], list):
            data = clean_list_type_col(data, col)

## Set metadata
target_name = 'dea_schedule'
task = 'm-classification'
task_type = 'wide'
source = 'fda'

## Clean for the target column
data.dropna(subset=[target_name], inplace=True)
data.reset_index(drop=True, inplace=True)
data = data[data[target_name]!='CI'].reset_index(drop=True)
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
        check_y = np.log(check_y)
        skewness = stats.skew(check_y)
        kurtosis = stats.kurtosis(check_y)
        print(f'After - skewness: {skewness} | kurtosis: {kurtosis}')

## Apply appropriate transformation 
## np.log, np.log1p, np.cbrt, np.arcsinh, np.sign(check_y) * np.log1p(np.abs(check_y))
# data[target_name] = np.log(data[target_name]) # applied

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
data_filename = 'drug-ndc'
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
dtype_method = 'num-str'
embed_method = 'tarenc'

scoring, result_criterion = set_score_criterion(task)

X_train, X_test, y_train, y_test, duration_embed, cat_features = embed_table(
    data_name,
    n_split,
    fold_index,
    dtype_method,
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
