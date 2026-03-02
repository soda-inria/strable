"""Common functions used for evaluation."""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime

def load_data(data_name):
    """Loads the locally saved raw data."""

    from configs.path_configs import path_configs

    # Path of the folder containing data
    data_folder = path_configs["path_data_processed"]

    # Dataset
    data_path = f"{data_folder}/{data_name}/data.parquet"
    data = pd.read_parquet(data_path)
    data.fillna(value=np.nan, inplace=True)

    # Configs
    config_path = f"{data_folder}/{data_name}/config.json"
    filename = open(config_path)
    data_config = json.load(filename)
    filename.close()

    return data, data_config


def set_split_cv(data, data_config, n_split, fold_index):
    """Train/Test split of with K-fold"""

    from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold

    # Load data, Set preliminary
    target_name = data_config["target_name"]
    task = data_config["task"]

    # Set the data
    X = data.drop(columns=target_name)
    y = data[target_name].copy()
    y = np.array(y)

    if task == "regression":
        cv = RepeatedKFold(n_splits=n_split, n_repeats=1, random_state=1234)
    else:
        cv = RepeatedStratifiedKFold(n_splits=n_split, n_repeats=1, random_state=1234)
    split_index = list(cv.split(X, y))[fold_index]

    X_train, X_test = X.iloc[split_index[0]], X.iloc[split_index[1]]
    y_train, y_test = y[split_index[0]], y[split_index[1]]

    return X_train, X_test, y_train, y_test


def set_split_random(data, data_config, num_train, random_state):
    """Train/Test split of the data with random split."""

    from sklearn.model_selection import train_test_split

    # Load data, Set preliminary
    target_name = data_config["target_name"]
    task = data_config["task"]

    # Set the data
    X = data.drop(columns=target_name)
    y = data[target_name].copy()
    y = np.array(y)
    stratify = y if "classification" in task else None

    split_index = train_test_split(
        np.arange(len(X)),
        train_size=num_train,
        random_state=random_state,
        stratify=stratify,
    )

    X_train, X_test = X.iloc[split_index[0]], X.iloc[split_index[1]]
    y_train, y_test = y[split_index[0]], y[split_index[1]]

    return X_train, X_test, y_train, y_test


def col_names_per_type(data, target_name=None):
    """
    Function to detect column names of different data types.
    This function relies on the data-type detection from the `skrub` package.
    """

    from skrub import TableVectorizer

    data_ = data.copy()
    if target_name is not None:
        data_.drop(columns=target_name, inplace=True)
    tabvec = TableVectorizer(cardinality_threshold=0, high_cardinality="passthrough")
    data_ = tabvec.fit_transform(data_)
    num_col = tabvec.kind_to_columns_["numeric"]
    cat_col = tabvec.kind_to_columns_["high_cardinality"]
    dat_col = tabvec.kind_to_columns_["datetime"]

    return num_col, cat_col, dat_col


def shorten_param(param_name):
    """Shorten the param_names for column names in search results."""

    if "__" in param_name:
        return param_name.rsplit("__", 1)[1]
    return param_name


def check_pred_output(y_train, y_pred):
    """Set the output as the mean of train data if it is nan."""

    if np.isnan(y_pred).sum() > 0:
        mean_pred = np.mean(y_train)
        y_pred[np.isnan(y_pred)] = mean_pred
    return y_pred


def reshape_pred_output(y_pred):
    """Reshape the predictive output accordingly."""

    from scipy.special import softmax

    num_pred = len(y_pred)
    if y_pred.shape == (num_pred, 2):
        y_pred = y_pred[:, 1]
    elif y_pred.shape == (num_pred, 1):
        y_pred = y_pred.ravel()
    else:
        if len(y_pred.shape) > 1:
            y_pred = softmax(y_pred, axis=1)
        else:
            pass
    return y_pred


def set_score_criterion(task):
    """Set scoring method for CV and score criterion in final result."""

    if task == "regression":
        scoring = "r2"
        score_criterion = ["r2", "rmse"]
    else:
        if task == "m-classification":
            scoring = "roc_auc_ovr"
        else:
            scoring = "roc_auc"
        score_criterion = [
            "roc_auc",
            "brier_score_loss",
            "f1_weighted",
        ]
    score_criterion += ["preprocess_time"]
    score_criterion += ["param_search_time"]
    score_criterion += ["inference_time"]
    score_criterion += ["run_time"]
    return scoring, score_criterion


def return_score(y_target, y_prob, y_pred, task):
    """Return score results for given task."""

    from sklearn.metrics import (
        r2_score,
        root_mean_squared_error,
        roc_auc_score,
        brier_score_loss,
        f1_score,
    )

    if task == "regression":
        score_r2 = r2_score(y_target, y_pred)
        score_rmse = root_mean_squared_error(y_target, y_pred)
        return score_r2, score_rmse
    else:
        if task == "m-classification":
            score_auc = roc_auc_score(
                y_target,
                y_prob,
                multi_class="ovr",
                average="macro",
            )
        else:
            score_auc = roc_auc_score(y_target, y_prob)
        score_brier = brier_score_loss(y_target, y_prob)
        score_f1 = f1_score(y_target, y_pred, average="weighted")
        return score_auc, score_brier, score_f1

def _calculate_tabpfn_output(estimator, X_test, batch_size, task):
    """
    Optimized TabPFN output calculation.
    1. Removes double inference (calculates y_pred from y_prob).
    2. Fixes concatenation overhead.
    3. Handles remainders automatically.
    """
    n_samples = len(X_test)
    
    # Storage for batch results
    output_list = []
    
    # Loop using range with step to handle batches + remainder automatically
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        X_batch = X_test[start_idx:end_idx]
        
        if task == "regression":
            # For regression, we just need the prediction
            batch_pred = estimator.predict(X_batch)
            output_list.append(batch_pred)
        else:
            # For classification, ONLY run predict_proba (the expensive part)
            batch_prob = estimator.predict_proba(X_batch)
            output_list.append(batch_prob)

    # Concatenate all batches once at the end
    if not output_list:
        return np.array([]), np.array([])

    concatenated_output = np.concatenate(output_list, axis=0)

    if task == "regression":
        y_pred = concatenated_output
        y_prob = None
    else:
        y_prob = concatenated_output
        # OPTIMIZATION: Derive y_pred directly from y_prob (instant)
        # We map the index of the highest probability to the actual class label
        class_indices = np.argmax(y_prob, axis=1)
        y_pred = estimator.classes_[class_indices]

    return y_prob, y_pred


def calculate_output(X_test, estimator, task):
    """Calculate output for different tasks and estimators."""

    BATCH_SIZE=1563
    print(f'Current hour before calculate_output: {datetime.now().strftime("%H:%M:%S")}, Batch size: {BATCH_SIZE}')

    est_name = estimator.__class__.__name__
    if task == "regression":
        if est_name == "TabPFNRegressor":
            y_prob, y_pred = _calculate_tabpfn_output(
                estimator,
                X_test,
                BATCH_SIZE,
                task,
            )
        else:
            y_prob = None
            y_pred = estimator.predict(X_test)
    else:
        if est_name == "RidgeClassifierCV":
            y_prob = estimator.decision_function(X_test)
            y_prob = 1 / (1 + np.exp(-y_prob))
            y_pred = estimator.predict(X_test)
        elif est_name == "TabPFNClassifier":
            y_prob, y_pred = _calculate_tabpfn_output(
                estimator,
                X_test,
                BATCH_SIZE,
                task,
            )
        else:
            y_prob = estimator.predict_proba(X_test)
            y_pred = estimator.predict(X_test)

    return y_prob, y_pred



def assign_estimator(
    estim_method,
    task,
    device,
    best_params={},
    cat_features=None,
):
    """Assign the specific estimator to train model."""

    # No search estimators
    if estim_method == "ridge":

        from sklearn.linear_model import RidgeCV, RidgeClassifierCV

        fixed_params = dict()
        fixed_params["alphas"] = [1e-2, 1e-1, 1, 10, 100]
        if task == "regression":
            estimator_ = RidgeCV(**fixed_params)
        else:
            estimator_ = RidgeClassifierCV(**fixed_params)

    elif estim_method == "tabpfn":

        from tabpfn import TabPFNRegressor, TabPFNClassifier

        fixed_params = dict()
        fixed_params["device"] = device
        fixed_params["ignore_pretraining_limits"] = True

        # TabPFN will automatically download or load the weights from its default cache
        if task == "regression":
            estimator_ = TabPFNRegressor(**fixed_params)
        else:
            estimator_ = TabPFNClassifier(**fixed_params)
    
    elif estim_method == "tabstar":
        
        from tabstar.tabstar_model import TabSTARClassifier, TabSTARRegressor

        fixed_params = dict()
        fixed_params["device"] = device
        fixed_params["random_state"] = 1234
        fixed_params["keep_model"] = False
        fixed_params["output_dir"] = ".tabstar_checkpoint/" + str(uuid.uuid4()) + '/'
        # TabSTAR and HuggingFace will handle downloading this model to the user's default cache
        fixed_params["pretrain_dataset_or_path"] = "alana89/TabSTAR"

        if task == "regression":
            estimator_ = TabSTARRegressor(**fixed_params)
        else:
            estimator_ = TabSTARClassifier(**fixed_params)

    elif estim_method == "contexttab":

        from sap_rpt_oss import SAP_RPT_OSS_Classifier, SAP_RPT_OSS_Regressor

        # device is automatically detected
        # random_state is already defined as random_state=self.seed + bagging_index
        
        if task == "regression":
            estimator_ = SAP_RPT_OSS_Regressor()
        else:
            estimator_ = SAP_RPT_OSS_Classifier()

    elif estim_method == "realtabpfn":

        from tabpfn import TabPFNRegressor, TabPFNClassifier
        from configs.path_configs import path_configs
        from glob import glob

        fixed_params = dict()
        fixed_params["device"] = device
        fixed_params["ignore_pretraining_limits"] = True

        # Safely check for the custom TabPFN weights
        model_type = "regressor" if task == "regression" else "classifier"
        checkpoint_pattern = f"{path_configs['models']}/tabpfn_2_5/*tabpfn-v2.5-{model_type}-v2.5_real*"
        checkpoints = glob(checkpoint_pattern)
        
        if not checkpoints:
            raise FileNotFoundError(
                f"\n❌ Custom TabPFN weights not found!\n"
                f"Expected to find a model matching: {checkpoint_pattern}\n"
                f"Please download the weights and place them in the correct directory."
            )
            
        fixed_params["model_path"] = checkpoints[0]
        
        if task == "regression":
            estimator_ = TabPFNRegressor(**fixed_params)
        else:
            estimator_ = TabPFNClassifier(**fixed_params)

    elif estim_method == "logistic":

        from sklearn.linear_model import LogisticRegressionCV

        fixed_params = dict()
        fixed_params["Cs"] = [1e-2, 1e-1, 1, 10, 100]
        estimator_ = LogisticRegressionCV(**fixed_params)

    elif estim_method == "realmlp":

        import uuid
        from pytabkit import RealMLP_HPO_Regressor, RealMLP_HPO_Classifier

        fixed_params = dict()
        fixed_params["device"] = device
        fixed_params["n_cv"] = 8
        fixed_params["n_repeats"] = 1
        fixed_params["n_hyperopt_steps"] = 100
        fixed_params["random_state"] = 1234
        fixed_params["tmp_folder"] = "./pytabkit/" + str(uuid.uuid4())
        fixed_params["verbosity"] = 0
        if task == "regression":
            estimator_ = RealMLP_HPO_Regressor(**fixed_params, **best_params)
        else:
            fixed_params["val_metric_name"] = "1-auc_ovr"
            estimator_ = RealMLP_HPO_Classifier(**fixed_params, **best_params)

    elif estim_method == "tabm":

        import uuid
        from pytabkit import TabM_HPO_Regressor, TabM_HPO_Classifier

        fixed_params = dict()
        fixed_params["device"] = device
        fixed_params["n_cv"] = 8
        fixed_params["n_repeats"] = 1
        fixed_params["n_hyperopt_steps"] = 100
        fixed_params["random_state"] = 1234
        fixed_params["tmp_folder"] = "./pytabkit/" + str(uuid.uuid4())
        fixed_params["verbosity"] = 0
        if task == "regression":
            estimator_ = TabM_HPO_Regressor(**fixed_params, **best_params)
        else:
            fixed_params["val_metric_name"] = "1-auc_ovr"
            estimator_ = TabM_HPO_Classifier(**fixed_params, **best_params)
    
    elif estim_method == "tarte":
        from tarte_ai import TARTEFinetuneRegressor, TARTEFinetuneClassifier

        fixed_params = dict()
        fixed_params["device"] = device
        fixed_params["num_model"] = 8
        fixed_params["n_jobs"] = 8
        fixed_params["num_heads"] = 24   # Based on paper Page 6
        fixed_params["batch_size"] = 32 
        fixed_params["disable_pbar"] = True
        fixed_params["random_state"] = 1234

        if task == "regression":
            estimator_ = TARTEFinetuneRegressor(**fixed_params, **best_params)
        else:
            fixed_params['loss'] = 'categorical_crossentropy'
            estimator_ = TARTEFinetuneClassifier(**fixed_params, **best_params)

    # GBDT Search estimators
    elif estim_method == "xgb":

        from xgboost import callback, XGBRegressor, XGBClassifier

        early_stopping_patience = 300
        callbacks = [
            callback.EarlyStopping(
                rounds=early_stopping_patience,
                min_delta=1e-3,
                save_best=True,
                maximize=False,
            )
        ]
        fixed_params = dict()
        fixed_params["n_estimators"] = 1000
        fixed_params["callbacks"] = callbacks

        if task == "regression":
            estimator_ = XGBRegressor(**fixed_params, **best_params)
        else:
            estimator_ = XGBClassifier(**fixed_params, **best_params)

    elif estim_method == "catboost":

        from catboost import CatBoostRegressor, CatBoostClassifier

        fixed_params = dict()
        fixed_params["allow_writing_files"] = False
        fixed_params["verbose"] = False
        fixed_params["thread_count"] = 1
        fixed_params["iterations"] = 1000
        fixed_params["od_type"] = "Iter"
        fixed_params["od_wait"] = 300
        fixed_params["cat_features"] = cat_features

        if task == "regression":
            estimator_ = CatBoostRegressor(**fixed_params, **best_params)
        else:
            estimator_ = CatBoostClassifier(**fixed_params, **best_params)

    # Scikit-learn search estimators
    elif estim_method == "histgb":

        from sklearn.ensemble import (
            HistGradientBoostingRegressor,
            HistGradientBoostingClassifier,
        )

        fixed_params = dict()
        fixed_params["early_stopping"] = True
        fixed_params["n_iter_no_change"] = 50
        fixed_params["random_state"] = 1234
        if task == "regression":
            estimator_ = HistGradientBoostingRegressor(
                **fixed_params,
                **best_params,
            )
        else:
            estimator_ = HistGradientBoostingClassifier(
                **fixed_params,
                **best_params,
            )

    elif estim_method == "randomforest":

        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

        fixed_params = dict()
        fixed_params["n_estimators"] = 250
        fixed_params["random_state"] = 1234
        fixed_params["n_jobs"] = len(os.sched_getaffinity(0))
        if task == "regression":
            estimator_ = RandomForestRegressor(**fixed_params, **best_params)
        else:
            estimator_ = RandomForestClassifier(**fixed_params, **best_params)

    elif estim_method == "extrees":

        from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier

        fixed_params = dict()
        fixed_params["n_estimators"] = 250
        fixed_params["random_state"] = 1234
        fixed_params["n_jobs"] = len(os.sched_getaffinity(0))
        if task == "regression":
            estimator_ = ExtraTreesRegressor(**fixed_params, **best_params)
        else:
            estimator_ = ExtraTreesClassifier(**fixed_params, **best_params)

    return estimator_
