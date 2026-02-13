"""Functions to embed (or prepare) the preprocessed table."""

import time
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from src.utils_evaluation import load_data, set_split_cv, col_names_per_type


def embed_table(
    data_name,
    n_split,
    fold_index,
    dtype_method,
    embed_method,
):
    """Function to encode the tables to prepare for evaluations."""

    # Load data
    data, data_config = load_data(data_name)
    _, cat_col_names, _ = col_names_per_type(data, data_config["target_name"])

    # Preliminaries for different cases
    if dtype_method == "num-only":  # Extract only the numerical
        data.drop(columns=cat_col_names, inplace=True)
    elif dtype_method == "str-only":  # Extract only the string
        keep_cols = cat_col_names + [data_config["target_name"]]
        data = data[keep_cols]
    else:
        pass

    # Clean the target for classification problems
    if data_config["task"] != "regression":
        label_enc = LabelEncoder()
        data[data_config["target_name"]] = label_enc.fit_transform(
            data[data_config["target_name"]]
        )

    # Set data with split
    X_train, X_test, y_train, y_test = set_split_cv(
        data,
        data_config,
        n_split,
        fold_index,
    )
    y_train, y_test = np.array(y_train), np.array(y_test)

    # Measure time
    duration_embed = 0
    cat_features = None
    start_time = time.perf_counter()

    # Run encodings
    if embed_method == "tabvec":
        X_train, X_test = prepare_tabvec(X_train, X_test)
    elif embed_method == "tarenc":
        X_train, X_test = prepare_tarenc(
            X_train,
            X_test,
            y_train,
            data_config["task"],
        )
    elif "llm-" in embed_method:
        X_train, X_test, duration_llm = prepare_llm(
            X_train,
            X_test,
            data_name,
            embed_method,
        )
        duration_embed += duration_llm
    elif embed_method == "catboost":
        X_train, X_test, cat_features = prepare_catboost(X_train, X_test)
    elif embed_method == "tabpfn":
        X_train, X_test = prepare_tabpfn(X_train, X_test)
    elif embed_method == "tabstar":
        X_train, X_test = prepare_tabstar(X_train, X_test)
        # convert y_train to pandas Series if not already (required for TabStar)
        if not isinstance(y_train, pd.Series):
            y_train = pd.Series(y_train, index=X_train.index)
    elif embed_method == "tarte":
        X_train, X_test = prepare_tarte(X_train, X_test)
    elif embed_method == "tarte-ft":
        X_train, X_test = prepare_tarte_ft(X_train, X_test, y_train)

    end_time = time.perf_counter()
    duration_embed += round(end_time - start_time, 4)

    return X_train, X_test, y_train, y_test, duration_embed, cat_features


def prepare_tabvec(X_train, X_test):
    """Function to prepare with StringEncoder(TabVec)."""

    from skrub import StringEncoder, TableVectorizer, SquashingScaler

    # First, set the dataframe in appropriate formats
    cleaner = TableVectorizer(cardinality_threshold=0, high_cardinality="passthrough")
    X_train = cleaner.fit_transform(X_train)
    X_test = cleaner.transform(X_test)

    # Encode
    text_encoder = StringEncoder(random_state=1234)
    num_transformer = SquashingScaler()
    encoder = TableVectorizer(
        cardinality_threshold=0,
        high_cardinality=text_encoder,
        numeric=num_transformer,
    )

    X_train = encoder.fit_transform(X_train)
    X_test = encoder.transform(X_test)

    return X_train, X_test


def prepare_tarenc(X_train, X_test, y_train, task):
    """Function to prepare with TargetEncoder."""

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import TargetEncoder
    from skrub import TableVectorizer, SquashingScaler

    # First, set the dataframe in appropriate format
    cleaner = TableVectorizer(cardinality_threshold=0, high_cardinality="passthrough")
    X_train = cleaner.fit_transform(X_train)
    X_test = cleaner.transform(X_test)

    # Encode
    if task == "regression":
        target_type = "continuous"
    else:
        if task == "m-classification":
            target_type = "multiclass"
        else:
            target_type = "binary"

    tarenc = TargetEncoder(
        categories="auto",
        target_type=target_type,
        random_state=1234,
    )
    text_encoder = Pipeline(
        [
            ("tarenc", tarenc),
            ("squash", SquashingScaler()),
        ]
    )
    num_transformer = SquashingScaler()
    encoder = TableVectorizer(
        cardinality_threshold=0,
        high_cardinality=text_encoder,
        numeric=num_transformer,
    )

    X_train = encoder.fit_transform(X_train, y=y_train)
    X_test = encoder.transform(X_test)

    return X_train, X_test


def prepare_llm(X_train, X_test, data_name, llm_model_name):
    """Function to prepare with LLM embeddings."""

    from skrub import TableVectorizer, SquashingScaler
    from src.llm_encoder import LLM_Encoder
    from configs.path_configs import path_configs

    # First, set the dataframe in appropriate format
    cleaner = TableVectorizer(cardinality_threshold=0, high_cardinality="passthrough")
    X_train = cleaner.fit_transform(X_train)
    X_test = cleaner.transform(X_test)

    # Encode
    cached_llm_embedding_path = f"{path_configs['base_path']}/data/llm_embeding/{llm_model_name}/{llm_model_name}|{data_name}.parquet"
    text_encoder = LLM_Encoder(
        cached_llm_embedding_path=cached_llm_embedding_path,
        n_components=30,
        random_state=1234,
    )
    num_transformer = SquashingScaler()
    encoder = TableVectorizer(
        cardinality_threshold=0,
        high_cardinality=text_encoder,
        numeric=num_transformer,
    )

    X_train = encoder.fit_transform(X_train)
    X_test = encoder.transform(X_test)

    # Extraction time
    time_folder = f"{path_configs['base_path']}/data/llm_embed_time/{llm_model_name}"
    time_path = f"{time_folder}/{llm_model_name}|{data_name}.npy"

    return X_train, X_test, np.load(time_path)


def prepare_catboost(X_train, X_test):
    """Function to prepare with CatBoost(Internal)."""

    from skrub import TableVectorizer

    tabvec = TableVectorizer(cardinality_threshold=0, high_cardinality="passthrough")
    X_train = tabvec.fit_transform(X_train)
    X_test = tabvec.transform(X_test)

    # Apply necessary encodings and extract cat_features for catboost
    categories = tabvec.kind_to_columns_["high_cardinality"]
    X_train[categories] = X_train[categories].replace(np.nan, "nan", regex=True)
    X_test[categories] = X_test[categories].replace(np.nan, "nan", regex=True)
    cat_features = [X_train.columns.get_loc(col) for col in categories]

    return X_train, X_test, cat_features


def prepare_tabpfn(X_train, X_test):
    """Function to prepare with TabPFN(Internal)."""

    from skrub import TableVectorizer

    tabvec = TableVectorizer(cardinality_threshold=0, high_cardinality="passthrough")
    X_train = tabvec.fit_transform(X_train)
    X_test = tabvec.transform(X_test)

    # Apply necessary dtype configurations
    categories = tabvec.kind_to_columns_["high_cardinality"]
    for col in categories:
        X_train[col] = X_train[col].astype("category")
        X_test[col] = X_test[col].astype("category")

    return X_train, X_test


def prepare_tabstar(X_train, X_test):
    """Function to prepare with TabSTAR(Internal)."""

    from skrub import TableVectorizer

    tabvec = TableVectorizer(cardinality_threshold=0, high_cardinality="passthrough")
    X_train = tabvec.fit_transform(X_train)
    X_test = tabvec.transform(X_test)

    # Convert all non-numerical columns to string
    categories = tabvec.kind_to_columns_["high_cardinality"]
    for col in categories:
        X_train[col] = X_train[col].astype(str)
        X_test[col] = X_test[col].astype(str)

    return X_train, X_test


def prepare_contexttab(X_train, X_test):
    """Function to prepare with ContextTab(Internal)."""

    from skrub import TableVectorizer

    tabvec = TableVectorizer(cardinality_threshold=0, high_cardinality="passthrough")
    X_train = tabvec.fit_transform(X_train)
    X_test = tabvec.transform(X_test)

    X_train = tabvec.fit_transform(X_train)
    X_test = tabvec.transform(X_test)

    return X_train, X_test


def prepare_tarte(X_train, X_test):
    """Function to prepare with TARTE-Finetune(Internal)."""

    from skrub import TableVectorizer, SquashingScaler, ApplyToCols, DatetimeEncoder
    from sklearn.pipeline import Pipeline
    from tarte_ai import TARTE_TablePreprocessor
    from tarte_ai import TARTE_TableEncoder

    num_transformer = SquashingScaler()
    datetime_pipe = Pipeline(
        [
            ("datetime", ApplyToCols(DatetimeEncoder())),
            ("squash", SquashingScaler()),
        ]
    )
    tabvec = TableVectorizer(
        cardinality_threshold=0,
        high_cardinality="passthrough",
        numeric=num_transformer,
        datetime=datetime_pipe,
    )
    X_train = tabvec.fit_transform(X_train)
    X_test = tabvec.transform(X_test)

    string_cols = tabvec.kind_to_columns_["high_cardinality"]

    tarte_tab_prepper = TARTE_TablePreprocessor()
    tarte_tab_encoder = TARTE_TableEncoder(layer_index=[0,2])
    tarte_pipe = Pipeline([("prep", tarte_tab_prepper), ("tabenc", tarte_tab_encoder)])

    X_train_tarte = tarte_pipe.fit_transform(X_train)
    X_test_tarte = tarte_pipe.transform(X_test)

    n_features = X_train_tarte.shape[1]
    X_train_tarte = pd.DataFrame(
        X_train_tarte, columns=[f"X{i}" for i in range(n_features)], index=X_train.index
    )
    X_test_tarte = pd.DataFrame(
        X_test_tarte, columns=[f"X{i}" for i in range(n_features)], index=X_test.index
    )

    X_train = pd.concat([X_train.drop(columns=string_cols), X_train_tarte], axis=1)
    X_test = pd.concat([X_test.drop(columns=string_cols), X_test_tarte], axis=1)

    return X_train, X_test


def prepare_tarte_ft(X_train, X_test, y_train):
    """Function to prepare with TARTE-Finetune(Internal)."""

    from skrub import TableVectorizer
    from tarte_ai import TARTE_TablePreprocessor

    tabvec = TableVectorizer(cardinality_threshold=0, high_cardinality="passthrough")
    X_train = tabvec.fit_transform(X_train)
    X_test = tabvec.transform(X_test)

    tarte_tab_prepper = TARTE_TablePreprocessor()
    X_train = tarte_tab_prepper.fit_transform(X_train, y_train)
    X_test = tarte_tab_prepper.transform(X_test)

    return X_train, X_test


# def prepare_tabvec():

#     return None
