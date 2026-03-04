"""Functions to embed (or prepare) the preprocessed table."""

import os
import time
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from skrub import DatetimeEncoder
from src.utils_evaluation import load_data, set_split_cv, col_names_per_type


def embed_table(data_name, n_split, fold_index, embed_method, normalization, no_pca):
    """Function to encode the tables to prepare for evaluations."""

    # Preliminaries
    encode_method = embed_method.split("_")[-1]
    dtype_method = embed_method.split("_")[0]

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

    # Run encoding of string columns
    if encode_method == "tabvec":
        X_train, X_test = prepare_tabvec(X_train, X_test)
    elif encode_method == "tarenc":
        X_train, X_test = prepare_tarenc(
            X_train,
            X_test,
            y_train,
            data_config["task"],
        )
    elif "llm-" in encode_method:
        
        if no_pca==False:
            '''30-dimension PCA with or without standard scaling'''
            print(f'Current hour before prepare_llm: {datetime.now().strftime("%H:%M:%S")}')
            X_train, X_test, duration_llm = prepare_llm(
                X_train,
                X_test,
                data_name,
                encode_method,
                normalization
            )
            duration_embed += duration_llm
        else:
            '''Treating Matryoshka representation by taking the first 30 dimensions'''
            print(f'Current hour before prepare_llm_no_pca_mrl: {datetime.now().strftime("%H:%M:%S")}')
            X_train, X_test, duration_llm = prepare_llm_no_pca_mrl(
                X_train,
                X_test,
                data_name,
                encode_method,
                n_dimensions=30 # This is the number of dimensions to keep for the no-PCA version. 
            )
            duration_embed += duration_llm
    elif encode_method == "catboost":
        X_train, X_test, cat_features = prepare_catboost(X_train, X_test)
    elif encode_method == "tabpfn":
        X_train, X_test = prepare_tabpfn(X_train, X_test)
    elif encode_method == "tabstar":
        X_train, X_test = prepare_tabstar(X_train, X_test, y_train)
        # convert y_train to pandas Series if not already
        if not isinstance(y_train, pd.Series):
            y_train = pd.Series(y_train, index=X_train.index)
    elif encode_method == "tarte":
        X_train, X_test = prepare_tarte(X_train, X_test, y_train)
    elif encode_method == "contexttab":
        X_train, X_test = prepare_contexttab(X_train, X_test)

    end_time = time.perf_counter()
    duration_embed += round(end_time - start_time, 4)

    return X_train, X_test, y_train, y_test, duration_embed, cat_features


def prepare_tabvec(X_train, X_test):
    """Function to prepare with StringEncoder(TabVec)."""

    from skrub import StringEncoder, TableVectorizer, SquashingScaler
    
    # First, set the dataframe in appropriate format
    cleaner = TableVectorizer(cardinality_threshold=0, high_cardinality="passthrough")
    X_train = cleaner.fit_transform(X_train)
    X_test = cleaner.transform(X_test)

    # Encode
    text_encoder = StringEncoder(random_state=1234)
    num_transformer = SquashingScaler()
    encoder = TableVectorizer(
        high_cardinality=text_encoder,
        numeric=num_transformer
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
        high_cardinality=text_encoder,
        numeric=num_transformer,
    )

    X_train = encoder.fit_transform(X_train, y=y_train)
    X_test = encoder.transform(X_test)

    return X_train, X_test

def prepare_llm(X_train, X_test, data_name, llm_model_name, normalization):
    """Function to prepare with LLM embeddings."""

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer, StandardScaler
    from sklearn.utils.validation import check_is_fitted
    from sklearn.exceptions import NotFittedError
    from sklearn.decomposition import PCA
    from skrub import TableVectorizer, SquashingScaler, DatetimeEncoder
    import pandas as pd
    import numpy as np
    from configs.path_configs import path_configs

    # First, set the dataframe in appropriate format
    cleaner = TableVectorizer(cardinality_threshold=0, high_cardinality="passthrough")
    X_train = cleaner.fit_transform(X_train)
    X_test = cleaner.transform(X_test)

    # Load LLM embeddings
    llm_embed_path = f"{path_configs['base_path']}/data/llm_embeding/{llm_model_name}/{llm_model_name}|{data_name}.parquet"
    llm_embeddings = pd.read_parquet(llm_embed_path)

    # Initialize stateful transformers
    pca = PCA(n_components=30, random_state=1234)
    scaler = StandardScaler()

    def _replace_with_llm_embedding(column, normalization=normalization):
        column_name = column.columns.tolist()[0]
        null_mask = column[column_name].isnull().to_numpy()
        df_col = pd.DataFrame(column)
        df_col.columns = ["name"]
        df_col = df_col.merge(how="left", right=llm_embeddings, on="name").drop(
            columns="name"
        )
        df_col = np.array(df_col)

        # ========================== ADD THIS BLOCK ==========================
        # 1. Update null_mask: treat failed lookups (NaNs in embeddings) as nulls
        # If the merge failed to find a key, the row will contain NaNs
        embedding_nan_mask = np.isnan(df_col).any(axis=1)
        null_mask = null_mask | embedding_nan_mask

        # 2. Guard Clause: If there are 0 valid samples (e.g. all NaNs in test col), return early.
        # This prevents the "Found array with 0 sample(s)" error in PCA.
        if np.all(null_mask):
            df_out = np.full((len(column), 30), np.nan)
            return pd.DataFrame(df_out).add_prefix(f"{column_name}_")
        # ====================================================================

        # Check if we are in training or inference mode based on PCA state
        # This ensures we fit Scaler/PCA on train and only transform on test
        try:
            check_is_fitted(pca)
            is_fitted = True
        except NotFittedError:
            is_fitted = False

        # Apply Normalization (StandardScaler)
        if normalization:
            if np.any(~null_mask):
                if not is_fitted:
                    # Fit and transform on training data
                    df_col[~null_mask] = scaler.fit_transform(df_col[~null_mask])
                else:
                    # Only transform on test data (using stats from train)
                    df_col[~null_mask] = scaler.transform(df_col[~null_mask])

        # Apply PCA
        if not is_fitted:
            out_pca = pca.fit_transform(df_col[~null_mask])
        else:
            out_pca = pca.transform(df_col[~null_mask])

        df_out = np.zeros(shape=(df_col.shape[0], 30))
        df_out[~null_mask] = out_pca
        df_out[null_mask] = np.nan
        df_out = pd.DataFrame(df_out)
        return df_out.add_prefix(f"{column_name}_")

    # Set the pipeline for the categoricals
    fn_transformer = FunctionTransformer(_replace_with_llm_embedding)
    num_transformer = SquashingScaler()
    encoder = TableVectorizer(
        cardinality_threshold=0,
        high_cardinality=fn_transformer,
        numeric=num_transformer,
    )

    # Encode
    X_train = encoder.fit_transform(X_train)
    X_test = encoder.transform(X_test)

    # Extraction time
    time_folder = f"{path_configs['base_path']}/data/llm_embed_time/{llm_model_name}"
    time_path = f"{time_folder}/{llm_model_name}|{data_name}.npy"

    if os.path.exists(time_path):
        time_data = np.load(time_path)
    else:
        # Ensure the directory exists first
        os.makedirs(time_folder, exist_ok=True)
        
        # Create a dummy array (e.g., zeros). 
        # Match the shape to your training/test set size as needed.
        time_data = np.zeros(len(X_train) + len(X_test)) 
        
        # Save it so it exists for next time
        np.save(time_path, time_data)

    return X_train, X_test, time_data


def prepare_llm_no_pca_mrl(X_train, X_test, data_name, llm_model_name, n_dimensions):
    """Function to prepare LLMs that have matrioska representation and thus do not require PCA."""

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer
    from skrub import TableVectorizer, SquashingScaler, DatetimeEncoder
    from configs.path_configs import path_configs

    # First, set the dataframe in appropriate format
    cleaner = TableVectorizer(cardinality_threshold=0, high_cardinality="passthrough")
    X_train = cleaner.fit_transform(X_train)
    X_test = cleaner.transform(X_test)

    # Load LLM embeddings
    llm_embed_path = f"{path_configs['base_path']}/data/llm_embeding/{llm_model_name}/{llm_model_name}|{data_name}.parquet"
    llm_embeddings = pd.read_parquet(llm_embed_path)

    def _replace_with_llm_embedding(column, n_dimensions=n_dimensions):
        column_name = column.columns.tolist()[0]
        null_mask = column[column_name].isnull().to_numpy()
        df_col = pd.DataFrame(column)
        df_col.columns = ["name"]
        df_col = df_col.merge(how="left", right=llm_embeddings, on="name").drop(
            columns="name"
        )
        df_col = np.array(df_col)

        # Instead of PCA, we simply slice the first X dimensions.
        # We select valid rows first to handle potential NaNs correctly
        valid_embeddings = df_col[~null_mask]
        
        # Safety check (optional but recommended)
        if valid_embeddings.shape[1] < n_dimensions:
            raise ValueError(f"Embeddings dimension {valid_embeddings.shape[1]} is less than the required {n_dimensions}.")
        # Take the first X dimensions (Matryoshka slicing)
        out_slice = valid_embeddings[:, :n_dimensions]

        df_out = np.zeros(shape=(df_col.shape[0], n_dimensions))
        df_out[~null_mask] = out_slice
        df_out[null_mask] = np.nan

        df_out = pd.DataFrame(df_out)
        return df_out.add_prefix(f"{column_name}_")

    # Set the pipeline for the categoricals
    fn_transformer = FunctionTransformer(_replace_with_llm_embedding)
    num_transformer = SquashingScaler()
    encoder = TableVectorizer(
        cardinality_threshold=0,
        high_cardinality=fn_transformer,
        numeric=num_transformer,
    )

    # Encode
    X_train = encoder.fit_transform(X_train)
    X_test = encoder.transform(X_test)

    # Extraction time
    time_folder = f"{path_configs['base_path']}/data/llm_embed_time/{llm_model_name}"
    time_path = f"{time_folder}/{llm_model_name}|{data_name}.npy"

    if os.path.exists(time_path):
        time_data = np.load(time_path)
    else:
        # Ensure the directory exists first
        os.makedirs(time_folder, exist_ok=True)
        
        # Create a dummy array (e.g., zeros). 
        # Match the shape to your training/test set size as needed.
        time_data = np.zeros(len(X_train) + len(X_test)) 
        
        # Save it so it exists for next time
        np.save(time_path, time_data)

    return X_train, X_test, time_data


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


def prepare_tabstar(X_train, X_test, y_train):
    """Function to prepare with TabSTAR(Internal).
    X_train, X_test: pd.DataFrame
    y_train: must be a pandas series
    """

    missing_markers = ['NOT AVAILABLE', 'unknown', 'null', 'None', 'nan', 'N/A']
    X_train = X_train.replace(missing_markers, np.nan)
    X_test = X_test.replace(missing_markers, np.nan)

    from skrub import TableVectorizer

    tabvec = TableVectorizer(cardinality_threshold=0, # prevent OneHotEncoding of low-cardinality columns
                             high_cardinality="passthrough", 
                            #  low_cardinality="passthrough",  # Keep categories as strings
                             numeric="passthrough",  # Keep numbers as numbers (default)
                             datetime=DatetimeEncoder() # Decompose dates as per paper (default)
                             ) 
    X_train = tabvec.fit_transform(X_train)
    X_test = tabvec.transform(X_test)

    # categorical and boolean fields encoded as text (Appendix A.1)

    numeric_cols = tabvec.kind_to_columns_["numeric"]
    non_numeric_cols = X_train.columns.difference(numeric_cols)

    # 2. Convert all these columns to string type for both datasets
    X_train[non_numeric_cols] = X_train[non_numeric_cols].astype(str)
    X_test[non_numeric_cols] = X_test[non_numeric_cols].astype(str)

    return X_train, X_test

def prepare_tarte(X_train, X_test, y_train):
    """Function to prepare with TARTE(Internal)."""

    from skrub import TableVectorizer
    from tarte_ai import TARTE_TablePreprocessor

    tabvec = TableVectorizer(cardinality_threshold=0, # prevent OneHotEncoding of low-cardinality columns
                             high_cardinality="passthrough", 
                            #  low_cardinality="passthrough",  # Keep categories as strings
                             numeric="passthrough",  # Keep numbers as numbers (default)
                             datetime=DatetimeEncoder() # Decompose dates as per paper (default)
                             ) 
    X_train = tabvec.fit_transform(X_train)
    X_test = tabvec.transform(X_test)

    preprocessor = TARTE_TablePreprocessor()
    X_train = preprocessor.fit_transform(X_train, y_train)
    X_test = preprocessor.transform(X_test)

    return X_train, X_test


def prepare_contexttab(X_train, X_test):
    """Function to prepare with TabSTAR(Internal)."""

    from skrub import TableVectorizer

    tabvec = TableVectorizer(cardinality_threshold=0, # prevent OneHotEncoding of low-cardinality columns
                             high_cardinality="passthrough", 
                            #  low_cardinality="passthrough",  # Keep categories as strings
                             numeric="passthrough",  # Keep numbers as numbers (default)
                             datetime=DatetimeEncoder() # Decompose dates as per paper (default)
                             ) 
    X_train = tabvec.fit_transform(X_train)
    X_test = tabvec.transform(X_test)

    return X_train, X_test