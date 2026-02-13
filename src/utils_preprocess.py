"""Common functions used for data preprocessing."""

import pandas as pd
import numpy as np
import re


def clean_backslash_operations(data):
    """Function to clean for backslash operations in dataframe."""

    # clean for newline, tab regex, duouble spaces
    data = data.replace("\n", " ", regex=True)
    data = data.replace("\t", " ", regex=True)
    data = data.replace("\r", "", regex=True)
    data = data.replace("  ", " ", regex=True)
    data.columns = data.columns.str.replace("\n", " ").tolist()
    data.columns = data.columns.str.replace("\t", " ").tolist()
    data.columns = data.columns.str.replace("\r", "").tolist()
    data.columns = data.columns.str.replace("  ", " ").tolist()

    return data


def clean_list_type_col(data, col):
    """Function to clean up for the given columns stored in list."""

    data[col] = data[col].astype(str)
    data[col] = data[col].str.replace("[", "").str.replace("]", "").str.replace("'", "")
    data.loc[data[col] == "nan", col] = np.nan

    return data


def clean_dict_type_col(data, col, clean_type="to_string"):
    """Function to clean up for the given columns stored in dictionary."""

    if clean_type == "to_string":
        data[col] = data[col].astype(str)
        data[col] = (
            data[col].str.replace("{", "").str.replace("}", "").str.replace("'", "")
        )
        data.loc[data[col] == "nan", col] = np.nan
    elif clean_type == "unpack":
        loc = data.columns.get_loc(col)
        unpack_df = pd.DataFrame.from_dict(data[col].to_dict()).transpose()
        for col_name in unpack_df.columns:
            data.insert(loc, column=col_name, value=unpack_df[col_name])
            loc += 1
        data.drop(columns=col, inplace=True)
    elif clean_type == "drop":
        data.drop(columns=col, inplace=True)

    return data


def clean_constant_null(data, proportion_null=1.0):
    """Drop columns with constant values or high fraction of missing values."""

    # Extract info.
    col_check = [
        (col, data[col].nunique(), data[col].isnull().sum()) for col in data.columns
    ]
    # Drop for constant columns
    col_unique = [x[0] for x in col_check if x[1] == 1]
    data.drop(columns=col_unique, inplace=True)
    # Drop for constant columns
    null_drop_threshold = int(data.shape[0] * proportion_null)
    col_null = [x[0] for x in col_check if x[2] >= null_drop_threshold]
    data.drop(columns=col_null, inplace=True)

    return data


def clean_personal_info(data):
    """Drop columns with emails, phone numbers or urls."""
    df_cat = data.select_dtypes(include="object").copy()
    drop_col = df_cat.columns[df_cat.apply(contains_pii) == True].tolist()
    return data.drop(columns=drop_col)


def contains_pii(text) -> bool:
    """Test whether a string contains PII (email number of phone numbers)."""
    if not isinstance(text, str):
        text = _cast_to_str(text)
    # Regular expression pattern that matches most email addresses, phone numbers, urls
    pattern = re.compile(
        r"(\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b)|"  # Email pattern
        # r'((?<!\d)(?:\(\d{1,4}\)\s?|\d{1,4}[-.\s])\d{1,4}[-.\s]\d{1,4}(?!\d))|' # Phone numbers
        r"https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)",  # urls
        re.IGNORECASE,
    )
    # Search text for matches
    match = pattern.search(text)
    # Return True if a match is found, False otherwise
    return bool(match)


def _cast_to_str(s) -> str:
    if isinstance(s, str):
        return s
    if isinstance(s, bytes):
        return _convert_bytes_to_string(s)
    else:
        return str(s)


def _convert_bytes_to_string(byte_sequence) -> str:
    # List of encodings to try
    encodings = ["utf-8", "ascii", "iso-8859-1", "utf-16", "utf-32"]

    # Iterate through each encoding
    for encoding in encodings:
        try:
            # Attempt to decode the byte sequence
            decoded_string = byte_sequence.decode(encoding)
            # If successful, return the string and encoding used
            return decoded_string
        except UnicodeDecodeError:
            # If decoding fails, continue to the next encoding
            continue


def extract_cramer_v(data, target_name):
    """Function to extract Cramer's V."""

    from skrub import column_associations
    from sklearn.preprocessing import LabelEncoder

    data_copy = data.copy()
    if (data_copy[target_name].dtype == "object") | (
        data_copy[target_name].dtype == "category"
    ):
        data_copy[target_name] = LabelEncoder().fit_transform(data_copy[target_name])

    def _extract_cramer_value(data, target_name, col_name, cat_cols):
        """Function to extract Cramer's V for each cols."""
        target_ = data[target_name].copy()
        if col_name in cat_cols:
            num_bin = min(data[col_name].nunique(), target_.nunique())
            target_ = pd.qcut(target_, num_bin, labels=False, duplicates="drop")
        data_ = data[[col_name]].copy()
        data_[target_name] = target_
        return column_associations(data_)

    # Select numerical columns
    num_cols = data_copy.select_dtypes(exclude="object").columns.tolist()
    if target_name in num_cols:
        num_cols.remove(target_name)

    # Select non-numerical columns
    cat_cols = data_copy.select_dtypes(include="object").columns.tolist()
    if target_name in cat_cols:
        cat_cols.remove(target_name)

    df_ass = pd.concat(
        [
            _extract_cramer_value(data_copy, target_name, col_name, cat_cols)
            for col_name in num_cols + cat_cols
        ]
    )
    df_ass.reset_index(drop=True, inplace=True)
    df_ass.drop(columns=["left_column_idx", "right_column_idx"], inplace=True)

    return df_ass


def col_names_per_type(data, target_name):
    """Extract column names per type."""

    from skrub import to_datetime

    # Preprocess for Datetime information
    data_ = data.drop(columns=target_name)
    dat_col_names = []
    for col in data_:
        if pd.api.types.is_datetime64_any_dtype(to_datetime(data_[col])):
            dat_col_names.append(col)
    # Use original column names without lowercasing to avoid mismatches
    cat_col_names_ = data_.select_dtypes(include="object").columns.str.replace(
        "\n", " ", regex=True
    )
    cat_col_names = list(set(cat_col_names_) - set(dat_col_names))
    num_col_names_ = data_.select_dtypes(exclude="object").columns.str.replace(
        "\n", " ", regex=True
    )
    num_col_names = list(set(num_col_names_) - set(dat_col_names))
    return num_col_names, cat_col_names, dat_col_names
