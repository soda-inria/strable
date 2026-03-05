"""LLM Encoder map the extracted(cached) embeddings."""

import warnings
import numbers
import pandas as pd
import numpy as np

from pathlib import Path
from typing import Union
from skrub import _dataframe as sbd
from skrub._apply_to_cols import SingleColumnTransformer
from skrub._scaling_factor import scaling_factor
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted


class LLM_Encoder(SingleColumnTransformer):
    """Encode string features with pre-stored embeddings from a pretrained language model.

    This applies a simple merge on pre-stored embeddings and
    follows the scikit-learn API, making it usable within a scikit-learn pipeline.

    Parameters
    ----------
    n_components : int or None, default=30,
        The number of embedding dimensions. As the number of dimensions is different
        across embedding models, this class uses a :class:`~sklearn.decomposition.PCA`
        to set the number of embedding to ``n_components`` during ``transform``.
        Set ``n_components=None`` to skip the PCA dimension reduction mechanism.

        See [1]_ for more details on the choice of the PCA and default
        ``n_components``.

    cached_llm_embedding_path : str, default=None
        Path to stored LLM embeddings.

    random_state : int, RandomState instance or None, default=None
        Used when the PCA dimension reduction mechanism is used, for reproducible
        results across multiple function calls.

    Attributes
    ----------

    See Also
    --------

    Notes
    -----

    References
    ----------
    .. [1]  L. Grinsztajn, M. Kim, E. Oyallon, G. Varoquaux
            "Vectorizing string entries for data processing on tables: when are larger
            language models better?", 2023.
            https://hal.science/hal-04345931

    """

    def __init__(
        self,
        *,
        cached_llm_embedding_path: Union[str, None] = None,
        n_components: Union[int, None] = 30,
        random_state: Union[int, None] = None,
        normalization: bool = False
    ):

        self.cached_llm_embedding_path = cached_llm_embedding_path
        self.n_components = n_components
        self.random_state = random_state
        self.normalization = normalization

    def fit_transform(self, column, y=None):
        """Fit the TextEncoder from ``column``.

        In practice, it loads the pre-trained model from disk and returns
        the embeddings of the column.

        Parameters
        ----------
        column : pandas or polars Series of shape (n_samples,)
            The string column to compute embeddings from.

        y : None
            Unused. Here for compatibility with scikit-learn.

        Returns
        -------
        X_out : pandas or polars DataFrame of shape (n_samples, n_components)
            The embedding representation of the input.
        """
        del y

        self._check_params()

        # Load LLM embeddings
        self.llm_embeddings_ = pd.read_parquet(self.cached_llm_embedding_path)

        # Set preliminaries
        mask_null = column.isnull().to_numpy()

        # Apply vectorize (merge)
        X_out_, column_name = self._vectorize(column)

        # Apply PCA
        if self.n_components is not None:
            X_out = np.zeros(shape=(X_out_.shape[0], self.n_components))
            if ((~mask_null).sum()) >= self.n_components:
                imputer_ = SimpleImputer(copy=False)
                pca_ = PCA(
                    n_components=self.n_components,
                    copy=False,
                    random_state=self.random_state,
                )
                steps = [("impute", imputer_)]
                if self.normalization:
                    from sklearn.preprocessing import StandardScaler
                    steps.append(("scaler", StandardScaler()))
                steps.append(("pca", pca_))
                # self.pipeline_ = Pipeline([("impute", imputer_), ("pca", pca_)])
                self.pipeline_ = Pipeline(steps)
                X_out_ = self.pipeline_.fit_transform(X_out_[~mask_null])
                self.scaling_factor_ = scaling_factor(X_out)
            else:
                warnings.warn(
                    f"The number of non-null entries shape is {(~mask_null).sum().item()}, "
                    f"which is too small to fit a PCA with "
                    f"n_components={self.n_components}. "
                    "The embeddings will be truncated by keeping the first "
                    f"{self.n_components} dimensions instead. "
                    "Set n_components=None to keep all dimensions and remove "
                    "this warning."
                )
                # self.n_components can be greater than the number
                # of dimensions of X_out.
                # Therefore, self.n_components_ below stores the resulting
                # number of dimensions of X_out.
                X_out_ = X_out_[~mask_null, : self.n_components]
                self.scaling_factor_ = scaling_factor(X_out)

        X_out[~mask_null] = X_out_
        X_out[mask_null] = np.nan

        # block normalize
        # self.scaling_factor_ = scaling_factor(X_out)
        # X_out /= self.scaling_factor_

        # Transform back to pandas df
        X_out = pd.DataFrame(X_out)

        return X_out.add_prefix(f"{column_name}_")

    def transform(self, column):
        """Transform ``column`` using the TextEncoder.

        This method uses the embedding model loaded in memory during ``fit``
        or ``fit_transform``.

        Parameters
        ----------
        column : pandas or polars Series of shape (n_samples,)
            The string column to compute embeddings from.

        Returns
        -------
        X_out : pandas or polars DataFrame of shape (n_samples, n_components)
            The embedding representation of the input.
        """
        check_is_fitted(self)

        # Error checking at fit time is done by the ToStr transformer,
        # but after ToStr is fitted it does not check the input type anymore,
        # while we want to ensure that the input column is a string or categorical
        # so we need to add the check here.
        if not (sbd.is_string(column) or sbd.is_categorical(column)):
            raise ValueError("Input column does not contain strings.")

        # Set preliminaries
        mask_null = column.isnull().to_numpy()

        # Apply vectorize (merge)
        X_out_, column_name = self._vectorize(column)
        X_out = np.zeros(shape=(X_out_.shape[0], self.n_components))

        if hasattr(self, "pipeline_"):
            X_out_ = self.pipeline_.transform(X_out_[~mask_null])
        elif self.n_components is not None:
            X_out_ = X_out_[~mask_null, : self.n_components]

        X_out[~mask_null] = X_out_
        X_out[mask_null] = np.nan

        # block scale
        X_out /= self.scaling_factor_

        # Transform back to pandas df
        X_out = pd.DataFrame(X_out)

        return X_out.add_prefix(f"{column_name}_")

    def _vectorize(self, column):
        column = pd.DataFrame(column)
        column_name = column.columns.tolist()[0]
        column.columns = ["name"]
        column = column.merge(how="left", right=self.llm_embeddings_, on="name").drop(
            columns="name"
        )
        return np.array(column), column_name

    def _check_params(self):
        # XXX: Use sklearn _parameter_constraints instead?
        if self.n_components is not None and not isinstance(
            self.n_components, numbers.Integral
        ):
            raise ValueError(
                f"Got n_components={self.n_components!r} but expected an integer "
                "or None."
            )

        if self.cached_llm_embedding_path is not None and not isinstance(
            self.cached_llm_embedding_path, (str, bytes, Path)
        ):
            raise ValueError(
                f"Got cached_llm_embedding_path={self.cached_llm_embedding_path} but expected a "
                "str, bytes or Path type."
            )

        return

    def __getstate__(self):
        state = self.__dict__.copy()

        # Remove llm embeddings as they are called from the cached file during fit
        remove_props = ["llm_embeddings"]
        for prop in remove_props:
            if prop in state:
                del state[prop]

        return state