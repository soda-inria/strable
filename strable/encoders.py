"""The paper's Tf-Idf string encoder."""

from __future__ import annotations

from sklearn.pipeline import make_pipeline

SEED = 1234
N_COMPONENTS = 30


def tfidf(random_state: int = SEED):
    """The paper's Tf-Idf encoder, as on the STRABLE leaderboard: a table in, numbers out.

    A first TableVectorizer, the cleaner, parses the table (dates become numbers) and leaves the
    strings as they are. A second one encodes each string column with at least 40 distinct values
    into 30 numbers with skrub's StringEncoder (Tf-Idf of character 3-4-grams, then a truncated
    SVD), one-hot encodes the other string columns, and passes every number, dates included,
    through a SquashingScaler.
    """
    from skrub import SquashingScaler, StringEncoder, TableVectorizer  # imported here: skrub is slow to import

    cleaner = TableVectorizer(cardinality_threshold=0, high_cardinality="passthrough")
    encoder = TableVectorizer(high_cardinality=StringEncoder(n_components=N_COMPONENTS, random_state=random_state),
                              numeric=SquashingScaler())
    return make_pipeline(cleaner, encoder)
