"""The kind of strings each column of a table holds, with the STRABLE paper's taxonomy.

Six types: categorical, name, structured_code, free_text, identifier, datetime. Each string column
gets a few metrics, computed on its first 1,000 non-missing cells (uniqueness and the datetime share
use every non-missing cell), and fixed rules on these metrics give its type. The metrics and rules
are those of ``classify_column_heuristic`` in
scripts/natural_language_test_VSE_vs_STRABLE_vs_CARTE_vs_TTB.py of the STRABLE repository.

The word lists are NLTK's corpora, as in the paper: ``words`` (its English file ``en``, the Unix
"words" file) for the dictionary hit rate, and ``stopwords`` (English). The first call downloads
them with ``nltk.download`` when they are missing.
"""

from __future__ import annotations

import re
from functools import cache
from typing import NamedTuple

import pandas as pd

TYPES = ("categorical", "name", "structured_code", "free_text", "identifier", "datetime")
METRICS = ("n_unique", "uniqueness", "avg_chars_per_cell", "avg_words_per_cell", "dict_hit_rate", "stopword_density",
           "symbol_density", "prop_numeric_chars", "prop_multiword", "prop_has_pattern", "prop_is_datetime")
SAMPLE = 1000

# A cell is a date when it matches any of these: 2024-03-28, 2024-03-28T14:30, 28/03/2024,
# 14:30:00 or 2:30 PM, 28 March 2024, March 28, 2024.
DATETIME = re.compile("|".join([
    r"\d{4}-\d{2}-\d{2}",
    r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}",
    r"\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}",
    r"\d{2}:\d{2}(:\d{2})?(\s?[APap][Mm])?",
    r"\d{1,2}\s+\w{3,9}\s+\d{4}",
    r"\w{3,9}\s+\d{1,2},?\s+\d{4}",
]))
# A cell holds a code when it contains an ISO date, a US zip code, an amount in dollars, a
# percentage, an ICD-style code (J45.5) or a letters-dash-digits code (NDC-12345).
CODE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{5}(-\d{4})?\b|\$[\d,]+(\.\d+)?|\b\d+(\.\d+)?%"
                  r"|\b[A-Z]\d{2}(\.\d+)?\b|\b[A-Z]{2,}-\d+\b")
NOT_LETTERS = re.compile(r"[^a-zA-Z]+")
SYMBOLS = re.compile(r"[^a-zA-Z0-9\s]")
WHITESPACE = re.compile(r"\s+")


class Profile(NamedTuple):
    """``columns``: one row per string column, its type and metrics. ``mix``: the share of each type."""

    columns: pd.DataFrame
    mix: pd.Series


def _nltk_corpus(name: str):
    """NLTK's corpus ``name``, downloaded quietly the first time it is missing."""
    import nltk  # imported here: only profile() needs it

    try:
        nltk.data.find(f"corpora/{name}")
    except LookupError:
        if not nltk.download(name, quiet=True):
            raise RuntimeError(
                f"strable.profile needs NLTK's {name!r} corpus and could not download it (offline?). "
                "Download it once while online with `python -m nltk.downloader words stopwords`, "
                "or set NLTK_DATA to a folder that holds both corpora.") from None
    return getattr(nltk.corpus, name)


@cache
def _english_words() -> frozenset:
    return frozenset(_nltk_corpus("words").words("en"))


@cache
def _stopwords() -> frozenset:
    return frozenset(_nltk_corpus("stopwords").words("english"))


def string_columns(df: pd.DataFrame) -> list[str]:
    """The columns skrub's TableVectorizer keeps as text: not numbers, not parsed dates, not all missing."""
    from skrub import TableVectorizer  # imported here: skrub is slow to import

    tv = TableVectorizer(cardinality_threshold=0, low_cardinality="passthrough", high_cardinality="passthrough",
                         numeric="passthrough", datetime="passthrough")
    tv.fit(df)
    return [col for col, kind in tv.column_to_kind_.items() if kind == "high_cardinality"]


def _share(tokens: list[str], words: frozenset) -> float:
    return sum(t in words for t in tokens) / len(tokens) if tokens else 0.0


def _digit_share(text: str) -> float:
    return sum(c.isdigit() for c in text) / len(text) if text else 0.0


def column_metrics(column: pd.Series) -> dict[str, float]:
    """The metrics of one string column, as in the paper."""
    cells = column.dropna().astype(str)
    sample = cells.iloc[:SAMPLE]
    text = " ".join(sample.tolist())
    n_unique = cells.nunique()
    return {
        "n_unique": n_unique,
        "uniqueness": n_unique / len(cells) if len(cells) else 0.0,
        "avg_chars_per_cell": sample.str.len().mean(),
        "avg_words_per_cell": sample.map(lambda s: len(WHITESPACE.split(s.strip()))).mean(),
        # as in the paper: lowercase then split for the dictionary, split then lowercase for stopwords
        "dict_hit_rate": _share([t for t in NOT_LETTERS.split(text.lower()) if t], _english_words()),
        "stopword_density": _share([t.lower() for t in NOT_LETTERS.split(text) if t], _stopwords()),
        "symbol_density": len(SYMBOLS.findall(text)) / len(text) if text else 0.0,
        "prop_numeric_chars": _digit_share("".join(sample.tolist())),
        "prop_multiword": sample.str.contains(r"\s", regex=True).mean(),
        "prop_has_pattern": sample.map(lambda s: CODE.search(s) is not None).mean(),
        "prop_is_datetime": cells.map(lambda s: DATETIME.search(s) is not None).mean() if len(cells) else 0.0,
    }


def classify(m: dict[str, float]) -> str:
    """The type of a column from its metrics. The rules run in order, most specific first."""
    if m["prop_is_datetime"] >= 0.7:
        return "datetime"
    # Patterns, digits or symbols dominate; columns with very few distinct values stay categorical.
    structured = (m["prop_has_pattern"] >= 0.2 or m["prop_numeric_chars"] >= 0.25 or m["symbol_density"] >= 0.15
                  or (m["symbol_density"] >= 0.1 and m["dict_hit_rate"] < 0.3))
    if structured and m["uniqueness"] >= 0.02:
        return "structured_code"
    # Near-unique, short, not words.
    if m["uniqueness"] >= 0.8 and m["dict_hit_rate"] < 0.3 and m["avg_words_per_cell"] < 3:
        return "identifier"
    # Several words per cell, real words and stopwords: prose rather than multi-word names.
    if (m["avg_words_per_cell"] >= 4 and m["dict_hit_rate"] >= 0.4 and m["prop_multiword"] >= 0.8
            and m["stopword_density"] >= 0.1):
        return "free_text"
    # About two real words per cell (a person or a place, even when repeated), or fairly unique short words.
    if (m["dict_hit_rate"] >= 0.5 and 1.5 <= m["avg_words_per_cell"] <= 4 and m["prop_multiword"] >= 0.5
            and m["prop_numeric_chars"] < 0.1):
        return "name"
    if m["uniqueness"] >= 0.3 and m["dict_hit_rate"] >= 0.3 and m["avg_words_per_cell"] < 4:
        return "name"
    return "categorical"


def profile(df: pd.DataFrame) -> Profile:
    """The type and metrics of each string column of ``df``, and the table's mix of types.

    String columns are those skrub's TableVectorizer keeps as text; columns it reads as numbers or
    dates are left out. Returns ``Profile(columns, mix)``: ``columns`` has one row per string column
    (index: the column name) with its ``type`` and metrics; ``mix`` is the share of each of the six
    types among the string columns (all 0 when there is none).
    """
    names = string_columns(df)
    metrics = [column_metrics(df[name]) for name in names]
    columns = pd.DataFrame(metrics, index=pd.Index(names, name="column"), columns=list(METRICS))
    columns.insert(0, "type", [classify(m) for m in metrics])
    mix = columns["type"].value_counts(normalize=True).reindex(list(TYPES), fill_value=0.0).rename("share")
    return Profile(columns, mix)
