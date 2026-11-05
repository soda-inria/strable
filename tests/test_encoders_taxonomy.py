"""The Tf-Idf encoder and the string taxonomy on a small table with one column of each kind."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import strable
from strable import taxonomy
from strable.taxonomy import TYPES

N = 200
WORDS = ["green", "valley", "river", "north", "lake", "stone", "bridge", "silver", "creek", "oak", "hill", "spring"]
SENTENCES = ["the parcel was left at the door and the customer was happy with it",
             "it arrived late but the box was in good shape and nothing was missing",
             "we had to call the support team because the order was not complete"]
EXPECTED = {"colour": "categorical", "place": "name", "review": "free_text", "code": "structured_code",
            "token": "identifier", "visit": "datetime"}


def _table():
    rng = np.random.default_rng(0)
    days = pd.date_range("2020-01-01", periods=N).strftime("%Y-%m-%d")
    return pd.DataFrame({
        "colour": rng.choice(["red", "green", "blue"], N),
        "place": [f"{a.title()} {b.title()}" for a, b in zip(rng.choice(WORDS, N), rng.choice(WORDS, N))],
        "review": [f"{s} on day {i}" for i, s in enumerate(rng.choice(SENTENCES, N))],
        "code": [f"NDC-{i:05d}" for i in rng.integers(0, 99999, N)],
        "token": ["".join(rng.choice(list("bcdfghjklmnpqrstvwxz"), 10)) for _ in range(N)],
        "visit": [f"visit {d}" for d in days],  # a date inside text: skrub keeps it as a string
        "day": days,  # a plain date: skrub parses it, so it is not a string column
        "x": rng.normal(size=N),
    })


@pytest.fixture
def fresh_word_lists():
    """Forget the word lists loaded by other tests, before and after the test."""
    taxonomy._english_words.cache_clear(), taxonomy._stopwords.cache_clear()
    yield
    taxonomy._english_words.cache_clear(), taxonomy._stopwords.cache_clear()


@pytest.fixture
def small_word_lists(monkeypatch):
    """Small stand-ins for NLTK's corpora, so that these tests run offline."""
    words = frozenset(WORDS + " ".join(SENTENCES + ["on day", "red green blue"]).split())
    stopwords = frozenset("the was at and it but in with to we had because not on".split())
    monkeypatch.setattr(taxonomy, "_english_words", lambda: words)
    monkeypatch.setattr(taxonomy, "_stopwords", lambda: stopwords)


def test_tfidf_encodes_strings_dates_and_numbers():
    out = strable.encoders.tfidf().fit_transform(_table())
    assert sum(c.startswith("token_") for c in out.columns) == 30  # high cardinality: 30 StringEncoder components
    assert [c for c in out.columns if c.startswith("colour")] == ["colour_blue", "colour_green", "colour_red"]
    assert {"day_year", "day_month", "day_day"} <= set(out.columns)
    assert out.abs().to_numpy().max() <= 3  # every number went through the SquashingScaler


def test_profile_gives_one_type_per_string_column(small_word_lists):
    columns, mix = strable.profile(_table())
    assert columns["type"].to_dict() == EXPECTED
    assert list(mix.index) == list(TYPES) and np.isclose(mix.sum(), 1) and (mix == 1 / 6).all()


def test_profile_without_string_columns(small_word_lists):
    columns, mix = strable.profile(_table()[["day", "x"]])
    assert columns.empty and (mix == 0).all()


def test_profile_offline_without_corpora_says_how_to_get_them(monkeypatch, fresh_word_lists):
    nltk = pytest.importorskip("nltk")

    def missing(resource):
        raise LookupError(resource)

    monkeypatch.setattr(nltk.data, "find", missing)
    monkeypatch.setattr(nltk, "download", lambda *args, **kwargs: False)
    with pytest.raises(RuntimeError, match="nltk.downloader words stopwords"):
        strable.profile(_table())


def test_profile_without_nltk_says_which_extra_installs_it(monkeypatch, fresh_word_lists):
    """NLTK is optional: without it, profile() names the extra to install."""
    monkeypatch.setitem(sys.modules, "nltk", None)  # import nltk now raises ImportError
    with pytest.raises(ImportError, match=r'pip install "strable\[taxonomy\]"'):
        strable.profile(_table())


@pytest.mark.network
def test_profile_downloads_the_nltk_corpora(tmp_path):
    """A fresh process whose NLTK folder and home are empty downloads the two corpora there."""
    pytest.importorskip("nltk")
    code = "from strable import taxonomy; print(len(taxonomy._english_words()), len(taxonomy._stopwords()))"
    env = dict(os.environ, NLTK_DATA=str(tmp_path), HOME=str(tmp_path))
    out = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).parents[1], env=env, check=True,
                         capture_output=True, text=True).stdout.split()
    assert int(out[0]) == 235_886 and int(out[1]) > 100
    assert (tmp_path / "corpora" / "words.zip").exists() and (tmp_path / "corpora" / "stopwords.zip").exists()
