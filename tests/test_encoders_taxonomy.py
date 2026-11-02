"""The Tf-Idf encoder and the string taxonomy on a small table with one column of each kind."""

import numpy as np
import pandas as pd

import strable
from strable.taxonomy import TYPES

N = 200
WORDS = ["green", "valley", "river", "north", "lake", "stone", "bridge", "silver", "creek", "oak", "hill", "spring"]
SENTENCES = ["the parcel was left at the door and the customer was happy with it",
             "it arrived late but the box was in good shape and nothing was missing",
             "we had to call the support team because the order was not complete"]


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


def test_tfidf_encodes_strings_dates_and_numbers():
    out = strable.encoders.tfidf().fit_transform(_table())
    assert sum(c.startswith("token_") for c in out.columns) == 30  # high cardinality: 30 StringEncoder components
    assert [c for c in out.columns if c.startswith("colour")] == ["colour_blue", "colour_green", "colour_red"]
    assert {"day_year", "day_month", "day_day"} <= set(out.columns)
    assert out.abs().to_numpy().max() <= 3  # every number went through the SquashingScaler


def test_profile_gives_one_type_per_string_column():
    columns, mix = strable.profile(_table())
    assert columns["type"].to_dict() == {"colour": "categorical", "place": "name", "review": "free_text",
                                         "code": "structured_code", "token": "identifier", "visit": "datetime"}
    assert list(mix.index) == list(TYPES) and np.isclose(mix.sum(), 1) and (mix == 1 / 6).all()


def test_profile_without_string_columns():
    columns, mix = strable.profile(_table()[["day", "x"]])
    assert columns.empty and (mix == 0).all()
