"""The leaderboard's published results: the copy shipped with the package, and today's (network)."""

import hashlib
import json

import pytest

import strable
from strable.results import RESULTS_DIR

COLUMNS = ["id", "encoder", "learner", "variant", "table", "task", "fold", "score", "total_s", "n_rows",
           "time_per_1k_s", "overlap", "api"]


def test_shipped_results_cover_the_108_tables_and_their_folds():
    board = strable.load_published_results()
    assert list(board.columns) == COLUMNS
    assert set(board["table"]) == set(strable.list_tables())
    assert set(board["fold"]) == {0, 1, 2} and set(board["task"]) <= {"regression", "binary", "multiclass"}
    assert not board.duplicated(["id", "table", "fold"]).any()
    assert board.groupby("id").size().max() == 108 * 3
    manifest = json.loads((RESULTS_DIR / "manifest.json").read_text())
    assert board.attrs == {"leaderboard_version": manifest["leaderboard_version"], "built": manifest["built"]}
    assert hashlib.sha256((RESULTS_DIR / "results_per_fold.parquet").read_bytes()).hexdigest() == manifest["sha256"]


def test_shipped_results_use_the_package_tasks():
    """Each table's task in the results is the one the package's folds give it."""
    board = strable.load_published_results()
    tasks = board.drop_duplicates("table").set_index("table")["task"]
    for group in ("regression", "binary", "multiclass"):
        assert set(tasks[tasks == group].index) == set(strable.list_tables(group))


@pytest.mark.network
def test_latest_results_are_downloaded_from_the_leaderboard():
    board = strable.load_published_results(latest=True)
    assert list(board.columns) == COLUMNS and board.attrs["leaderboard_version"].startswith("v")
    assert len(board) >= len(strable.load_published_results()) * 0.9
