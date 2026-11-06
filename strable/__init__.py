"""STRABLE: a benchmark of 108 real-world tables with strings.

Load the tables, get the benchmark's cross-validation folds, score a scikit-learn pipeline with the
leaderboard's protocol, compare it with the leaderboard's published results, and see which kinds of
strings a table holds.
"""

from . import encoders
from .datasets import get_folds, list_tables, load_table
from .evaluation import evaluate
from .results import load_published_results
from .taxonomy import profile

__version__ = "0.4.0"
__all__ = ["encoders", "evaluate", "get_folds", "list_tables", "load_published_results", "load_table", "profile"]
