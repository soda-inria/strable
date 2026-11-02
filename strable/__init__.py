"""STRABLE: a benchmark of 108 real-world tables with strings.

Load the tables, get the benchmark's cross-validation folds, score a scikit-learn pipeline with the
leaderboard's protocol, and see which kinds of strings a table holds.
"""

from . import encoders
from .datasets import get_folds, list_tables, load_table
from .evaluation import evaluate
from .taxonomy import profile

__version__ = "0.1.0"
__all__ = ["encoders", "evaluate", "get_folds", "list_tables", "load_table", "profile"]
