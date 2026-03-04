"""
Parameter distributions for hyperparameter optimization
"""

import numpy as np
import copy

from typing import List
from scipy.stats import loguniform, randint, uniform, norm, multinomial
from scipy.stats._distn_infrastructure import rv_continuous_frozen, rv_discrete_frozen


class loguniform_int:
    """Integer valued version of the log-uniform distribution"""

    def __init__(self, a, b):
        self._distribution = loguniform(a, b)

    def rvs(self, *args, **kwargs):
        """Random variable sample"""
        return self._distribution.rvs(*args, **kwargs).astype(int)


class norm_int:
    """Integer valued version of the normal distribution"""

    def __init__(self, a, b):
        self._distribution = norm(a, b)

    def rvs(self, *args, **kwargs):
        """Random variable sample"""
        if self._distribution.rvs(*args, **kwargs).astype(int) < 1:
            return 1
        else:
            return self._distribution.rvs(*args, **kwargs).astype(int)


class choice:
    """chooses based on multinomial distribution"""

    def __init__(self, options, prob_distribution):
        self._distribution = multinomial(1, prob_distribution)
        try:
            self.options = np.array(options)
        except:
            self.options = options

    def rvs(self, *args, **kwargs):
        """Random variable sample"""
        output = self._distribution.rvs(*args, **kwargs).astype(bool)[0]
        if isinstance(self.options, List):
            idx = output.nonzero()[0].item()
            return self.options[idx]
        else:
            if isinstance(self.options[output].item(), rv_continuous_frozen) | (
                isinstance(self.options[output].item(), rv_discrete_frozen)
            ):
                return self.options[output].item().rvs().item()
            else:
                return self.options[output].item()


param_distributions_total = dict()

# ridge regression
param_distributions = dict()
param_distributions_total["ridge"] = param_distributions

# logistic regression
param_distributions = dict()
param_distributions_total["logistic"] = param_distributions

# tabpfn
param_distributions = dict()
param_distributions_total["tabpfn"] = param_distributions

# tabstar
param_distributions = dict()
param_distributions_total["tabstar"] = param_distributions

# contexttab
param_distributions = dict()
param_distributions_total["contexttab"] = param_distributions

# xgb
param_distributions = dict()
# param_distributions["n_estimators"] = randint(50, 1001)
param_distributions["max_depth"] = randint(2, 7)
param_distributions["min_child_weight"] = loguniform(1, 100)
param_distributions["subsample"] = uniform(0.5, 1 - 0.5)
param_distributions["learning_rate"] = loguniform(1e-5, 1)
param_distributions["colsample_bylevel"] = uniform(0.5, 1 - 0.5)
param_distributions["colsample_bytree"] = uniform(0.5, 1 - 0.5)
param_distributions["gamma"] = loguniform(1e-8, 7)
param_distributions["reg_lambda"] = loguniform(1, 4)
param_distributions["alpha"] = loguniform(1e-8, 100)
param_distributions_total["xgb"] = param_distributions

# catboost
param_distributions = dict()
# param_distributions["iterations"] = randint(50, 1001)
param_distributions["max_depth"] = randint(2, 7)
param_distributions["learning_rate"] = loguniform(1e-5, 1)
param_distributions["bagging_temperature"] = uniform(0, 1)
param_distributions["l2_leaf_reg"] = loguniform(1, 10)
param_distributions["random_strength"] = randint(1, 21)
param_distributions["one_hot_max_size"] = randint(0, 26)
param_distributions["leaf_estimation_iterations"] = randint(1, 21)
param_distributions_total["catboost"] = param_distributions

# histgb
param_distributions = dict()
param_distributions["learning_rate"] = loguniform(1e-2, 10)
param_distributions["max_depth"] = [None, 2, 3, 4]
param_distributions["max_leaf_nodes"] = norm_int(31, 5)
param_distributions["min_samples_leaf"] = norm_int(20, 2)
param_distributions["l2_regularization"] = loguniform(1e-6, 1e3)
param_distributions_total["histgb"] = param_distributions

# RandomForest
param_distributions = dict()
param_distributions["max_depth"] = choice([None, 2, 3, 4], [0.7, 0.1, 0.1, 0.1])
param_distributions["max_features"] = [
    "sqrt",
    "log2",
    None,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
]
param_distributions["min_samples_split"] = choice([2, 3], [0.95, 0.05])
param_distributions["min_samples_leaf"] = loguniform_int(1.5, 50.5)
param_distributions["bootstrap"] = [True, False]
param_distributions["min_impurity_decrease"] = choice(
    [0, 0.01, 0.02, 0.05], [0.85, 0.05, 0.05, 0.05]
)
param_distributions_total["randomforest"] = param_distributions

# ExtraTrees
param_distributions = dict()
param_distributions["max_features"] = ["sqrt", 0.5, 0.75, 1.0]
param_distributions["min_samples_split"] = loguniform_int(2, 32)
param_distributions["min_impurity_decrease"] = choice(
    [0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3], [0.5, 0.1, 0.1, 0.1, 0.1, 0.1]
)
param_distributions_total["extrees"] = param_distributions

# tarte-nn
param_distributions = dict()
lr_grid = [1e-4, 5e-4, 1e-3]
param_distributions["learning_rate"] = lr_grid
param_distributions["num_layers"] = [1, 3]
param_distributions_total["tarte"] = param_distributions

# carte-gnn
param_distributions = dict()
lr_grid = [1e-4, 2.5e-4, 5e-4, 7.5e-4, 1e-3]
param_distributions["learning_rate"] = lr_grid
param_distributions_total["gnn"] = param_distributions

# resnet
param_distributions = dict()
param_distributions["normalization"] = ["batchnorm", "layernorm"]
param_distributions["num_layers"] = randint(1, 9)
param_distributions["hidden_dim"] = randint(32, 513)
param_distributions["hidden_factor"] = randint(1, 3)
param_distributions["hidden_dropout_prob"] = uniform(0.0, 0.5)
param_distributions["residual_dropout_prob"] = uniform(0.0, 0.5)
param_distributions["learning_rate"] = loguniform(1e-5, 1e-2)
param_distributions["weight_decay"] = loguniform(1e-8, 1e-2)
param_distributions["batch_size"] = [16, 32]
param_distributions_total["resnet"] = param_distributions

# mlp
param_distributions = dict()
param_distributions["hidden_dim"] = [2**x for x in range(4, 9)]
param_distributions["num_layers"] = randint(1, 4)
param_distributions["dropout"] = uniform(0.0, 0.5)
param_distributions["learning_rate"] = loguniform(1e-5, 1e-2)
param_distributions["weight_decay"] = loguniform(1e-8, 1e-2)
param_distributions["batch_size"] = [16, 32]
param_distributions_total["mlp"] = param_distributions

# realmlp
param_distributions = dict()
param_distributions["num_emb_type"] = ["none", "pbld", "pl", "plr"]
param_distributions["add_front_scale"] = choice([True, False], [0.6, 0.4])
param_distributions["lr"] = loguniform(2e-2, 3e-1)
param_distributions["p_drop"] = choice([0.0, 0.15, 0.3], [0.3, 0.5, 0.2])
param_distributions["wd"] = [0.0, 2e-2]
param_distributions["plr_sigma"] = loguniform(0.05, 0.5)
param_distributions["hidden_sizes"] = choice(
    [[256] * 3, [64] * 5, [512]], [0.6, 0.2, 0.2]
)
param_distributions["act"] = ["selu", "mish", "relu"]
param_distributions["ls_eps"] = choice([0.0, 0.1], [0.3, 0.7])
param_distributions_total["realmlp"] = param_distributions

# tabm
param_distributions = dict()
param_distributions["n_blocks"] = randint(1, 5)
param_distributions["d_block"] = randint(64, 1025)
param_distributions["dropout"] = uniform(0.0, 0.5)
param_distributions["num_emb_n_bins"] = randint(8, 33)
param_distributions["lr"] = loguniform(5e-5, 3e-3)
param_distributions["weight_decay"] = choice([0, loguniform(1e-4, 1e-1)], [0.5, 0.5])
param_distributions_total["tabm"] = param_distributions
