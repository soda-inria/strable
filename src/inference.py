"""Final fit on the training set before scoring on test."""

from sklearn.ensemble import VotingClassifier, VotingRegressor


def run_inference(
    best_estimator,
    fit_with_val,
    X_train,
    y_train,
    tune_indicator,
    cv,
):
    """Fit ``best_estimator`` and return it ready to predict."""

    is_voting_ensemble = isinstance(
        best_estimator, (VotingRegressor, VotingClassifier)
    )

    # Voting ensembles are already fitted (each leaf is a FrozenEstimator).
    # Calling .fit on them is a no-op but satisfies sklearn's contract that
    # an estimator must be fitted before .predict.
    if is_voting_ensemble:
        best_estimator.fit(X_train, y_train)
        return best_estimator

    # Default mode + needs-eval-set: pick the first fold to provide a
    # validation set for early stopping.
    if tune_indicator == "default" and fit_with_val:
        train_idx, valid_idx = next(iter(cv.split(X_train, y_train)))
        X_train_ = X_train[train_idx] if not hasattr(X_train, "iloc") else X_train.iloc[train_idx]
        X_valid = X_train[valid_idx] if not hasattr(X_train, "iloc") else X_train.iloc[valid_idx]
        y_train_, y_valid = y_train[train_idx], y_train[valid_idx]
        eval_set = [(X_valid, y_valid)]
        best_estimator.fit(X_train_, y_train_, eval_set=eval_set, verbose=False)
        return best_estimator

    # Everything else: fit on the full training set.
    best_estimator.fit(X_train, y_train)
    return best_estimator
