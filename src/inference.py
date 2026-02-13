"""Function to run inference given a method."""


def run_inference(
    estimator,
    fit_with_val,
    X_train,
    y_train,
    tune_indicator,
    cv,
):
    """Function to run inference. Needed to separate out the models with validation set."""

    # Final fit and predict
    if (tune_indicator == "default") and fit_with_val:
        split_index = list(cv.split(X_train, y_train))[0]
        X_train_, X_valid = X_train[split_index[0]], X_train[split_index[1]]
        y_train_, y_valid = y_train[split_index[0]], y_train[split_index[1]]
        eval_set = [(X_valid, y_valid)]
        estimator.fit(X_train_, y_train_, eval_set=eval_set, verbose=False)
    else:
        estimator.fit(X_train, y_train)

    return estimator
