"""Function to run inference given a method."""

from src.utils_evaluation import assign_estimator


def run_inference(
    X_train,
    y_train,
    task,
    estim_method,
    cv,
    device,
    best_params,
    best_split_idx,
    cat_features=None,
):
    """Function to run inference. Needed to separate out the models with validation set."""

    # Preliminary settings
    gbdt_estimators_with_val = ["xgb", "catboost"]

    # Set the estimator
    estimator = assign_estimator(
        estim_method,
        task,
        device,
        best_params=best_params,
        cat_features=cat_features,
    )

    if estim_method=='tarte' and len(X_train) > 1000:
        # Tarte specific setting for large datasets
        print(f"Setting TARTE batch_size to 256 in inference for large dataset: {len(X_train)}")
        estimator.set_params(batch_size=256)

    # Final fit and predict
    if estim_method in gbdt_estimators_with_val:
        split_index = list(cv.split(X_train, y_train))[best_split_idx]
        X_train_, X_valid = X_train[split_index[0]], X_train[split_index[1]]
        y_train_, y_valid = y_train[split_index[0]], y_train[split_index[1]]
        eval_set = [(X_valid, y_valid)]
        estimator.fit(X_train_, y_train_, eval_set=eval_set, verbose=False)
    else:
        estimator.fit(X_train, y_train)

    return estimator
