from src.tune import tune_random_forest, tune_logistic_regression
import joblib

def train_model(X_train, y_train):

    trained_models = {}

    rf_model, rf_params, rf_score = tune_random_forest(
        X_train,
        y_train
    )

    lr_model, lr_params, lr_score = tune_logistic_regression(
        X_train,
        y_train
    )

    trained_models["random_forest"] = rf_model
    trained_models["logistic_regression"] = lr_model

    print("RF Params:", rf_params)
    print("LR Params:", lr_params)

    return trained_models


def save_model(model, path="models/model.pkl"):
    joblib.dump(model, path)