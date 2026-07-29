from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def tune_random_forest(X_train, y_train):
    """
    Tune Random Forest using GridSearchCV.
    Returns:
        best_model, best_params, best_score
    """

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(random_state=42))
    ])

    param_grid = {
        "classifier__n_estimators": [100, 200, 300],
        "classifier__max_depth": [None, 10, 20],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 4]
    }

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="f1",
        verbose=1
    )

    grid.fit(X_train, y_train)

    return (
        grid.best_estimator_,
        grid.best_params_,
        grid.best_score_
    )


def tune_logistic_regression(X_train, y_train):
    """
    Tune Logistic Regression using GridSearchCV.
    Returns:
        best_model, best_params, best_score
    """

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42))
    ])

    param_grid = {
        "classifier__C": [0.01, 0.1, 1, 10, 100],
        "classifier__solver": ["liblinear", "lbfgs"]
    }

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="f1",
        verbose=1
    )

    grid.fit(X_train, y_train)

    return (
        grid.best_estimator_,
        grid.best_params_,
        grid.best_score_
    )