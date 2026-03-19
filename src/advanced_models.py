from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV


def train_xgboost(X_train, y_train):

    model = XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss"
    )

    params = {
        "n_estimators": [100, 200],
        "max_depth": [3, 6, 10],
        "learning_rate": [0.01, 0.1],
        "subsample": [0.8, 1]
    }

    grid = GridSearchCV(
        model,
        params,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    return grid.best_estimator_