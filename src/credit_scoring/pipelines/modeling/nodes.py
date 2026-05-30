import pandas as pd
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


def train_baseline_model(
    model_input: pd.DataFrame,
    parameters: dict,
) -> tuple[RandomForestClassifier, dict]:
    y = model_input["Credit_Score"]
    x = model_input.drop(columns=["Credit_Score"])

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=parameters["test_size"],
        random_state=parameters["random_state"],
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=parameters["n_estimators"],
        min_samples_split=parameters["min_samples_split"],
        min_samples_leaf=parameters["min_samples_leaf"],
        class_weight=parameters["class_weight"],
        random_state=parameters["random_state"],
        n_jobs=-1,
    )

    model.fit(x_train, y_train)

    predictions = model.predict(x_test)

    metrics = {
        "model": "RandomForestClassifier",
        "accuracy": float(accuracy_score(y_test, predictions)),
        "f1_macro": float(f1_score(y_test, predictions, average="macro")),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
    }

    mlflow.log_params({
        "model": "RandomForestClassifier",
        **parameters,
    })

    mlflow.log_metrics({
        "accuracy": metrics["accuracy"],
        "f1_macro": metrics["f1_macro"],
    })

    return model, metrics