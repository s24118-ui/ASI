import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


def train_baseline_model(model_input: pd.DataFrame) -> tuple[RandomForestClassifier, dict]:
    y = model_input["Credit_Score"]
    x = model_input.drop(columns=["Credit_Score"])

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=300,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
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

    return model, metrics