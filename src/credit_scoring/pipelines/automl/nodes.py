from pathlib import Path

import pandas as pd


def train_autogluon_model(model_input: pd.DataFrame) -> dict:
    from autogluon.tabular import TabularPredictor

    model_path = Path("data/06_models/autogluon")

    predictor = TabularPredictor(
        label="Credit_Score",
        path=str(model_path),
        eval_metric="f1_macro",
    ).fit(
        model_input,
        presets="medium",
        time_limit=1800,
    )

    leaderboard = predictor.leaderboard(silent=True)
    best_model = leaderboard.iloc[0]

    return {
        "best_model": str(best_model["model"]),
        "score_val": float(best_model["score_val"]),
        "eval_metric": str(best_model["eval_metric"]),
        "model_path": str(model_path),
    }
