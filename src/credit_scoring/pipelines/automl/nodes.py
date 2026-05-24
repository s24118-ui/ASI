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
    dynamic_stacking=False,
    num_bag_folds=0,
    num_stack_levels=0,
    )


    leaderboard = predictor.leaderboard(silent=True)
    best_model = leaderboard.iloc[0]

    metrics = {
        "best_model": str(best_model["model"]),
        "score_val": float(best_model["score_val"]),
        "eval_metric": str(best_model["eval_metric"]),
        "model_path": str(model_path),
        "trained_models_count": int(len(leaderboard)),
    }

    return metrics, leaderboard