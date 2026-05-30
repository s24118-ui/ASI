from pathlib import Path

import pandas as pd
import mlflow

def train_autogluon_model(model_input: pd.DataFrame, parameters: dict) -> dict:
    from autogluon.tabular import TabularPredictor

    model_path = Path(parameters["model_path"])

    predictor = TabularPredictor(
        label=parameters["label"],
        path=str(model_path),
        eval_metric=parameters["eval_metric"],
    ).fit(
        model_input,
        presets=parameters["presets"],
        dynamic_stacking=parameters["dynamic_stacking"],
        num_bag_folds=parameters["num_bag_folds"],
        num_stack_levels=parameters["num_stack_levels"],
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

    mlflow.log_params({
        "model": "AutoGluon TabularPredictor",
        "best_model": metrics["best_model"],
        **parameters,
    })

    mlflow.log_metrics({
        "score_val": metrics["score_val"],
        "trained_models_count": metrics["trained_models_count"],
    })

    mlflow.log_text(
        leaderboard.to_csv(index=False),
        "automl_leaderboard.csv",
    )

    return metrics, leaderboard