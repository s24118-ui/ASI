from kedro.pipeline import Pipeline, node, pipeline

from .nodes import train_autogluon_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=train_autogluon_model,
                inputs="model_input",
                outputs=["automl_metrics", "automl_leaderboard"],
                name="train_autogluon_model_node",
            ),
        ]
    )
