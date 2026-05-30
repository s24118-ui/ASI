from kedro.pipeline import Pipeline, node, pipeline

from .nodes import train_baseline_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=train_baseline_model,
                inputs=["model_input", "params:modeling"],
                outputs=["baseline_model", "metrics"],
                name="train_baseline_model_node",
            ),
        ]
    )
