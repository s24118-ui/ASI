from kedro.pipeline import Pipeline, node, pipeline

from .nodes import clean_credit_data, prepare_model_input


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=clean_credit_data,
                inputs="credit_score_raw",
                outputs="credit_score_clean",
                name="clean_credit_data_node",
            ),
            node(
                func=prepare_model_input,
                inputs="credit_score_clean",
                outputs="model_input",
                name="prepare_model_input_node",
            ),
        ]
    )