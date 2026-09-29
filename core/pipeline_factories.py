from typing import List
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


def build_multisource_pipeline(
    transformers: List[Pipeline], n_features_per_src: int
) -> ColumnTransformer:
    """
    Builds a ColumnTransformer that splits input data into equal blocks
    and assigns them to corresponding transformers.
    Source for ColumnTransformer: https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html

    Args:
        transformers (List[Pipeline]): List of transformers for individual sources.
        n_features_per_src (int): Number of feature columns per source.

    Returns:
        ColumnTransformer: Configured pipeline object for data splitting.
    """
    col_transformers = []

    for i, pipe in enumerate(transformers):
        # Calculate indices of feature columns to be used by each transformer
        start_idx = i * n_features_per_src
        end_idx = (i + 1) * n_features_per_src
        columns_to_transform = slice(start_idx, end_idx)

        # Prepare transformers as tuples in format required by ColumnTransformer
        col_transformers.append(
            (
                f"src_{i}",  # name
                pipe,  # transformer
                columns_to_transform,  # columns
            )
        )
    return ColumnTransformer(transformers=col_transformers, remainder="drop")
