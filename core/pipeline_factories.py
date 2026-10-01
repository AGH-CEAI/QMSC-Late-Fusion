from sklearn.base import ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def build_classification_pipeline(
    preprocessor: ColumnTransformer, classifier: ClassifierMixin
) -> Pipeline:
    """
    Builds a complete scikit-learn pipeline by combining a preprocessor and a classifier.

    Args:
        preprocessor (ColumnTransformer): The component handling data transformations.
        classifier (ClassifierMixin): The final estimator used for predictions.

    Returns:
        Pipeline: A constructed pipeline ready for training and evaluation.
    """
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )
