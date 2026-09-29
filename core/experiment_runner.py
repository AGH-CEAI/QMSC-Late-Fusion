from typing import List

from sklearn.base import ClassifierMixin
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline

import core.pipeline_factories as pipe
from data.loaders.hidden_manifold import BaseDataLoader


class ExperimentRunner:
    """
    Executes a machine learning experiment using cross-validation.
    """

    def __init__(
        self,
        data_loader: BaseDataLoader,
        transformers: List[Pipeline],
        classifier: ClassifierMixin,
        config: dict,
    ):
        """
        Initializes the experiment runner.

        Args:
            data_loader (BaseDataLoader): Component for loading the training data.
            transformers (List[Pipeline]): List of pipelines for feature extraction.
            classifier (ClassifierMixin): A scikit-learn compatible classifier.
            config (dict): Configuration dictionary containing 'feature_extractor',
                           'training', and 'evaluation' settings.
        """
        # Object fields
        self.data_loader = data_loader
        self.transformers = transformers
        self.classifier = classifier

        # Configuration fields
        self.feature_extractor_config = config["feature_extractor"]
        self.training_config = config["training"]
        self.eval_config = config["evaluation"]

    def run(self):
        """
        Runs the full experiment pipeline: loads data, builds the multisource
        classification pipeline, evaluates it via cross-validation, and logs results.
        """
        # Load data
        X, y = self.data_loader.get_train()

        # Build pipeline
        preprocessor = pipe.build_multisource_transformer(
            transformers=self.transformers,
            n_features_per_src=self.feature_extractor_config[
                "feature_dimension"
            ],
        )
        pipeline = pipe.build_classification_pipeline(
            preprocessor=preprocessor, classifier=self.classifier
        )

        # Cross-validation
        score = cross_validate(
            estimator=pipeline,
            X=X,
            y=y,
            scoring=self.eval_config["scoring"],
            cv=self.training_config["n_folds"],
        )

        # Log model, params and results
