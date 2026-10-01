from typing import List

from pennylane import QNode
from sklearn.base import BaseEstimator
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

import core.pipeline_factories as pipe
from data.hidden_manifold import BaseDataLoader


class ExperimentRunner:
    """
    Executes a machine learning experiment using cross-validation.
    """

    def __init__(
        self,
        data_loader: BaseDataLoader,
        quantum_circuits: List[QNode],
        classifier: BaseEstimator,
        final_estimator: BaseEstimator,
        config: dict,
    ):
        """
        Initializes the experiment runner.

        Args:
            data_loader (BaseDataLoader): Component for loading the training data.
            quantum_circuits (List[QNode]): List of quantum circuits for feature extraction.
            classifier (BaseEstimator): A scikit-learn compatible base classifier.
            final_estimator (BaseEstimator): A scikit-learn compatible final estimator for the stacking ensemble.
            config (dict): Configuration dictionary containing 'feature_extractor',
                           'training', and 'evaluation' settings.
        """
        # Object fields
        self.data_loader = data_loader
        self.quantum_circuits = quantum_circuits
        self.classifier = classifier
        self.final_estimator = final_estimator
        self.ensamble = None

        # Configuration fields
        self.feature_extractor_config = config["feature_extractor"]
        self.training_config = config["training"]
        self.eval_config = config["evaluation"]
        self.seed = config["seed"]

    def run(self):
        """
        Runs the full experiment pipeline: loads data, builds the multisource
        classification pipeline, evaluates it via cross-validation, and logs results.
        """
        # Load data
        X, y = self.data_loader.get_train()

        # Build cross-validator
        cv = StratifiedKFold(
            n_splits=self.training_config["n_folds"],
            shuffle=True,
            random_state=self.seed,
        )

        # Build classification pipeline
        estimators = pipe.build_estimators(
            quantum_circuits=self.quantum_circuits, classifier=self.classifier
        )
        ensamble = StackingClassifier(
            estimators=estimators,
            final_estimator=self.final_estimator,
            stack_method="predict_proba",
            n_job=-1,
            cv=cv,
        )

        # Cross-validation
        score = cross_validate(
            estimator=ensamble,
            X=X,
            y=y,
            scoring=self.eval_config["scoring"],
            cv=cv,
        )

        # Log model, params and results (TODO)
        return score
