from typing import List

import mlflow
from pennylane import QNode
from sklearn.base import BaseEstimator
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

import core.pipeline_factories as pipe
import utils.mlflow as mf
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
            config (dict): Configuration dictionary
        """

        self.data_loader = data_loader
        self.quantum_circuits = quantum_circuits
        self.classifier = classifier
        self.final_estimator = final_estimator
        self.ensamble = None

        self.seed = config["seed"]
        self.config = config

    def run(self):
        """
        Runs the full experiment pipeline: loads data, builds the multisource
        classification pipeline, evaluates it via cross-validation, and logs results.
        """
        # Load data
        X, y = self.data_loader.get_train()

        # Build cross-validator
        cv = StratifiedKFold(
            n_splits=self.config["training"]["n_folds"],
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
            n_jobs=-1,
            cv=cv,
        )

        # Cross-validation
        mf.setup_mlflow(self.config["experiment_name"])
        # mlflow.sklearn.autolog()
        with mlflow.start_run(run_name=self.config["run_name"]) as run:
            score = cross_validate(
                estimator=ensamble,
                X=X,
                y=y,
                scoring=self.config["evaluation"]["scoring"],
                cv=cv,
            )
            mf.log_cross_val_metrics(score=score)
            mf.log_params(config=self.config)

        return score
