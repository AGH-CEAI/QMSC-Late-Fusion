from typing import List, Tuple

from pennylane import QNode
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer


def build_reservoir_estimator(
    qc: QNode, classifier: MLPClassifier
) -> Pipeline:
    qc_trans = FunctionTransformer(qc)
    return Pipeline(steps=[("qc", qc_trans), ("nn", classifier)])


def build_estimators(
    quantum_circuits: List[QNode], classifier: MLPClassifier
) -> List[Tuple[str, Pipeline]]:
    estimators = list()
    for i, qc in enumerate(quantum_circuits):
        est = build_reservoir_estimator(qc=qc, nn=classifier)
        estimators.append((f"dev_{i}", est))

    return estimators
