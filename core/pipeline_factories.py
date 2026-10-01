from pennylane import QNode
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer


def build_reservoir_estimator(qc: QNode, nn: MLPClassifier) -> Pipeline:
    qc_trans = FunctionTransformer(qc)
    return Pipeline(steps=[("qc", qc_trans), ("nn", nn)])
