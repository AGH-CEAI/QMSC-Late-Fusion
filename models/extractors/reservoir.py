import pennylane as qml
from pennylane import numpy as np


def get_angle_embedding(n_features: int) -> qml.QNode:
    dev = qml.device("default.qubit", wires=n_features)

    @qml.qnode(dev)
    def circuit(X):
        qml.AngleEmbedding(X, wires=range(n_features))
        return qml.probs(wires=range(n_features))

    return circuit


def get_random_qc(n_features: int, depth: int, dev, seed: int) -> qml.QNode:
    # Generate static random weights
    shape = qml.RandomLayers.shape(n_layers=depth, n_rotations=n_features)
    weights = np.random.random(size=shape)

    @qml.qnode(dev)
    def circuit(X):
        qml.AngleEmbedding(X, wires=range(n_features))
        qml.RandomLayers(weights=weights, wires=n_features, seed=seed)
        return qml.probs(wires=range(n_features))

    return circuit
