import pennylane as qml


def get_angle_embedding(n_features: int) -> qml.QNode:
    dev = qml.device("default.qubit", wires=n_features)

    @qml.qnode(dev)
    def circuit(X):
        qml.AngleEmbedding(X, wires=range(n_features))
        return qml.expval(qml.PauliZ(wires=range(n_features)))

    return circuit
