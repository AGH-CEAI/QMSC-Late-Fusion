import pennylane as qml


def qc_angle_emb_kernel(dev):
    @qml.qnode(dev)
    def circuit(x1, x2):
        qml.templates.AngleEmbedding(x1, wires=dev.wires)
        qml.adjoint(qml.templates.AngleEmbedding)(x2, wires=dev.wires)
        return qml.probs(wires=dev.wires)

    return lambda x1, x2: circuit(x1, x2)[0]


def qc_amplitude_emb_kernel(dev):
    @qml.qnode(dev)
    def circuit(x1, x2):
        qml.templates.AmplitudeEmbedding(
            x1, wires=dev.wires, pad_with=0, normalize=True
        )
        qml.adjoint(qml.templates.AmplitudeEmbedding)(
            x2, wires=dev.wires, pad_with=0, normalize=True
        )
        return qml.probs(wires=dev.wires)

    return lambda x1, x2: circuit(x1, x2)[0]
