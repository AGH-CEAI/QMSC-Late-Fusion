import pennylane as qml


def circuit_to_kernel(dev, circuit):
    """
    Takes quantum device and circuit as arguments and returns kernel function.
    Source: https://pennylane.ai/demos/tutorial_kernels_module
    """

    @qml.qnode(dev)
    def kernel(x1, x2):
        circuit(x1, wires=dev.wires)
        qml.adjoint(circuit)(x2, wires=dev.wires)
        return qml.probs(wires=dev.wires)

    return lambda x1, x2: kernel(x1, x2)[0]


def repeated_amplitude_encoding(n_reapeats: int):
    def circuit(x, wires):
        for i in range(n_reapeats):
            qml.AmplitudeEmbedding(x, wires, pad_with=True, normalize=True)

    return circuit
