import numpy as np
import pennylane as qml


def circuit_to_kernel(dev, circuit):
    """
    Takes quantum device and circuit as arguments and returns kernel function.
    Source: https://pennylane.ai/demos/tutorial_kernel_based_training
    """

    @qml.qnode(dev)
    def kernel(x1, x2):
        circuit(x1, wires=dev.wires)
        qml.adjoint(circuit)(x2, wires=dev.wires)
        return qml.probs(wires=dev.wires)

    return lambda A, B: np.array([[kernel(a, b)[0] for b in B] for a in A])
