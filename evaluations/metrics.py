from itertools import combinations

import numpy as np
import pennylane as qml


def quantum_circuit_expressibility(
    circuit, param_ranges: list, num_samples: int = 50
):
    """
    Quantum Circuit Expressibility (QCE).

    Measures how well a parameterized quantum circuit can explore the Hilbert space.

    Mathematically, it computes:
    \[
    \text{QCE} = 1 - \frac{1}{N(N-1)} \sum_{i < j} |\langle \psi_i | \psi_j \rangle|^2
    \]
    where \( \psi_i \) and \( \psi_j \) are statevectors generated from random parameters.

    Source: https://gitlab.com/illesova.silvie.scholar/qmetric/-/blob/main/qmetric/quantum_circuit_metrics.py?ref_type=heads

    Parameters:
    - circuit: parameterized QNode.
    - param_ranges: list of (low, high) for each parameter.
    - num_samples: number of parameter samples to draw.

    Returns:
    - float: QCE score in [0, 1]. Higher is more expressive.
    """
    states = []
    for _ in range(num_samples):
        param_values = [
            np.random.uniform(low, high) for low, high in param_ranges
        ]
        state = circuit(param_values)
        states.append(state)

    overlaps = [
        qml.math.fidelity_statevector(s1, s2)
        for s1, s2 in combinations(states, 2)
    ]
    N = num_samples
    return 1 - (sum(overlaps) / (N * (N - 1)))
