from models.extractors.reservoir import QuantumReservoir
from qiskit.circuit.random import random_circuit
from qiskit.circuit.library import z_feature_map
import numpy as np


def test_reservoir():
    encoder = z_feature_map(feature_dimension=4)
    qc = random_circuit(num_qubits=4, depth=5, measure=False, seed=42)

    reservoir = QuantumReservoir(
        encoding_qc=encoder,
        reservoir_qc=qc,
    )

    return reservoir.extract_features(np.array([0.34, 0.5, 0.12, 0.9]))
