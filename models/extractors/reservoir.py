from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
import numpy.typing as npt
import numpy as np


class QuantumReservoir:
    def __init__(
        self,
        encoding_qc: QuantumCircuit,
        reservoir_qc: QuantumCircuit,
    ):
        self.circuit = encoding_qc.compose(reservoir_qc)

    def _extract_features_single(self, x: npt.NDArray) -> npt.NDArray:
        bound_qc = self.circuit.assign_parameters(x)
        state = Statevector.from_instruction(bound_qc)
        return state.probabilities()

    def extract_features_batch(self, x: npt.NDArray) -> npt.NDArray:
        return np.array(
            [self._extract_features_single(sample) for sample in x]
        )
