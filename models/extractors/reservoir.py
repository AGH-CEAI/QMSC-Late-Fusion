from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import StatePreparation
import numpy.typing as npt
import numpy as np


class QuantumReservoir:
    def __init__(
        self,
        reservoir_qc: QuantumCircuit,
        encoding_qc: QuantumCircuit = None,
    ):
        self.reservoir_qc = reservoir_qc
        self.encoding_qc = encoding_qc

    def _extract_features_single(self, x: npt.NDArray) -> npt.NDArray:
        if self.encoding_qc is not None:
            # Parametric encoding
            bound_encoding = self.encoding_qc.assign_parameters(x)
            bound_qc = bound_encoding.compose(self.reservoir_qc)
        else:
            # Amplitude Encoding
            prep = StatePreparation(params=x, normalize=True)
            bound_qc = self.reservoir_qc.compose(prep, front=True)
        state = Statevector.from_instruction(bound_qc)
        return state.probabilities()

    def extract_features_batch(self, x: npt.NDArray) -> npt.NDArray:
        return np.array(
            [self._extract_features_single(sample) for sample in x]
        )
