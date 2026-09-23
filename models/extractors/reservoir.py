from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import StatePreparation
import numpy.typing as npt
import numpy as np


class QuantumReservoir:
    """
    A Quantum Reservoir for extracting features from input data.

    Supports Parametric Encoding (if `encoding_qc` is provided) and
    Amplitude Encoding (if `encoding_qc` is None).

    Args:
        reservoir_qc (QuantumCircuit): The quantum circuit acting as the parameter-free reservoir.
        encoding_qc (QuantumCircuit, optional): The quantum circuit for data encoding. Defaults to None.
    """

    def __init__(
        self,
        reservoir_qc: QuantumCircuit,
        encoding_qc: QuantumCircuit = None,
    ):
        self.reservoir_qc = reservoir_qc
        self.encoding_qc = encoding_qc

    def _extract_features_single(self, x: npt.NDArray) -> npt.NDArray:
        """
        Extracts features for a single input sample.

        Args:
            x (npt.NDArray): A single input data array.

        Returns:
            npt.NDArray: Measurement probabilities of the resulting quantum state.
        """
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
        """
        Extracts reservoir features for a batch of input samples.

        Args:
            x (npt.NDArray): A batch of input data samples.

        Returns:
            npt.NDArray: An array of extracted features (probabilities) for each sample.
        """
        return np.array(
            [self._extract_features_single(sample) for sample in x]
        )

    def copy(self):
        return QuantumReservoir(
            reservoir_qc=self.reservoir_qc.copy(),
            encoding_qc=self.encoding_qc.copy(),
        )
