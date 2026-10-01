import numpy as np
import numpy.typing as npt
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from sklearn.base import BaseEstimator, TransformerMixin


class QuantumReservoir(TransformerMixin, BaseEstimator):
    """
    A Quantum Reservoir for extracting features from input data.

    Supports Parametric Encoding

    Args:
        reservoir_qc (QuantumCircuit): The quantum circuit acting as the parameter-free reservoir.
        encoding_qc (QuantumCircuit): The quantum circuit for data encoding. Defaults to None.
    """

    def __init__(
        self,
        reservoir_qc: QuantumCircuit,
        encoding_qc: QuantumCircuit,
    ):
        self.quantum_circuit = encoding_qc.compose(reservoir_qc)

    def fit(self, X: npt.NDArray, y=None):
        """Scikit-learn API requirement. Returns self."""
        return self

    def transform(self, X: npt.NDArray) -> npt.NDArray:
        """
        Extracts reservoir features for a batch of input samples.

        Args:
            x (npt.NDArray): A batch of input data samples.

        Returns:
            npt.NDArray: An array of extracted features (probabilities) for each sample.
        """
        return np.array(
            [self._extract_features_single(sample) for sample in X]
        )

    def copy(self):
        return QuantumReservoir(
            reservoir_qc=self.reservoir_qc.copy(),
            encoding_qc=self.encoding_qc.copy(),
        )

    def _extract_features_single(self, x: npt.NDArray) -> npt.NDArray:
        """
        Extracts features for a single input sample.

        Args:
            x (npt.NDArray): A single input data array.

        Returns:
            npt.NDArray: Measurement probabilities of the resulting quantum state.
        """
        bound_encoding = self.quantum_circuit.assign_parameters(x)
        state = Statevector.from_instruction(bound_encoding)
        return state.probabilities()
