from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
import numpy.typing as npt


class QuantumReservoir:
    def __init__(
        self,
        encoding_qc: QuantumCircuit,
        reservoir_qc: QuantumCircuit,
    ):
        self.circuit = encoding_qc.compose(reservoir_qc)

    def extract_features(self, x: npt.NDArray) -> npt.NDArray:
        bound_qc = self.circuit.assign_parameters(x)
        state = Statevector.from_instruction(bound_qc)
        return state.probabilities()
