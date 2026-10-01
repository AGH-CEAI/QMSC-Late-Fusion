from typing import Tuple

import numpy.typing as npt
import pytest
import yaml
from qiskit import QuantumCircuit
from qiskit.circuit.library import z_feature_map
from qiskit.circuit.random import random_circuit

import data.loaders.hidden_manifold as data
import models.extractors.reservoir as res


class TestQuantumReservoir:
    def test_init(
        self,
        dataset: Tuple[npt.NDArray, npt.NDArray],
        encoding_qc: QuantumCircuit,
        reservoir_qc: QuantumCircuit,
    ):
        X, y = data
        qc = res.QuantumReservoir(
            reservoir_qc=reservoir_qc, encoding_qc=encoding_qc
        )

        assert isinstance(qc.quantum_circuit, QuantumCircuit)
        assert qc.quantum_circuit.num_qubits == 4
        assert len(qc.quantum_circuit.parameters) == 4


@pytest.fixture
def dataset(config: dict) -> Tuple[npt.NDArray, npt.NDArray]:
    hm = data.HiddenManifold(
        data_path=config["datasets"]["hidden-manifold"]["data_path"],
        dim=4,
        diff=False,
    )
    return hm.get_train()


@pytest.fixture
def config() -> dict:
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    return config


@pytest.fixture
def encoding_qc() -> QuantumCircuit:
    return z_feature_map(feature_dimension=4)


@pytest.fixture
def reservoir_qc() -> QuantumCircuit:
    return random_circuit(num_qubits=4, depth=3, measure=False)
