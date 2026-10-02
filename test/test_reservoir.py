from typing import Tuple

import numpy.typing as npt
import pennylane as qml
import pytest
import yaml

import data.hidden_manifold as data
import models.extractors.reservoir as res


class TestReservoir:
    def test_qnode_init(self):
        qc = res.get_angle_embedding(4)
        assert isinstance(qc, qml.QNode)
        assert len(qc.device.wires) == 4
        print(qml.draw(qc)([1, 2, 3, 4]))

    def test_qnode_run(self):
        qc = res.get_angle_embedding(4)
        result = qc([1, 2, 3, 4])
        assert len(result) == 2**4
        assert round(sum(result)) == 1


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
