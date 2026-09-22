import yaml
import utils.mlflow as mf

from scripts.test import test_reservoir
from scripts.experiments.reservoir.initial_experiments import (
    initial_1,
    initial_classical_1,
    initial_2_amplitude,
)


def main():
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    mf.setup_mlflow("QMSC_Late_Fusion")
    config["training"]["max_iter"] = 100
    initial_1(config.copy())
    initial_classical_1(config.copy())
    initial_2_amplitude(config.copy())


if __name__ == "__main__":
    main()
