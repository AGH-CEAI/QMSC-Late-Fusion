import yaml
import utils.mlflow as mf

from scripts.experiments.reservoir.initial_experiments import (
    initial_1,
    initial_classical_1,
    initial_classical_2,
    initial_classical_3,
    initial_2_amplitude,
)
from scripts.experiments.reservoir.msc import msc_1


def main():
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    mf.setup_mlflow("QMSC_Late_Fusion")
    config["training"]["max_iter"] = 200
    initial_1(config.copy())
    initial_2_amplitude(config.copy())
    initial_classical_1(config.copy())
    initial_classical_2(config.copy())
    initial_classical_3(config.copy())
    msc_1(config.copy())


if __name__ == "__main__":
    main()
