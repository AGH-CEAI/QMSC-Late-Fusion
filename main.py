import yaml
import utils.mlflow as mf
from scripts.experiments.reservoir.initial_experiments import (
    initial_1,
    initial_classical_1,
)


def main():
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    mf.setup_mlflow("QMSC_Late_Fusion")
    # initial_1(config)
    config["training"]["max_iter"] = 500
    initial_classical_1(config)


if __name__ == "__main__":
    main()
