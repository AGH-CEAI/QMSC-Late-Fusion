import yaml

import experiments.hidden_manifold.noiseless as exp
import utils.mlflow as mf


def main():
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    mf.setup_mlflow("QMSC_Late_Fusion")
    exp.single_dev_noiseless_exp(config)


if __name__ == "__main__":
    main()
