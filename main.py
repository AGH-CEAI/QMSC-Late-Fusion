import yaml

import utils.mlflow as mf


def main():
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    mf.setup_mlflow("QMSC_Late_Fusion")


if __name__ == "__main__":
    main()
