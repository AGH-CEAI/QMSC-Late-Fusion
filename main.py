import yaml

import experiments.hidden_manifold.noiseless as exp


def main():
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    exp.single_dev_noiseless_exp(config)


if __name__ == "__main__":
    main()
