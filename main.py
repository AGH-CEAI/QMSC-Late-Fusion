import yaml

import experiments.hidden_manifold.noiseless as exp


def main():
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    exp.svc_classifiers_exp(config)
    # exp.checks(config)


if __name__ == "__main__":
    main()
