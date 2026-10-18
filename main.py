import yaml

import evaluations.kernel_evaluations as eval
import experiments.hidden_manifold.noiseless as exp


def main():
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    eval.svm_accuracy_check()


if __name__ == "__main__":
    main()
