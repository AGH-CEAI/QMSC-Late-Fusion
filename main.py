import yaml

import evaluations.model_evaluations as eval
import experiments.hidden_manifold.noiseless as exp


def main():
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # eval.plot_compare_plots(config)
    eval.show_histogram()


if __name__ == "__main__":
    main()
