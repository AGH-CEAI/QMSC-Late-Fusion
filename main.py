import yaml
import data.loaders.hidden_manifold as dt
from scripts.test import test_reservoir
from scripts.experiments.reservoir.initial_experiments import (
    initial_1,
    initial_classical_1,
    initial_2,
    initial_classical_2,
)


def main():
    initial_2()
    initial_classical_2()


if __name__ == "__main__":
    main()
