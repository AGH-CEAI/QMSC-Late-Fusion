import yaml
import data.loaders.hidden_manifold as dt
from scripts.test import test_reservoir


def main():
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    x_train, y_train, x_test, y_test = dt.get_manifold(
        config["datasets"]["hidden-manifold"], dim=8, diff=False
    )

    print(test_reservoir())


if __name__ == "__main__":
    main()
