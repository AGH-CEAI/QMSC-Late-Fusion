from sklearn.neural_network import MLPClassifier

from core.experiment_runner import ExperimentRunner
from data.hidden_manifold import HiddenManifold
from models.extractors.reservoir import get_angle_embedding


def single_dev_noiseless_exp(config: dict):
    data_loader_config = config["datasets"]["hidden-manifold"]
    training_config = config["training"]

    data_loader = HiddenManifold(
        data_path=data_loader_config["data_path"],
        dim=data_loader_config["dim"],
        diff=data_loader_config["diff"],
    )

    quantum_circuits = [get_angle_embedding(data_loader_config["dim"])]

    classifier = MLPClassifier(
        hidden_layer_sizes=training_config["hidden_layer_sizes"],
        solver=training_config["solver"],
        random_state=config["seed"],
        max_iter=training_config["max_iter"],
        learning_rate=training_config["learning_rate"],
        learning_rate_init=training_config["learning_rate_init"],
        early_stopping=training_config["early_stopping"],
    )

    exp = ExperimentRunner(
        data_loader=data_loader,
        quantum_circuits=quantum_circuits,
        classifier=classifier,
        final_estimator=classifier,
        config=config,
    )

    score = exp.run()
    print(score)
    print(score["test_accuracy"])
