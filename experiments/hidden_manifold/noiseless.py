import pennylane as qml
from sklearn.neural_network import MLPClassifier

from core.experiment_runner import ExperimentRunner
from data.hidden_manifold import HiddenManifold
from models.extractors.reservoir import get_angle_embedding, get_random_qc


################################################################################
# HELPER FUNCTIONS #############################################################
################################################################################
def get_classifier_(config: dict):
    training_config = config["training"]
    return MLPClassifier(
        hidden_layer_sizes=training_config["hidden_layer_sizes"],
        solver=training_config["solver"],
        random_state=config["seed"],
        max_iter=training_config["max_iter"],
        learning_rate=training_config["learning_rate"],
        learning_rate_init=training_config["learning_rate_init"],
        early_stopping=training_config["early_stopping"],
        activation=training_config["activation"],
    )


def get_random_qc_(config: dict):
    return get_random_qc(
        n_features=config["reservoir"]["num_qubits"],
        depth=config["reservoir"]["depth"],
        dev=qml.device(
            "default.qubit", wires=config["reservoir"]["num_qubits"]
        ),
        seed=config["seed"],
    )


def checks(config):
    print(qml.draw(get_random_qc_(config))([1, 2, 3, 4, 5, 6]))
    print(get_random_qc_(config)([1, 2, 3, 4, 5, 6]))


################################################################################
# EXPERIMENTS ##################################################################
################################################################################


def single_dev_noiseless_exp(config: dict):
    data_loader_config = config["datasets"]["hidden-manifold"]

    data_loader = HiddenManifold(
        data_path=data_loader_config["data_path"],
        dim=data_loader_config["dim"],
        diff=data_loader_config["diff"],
    )

    quantum_circuits = [get_angle_embedding(data_loader_config["dim"])]

    classifier = get_classifier_(config)

    exp = ExperimentRunner(
        data_loader=data_loader,
        quantum_circuits=quantum_circuits,
        classifier=classifier,
        final_estimator=classifier,
        config=config,
    )

    score = exp.run()
    print(score["test_accuracy"])


def single_dev_random_noiseless_exp(config: dict):
    data_loader_config = config["datasets"]["hidden-manifold"]

    data_loader = HiddenManifold(
        data_path=data_loader_config["data_path"],
        dim=data_loader_config["dim"],
        diff=data_loader_config["diff"],
    )

    quantum_circuits = [get_random_qc_(config)]

    classifier = get_classifier_(config)

    exp = ExperimentRunner(
        data_loader=data_loader,
        quantum_circuits=quantum_circuits,
        classifier=classifier,
        final_estimator=classifier,
        config=config,
    )

    score = exp.run()
    print(score["test_accuracy"])
