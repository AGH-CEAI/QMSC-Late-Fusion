import pennylane as qml
import sklearn as sk
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


def get_dataloader_(config):
    data_loader_config = config["datasets"]["hidden-manifold"]

    return HiddenManifold(
        data_path=data_loader_config["data_path"],
        dim=data_loader_config["dim"],
        diff=data_loader_config["diff"],
    )


################################################################################
# EXPERIMENTS ##################################################################
################################################################################


def single_dev_noiseless_exp(config: dict):
    quantum_circuits = [get_angle_embedding(config["reservoir"]["num_qubits"])]

    exp = ExperimentRunner(
        data_loader=get_dataloader_(),
        quantum_circuits=quantum_circuits,
        classifier=get_classifier_(config),
        final_estimator=get_classifier_(config),
        config=config,
    )

    score = exp.run()
    print(score["test_accuracy"])


def single_dev_random_noiseless_exp(config: dict):
    quantum_circuits = [get_random_qc_(config)]

    exp = ExperimentRunner(
        data_loader=get_dataloader_(config),
        quantum_circuits=quantum_circuits,
        classifier=get_classifier_(config),
        final_estimator=get_classifier_(config),
        config=config,
    )

    score = exp.run()
    print(score["test_accuracy"])


def different_classifiers_exp(config: dict):
    quantum_circuits = [get_random_qc_(config)]

    classifiers = [
        get_classifier_(config),
        sk.svm.SVC(
            kernel="rbf", probability=True, random_state=config["seed"]
        ),
        sk.tree.DecisionTreeClassifier(random_state=config["seed"]),
    ]

    for classifier in classifiers:
        exp = ExperimentRunner(
            data_loader=get_dataloader_(config),
            quantum_circuits=quantum_circuits,
            classifier=classifier,
            final_estimator=classifier,
            config=config,
        )
        score = exp.run()
    print(score["test_accuracy"])


def svc_classifiers_exp(config: dict):
    quantum_circuits = [get_random_qc_(config)]

    exp = ExperimentRunner(
        data_loader=get_dataloader_(config),
        quantum_circuits=quantum_circuits,
        classifier=get_classifier_(config),
        final_estimator=sk.svm.SVC(
            kernel="rbf",
            random_state=config["seed"],
            max_iter=config["training"]["max_iter"],
        ),
        config=config,
    )
    score = exp.run()
    print(score["test_accuracy"])
