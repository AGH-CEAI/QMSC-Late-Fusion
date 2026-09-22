import yaml
import data.loaders.hidden_manifold as dt
from models.extractors.reservoir import QuantumReservoir
from qiskit.circuit.random import random_circuit
from qiskit.circuit.library import z_feature_map
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix


def initial_1(seed: int = 42):
    # Load hidden-manifold data
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    x_train, y_train, x_test, y_test = dt.get_manifold(
        config["datasets"]["hidden-manifold"], dim=8, diff=False
    )

    # Quantum Extreme Reservoi Computing model
    encoder = z_feature_map(feature_dimension=8)
    qc = random_circuit(num_qubits=8, depth=1, measure=False, seed=seed)
    reservoir = QuantumReservoir(
        encoding_qc=encoder,
        reservoir_qc=qc,
    )

    # Classical Classifier
    clf = MLPClassifier(
        solver="sgd",
        alpha=1e-5,
        hidden_layer_sizes=(),
        random_state=42,
        max_iter=300,
    )

    # Feature Extraction:
    train_features = reservoir.extract_features_batch(x_train)
    test_features = reservoir.extract_features_batch(x_test)

    print(test_features[0])

    # Fit output layer
    clf.fit(train_features, y_train)

    y_pred = clf.predict(test_features)

    mean_acc = clf.score(test_features, y_test)
    print(mean_acc)

    conf = confusion_matrix(y_test, y_pred=y_pred)
    print(conf)


def initial_classical_1(seed: int = 42):
    # Load hidden-manifold data
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    x_train, y_train, x_test, y_test = dt.get_manifold(
        config["datasets"]["hidden-manifold"], dim=8, diff=False
    )

    # Classical Classifier
    clf = MLPClassifier(
        solver="sgd",
        alpha=1e-5,
        hidden_layer_sizes=(),
        random_state=42,
        max_iter=300,
    )

    # Fit output layer
    clf.fit(x_train, y_train)

    mean_acc = clf.score(x_test, y_test)

    print(mean_acc)


##################################################################


def initial_2(seed: int = 42):
    # Load hidden-manifold data
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    x_train, y_train, x_test, y_test = dt.get_manifold(
        config["datasets"]["hidden-manifold"], dim=4, diff=False
    )

    # Quantum Extreme Reservoi Computing model
    encoder = z_feature_map(feature_dimension=4)
    qc = random_circuit(num_qubits=4, depth=3, measure=False, seed=seed)
    reservoir = QuantumReservoir(
        encoding_qc=encoder,
        reservoir_qc=qc,
    )

    # Classical Classifier
    clf = MLPClassifier(
        solver="sgd",
        alpha=1e-5,
        hidden_layer_sizes=(),
        random_state=42,
        max_iter=300,
    )

    # Feature Extraction:
    train_features = reservoir.extract_features_batch(x_train)
    test_features = reservoir.extract_features_batch(x_test)

    print(test_features[0])

    # Fit output layer
    clf.fit(train_features, y_train)

    y_pred = clf.predict(test_features)

    mean_acc = clf.score(test_features, y_test)
    print(mean_acc)

    conf = confusion_matrix(y_test, y_pred=y_pred)
    print(conf)


def initial_classical_2(seed: int = 42):
    # Load hidden-manifold data
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    x_train, y_train, x_test, y_test = dt.get_manifold(
        config["datasets"]["hidden-manifold"], dim=4, diff=False
    )

    # Classical Classifier
    clf = MLPClassifier(
        solver="sgd",
        alpha=1e-5,
        hidden_layer_sizes=(),
        random_state=42,
        max_iter=300,
    )

    # Fit output layer
    clf.fit(x_train, y_train)

    mean_acc = clf.score(x_test, y_test)

    print(mean_acc)
