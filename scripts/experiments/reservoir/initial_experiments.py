import data.loaders.hidden_manifold as dt
from models.extractors.reservoir import QuantumReservoir, QuantumReservoir2
from qiskit.circuit.random import random_circuit
from qiskit.circuit.library import z_feature_map, StatePreparation
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix


def initial_1(config: dict):
    # Load hidden-manifold data
    x_train, y_train, x_test, y_test = dt.get_manifold(
        **config["datasets"]["hidden-manifold"]
    )

    # Quantum Extreme Reservoi Computing model
    encoder = z_feature_map(
        feature_dimension=config["datasets"]["hidden-manifold"]["dim"]
    )
    qc = random_circuit(**config["feature_extractor"]["random_circuit"])
    reservoir = QuantumReservoir(
        encoding_qc=encoder,
        reservoir_qc=qc,
    )

    # Classical Classifier
    clf = MLPClassifier(**config["training"])

    # Feature Extraction:
    train_features = reservoir.extract_features_batch(x_train)
    test_features = reservoir.extract_features_batch(x_test)

    # print(test_features[0])

    # Fit output layer
    clf.fit(train_features, y_train)

    y_pred = clf.predict(test_features)

    mean_acc = clf.score(test_features, y_test)
    print("\n\n_____\nHybrid model with z_feature_map:\n")
    print(mean_acc)

    conf = confusion_matrix(y_test, y_pred=y_pred)
    print(conf)


def initial_classical_1(config):
    # Load hidden-manifold data
    x_train, y_train, x_test, y_test = dt.get_manifold(
        **config["datasets"]["hidden-manifold"]
    )

    # Classical Classifier
    clf = MLPClassifier(**config["training"])

    # Fit output layer
    clf.fit(x_train, y_train)

    mean_acc = clf.score(x_test, y_test)
    print("\n\n_____\nClassical single layer:\n")
    print(mean_acc)


def initial_2_amplitude(config: dict):
    config["feature_extractor"]["random_circuit"]["num_qubits"] = 3
    # Load hidden-manifold data
    x_train, y_train, x_test, y_test = dt.get_manifold(
        **config["datasets"]["hidden-manifold"]
    )

    # Quantum Extreme Reservoi Computing model
    qc = random_circuit(**config["feature_extractor"]["random_circuit"])
    reservoir = QuantumReservoir(reservoir_qc=qc)

    # Classical Classifier
    clf = MLPClassifier(**config["training"])

    # Feature Extraction:
    train_features = reservoir.extract_features_batch(x_train)
    test_features = reservoir.extract_features_batch(x_test)

    # print(test_features[0])

    # Fit output layer
    clf.fit(train_features, y_train)

    y_pred = clf.predict(test_features)

    mean_acc = clf.score(test_features, y_test)
    print("\n\n_____\nHybrid model with amplitude:\n")
    print(mean_acc)

    conf = confusion_matrix(y_test, y_pred=y_pred)
    print(conf)
