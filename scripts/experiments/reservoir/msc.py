import data.loaders.hidden_manifold as dt
from data.processors.hidden_manifold import split_to_multisource
from models.extractors.reservoir import QuantumReservoir
from qiskit.circuit.random import random_circuit
from qiskit.circuit.library import z_feature_map
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
import numpy as np


def msc_1(config: dict):
    """
    Seperate quantum reservoi for feater extraction.
    Fusion with classical linear layer.
    """
    # Load hidden-manifold data
    x_train, y_train, x_test, y_test = dt.get_manifold(
        **config["datasets"]["hidden-manifold"]
    )

    # Quantum Extreme Reservoi Computing model
    encoder = z_feature_map(**config["feature_extractor"]["encoder"])
    qc = random_circuit(**config["feature_extractor"]["random_circuit"])
    reservoir_A = QuantumReservoir(
        encoding_qc=encoder,
        reservoir_qc=qc,
    )
    reservoir_B = reservoir_A.copy()

    # Classical Classifier
    clf = MLPClassifier(**config["training"])

    # Divide input to simulate multiple source
    train_x_A, train_x_B = split_to_multisource(
        x=x_train,
        n_feat_per_source=config["feature_extractor"]["encoder"][
            "feature_dimension"
        ],
    )

    test_x_A, test_x_B = split_to_multisource(
        x=x_test,
        n_feat_per_source=config["feature_extractor"]["encoder"][
            "feature_dimension"
        ],
    )

    # Extract features
    train_features = np.concatenate(
        (
            reservoir_A.extract_features_batch(train_x_A),
            reservoir_B.extract_features_batch(train_x_B),
        ),
        axis=1,
    )
    test_features = np.concatenate(
        (
            reservoir_A.extract_features_batch(test_x_A),
            reservoir_B.extract_features_batch(test_x_B),
        ),
        axis=1,
    )

    # Fit output layer
    clf.fit(train_features, y_train)

    y_pred = clf.predict(test_features)

    mean_acc = clf.score(test_features, y_test)
    print("\n\n_____\nHybrid model with z_feature_map for multisource:\n")
    print(mean_acc)

    conf = confusion_matrix(y_test, y_pred=y_pred)
    print(conf)
