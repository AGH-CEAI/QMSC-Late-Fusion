import matplotlib.pyplot as plt
import pennylane as qml
import pennylane.numpy as np
import sklearn.preprocessing as pre
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC, LinearSVC

from data.hidden_manifold import HiddenManifold
from evaluations.metrics import (
    quantum_circuit_expressibility,
    quantum_output_sensitivity,
)
from models.extractors.reservoir import get_random_qc


def plot_pca(n_components, features, labels):
    pca = PCA(n_components=n_components)
    pca_features = pca.fit_transform(features)
    plt.scatter(pca_features[:, 0], pca_features[:, 1], c=labels)
    plt.show()


def plot_compare_plots(config):
    hm = HiddenManifold()
    x, y = hm.get_train()
    fig, axs = plt.subplots(3, 5)
    fig.suptitle(
        "First 2 PCA components: Raw data vs. Quantum circuit probs",
        weight="bold",
        fontsize=11,
    )
    axs[0, 0].set_ylabel("Original input", fontsize=11, weight="bold")
    axs[1, 0].set_ylabel(
        "Scaled input (-pi/1 , pi/2)", fontsize=11, weight="bold"
    )
    axs[2, 0].set_ylabel(
        "Normal distribution",
        fontsize=11,
        weight="bold",
    )

    scaler = pre.MinMaxScaler(feature_range=(-np.pi / 2, np.pi / 2))
    x_scaled = scaler.fit_transform(x)

    qt = pre.QuantileTransformer(output_distribution="normal")
    x_norm_dist = scaler.fit_transform(qt.fit_transform(x_scaled))

    x_raw_sc = PCA(n_components=2).fit_transform(x_scaled)
    x_raw = PCA(n_components=2).fit_transform(x)
    x_raw_qt = PCA(n_components=2).fit_transform(x_norm_dist)

    axs[0, 0].set_title("Raw input")
    axs[0, 0].scatter(x_raw[:, 0], x_raw[:, 1], c=y)
    axs[1, 0].scatter(x_raw_sc[:, 0], x_raw_sc[:, 1], c=y)
    axs[2, 0].scatter(x_raw_qt[:, 0], x_raw_qt[:, 1], c=y)

    dev = qml.device("default.qubit", wires=config["reservoir"]["num_qubits"])
    for i, depth in enumerate(range(0, 4, 1)):
        qc = get_random_qc(
            n_features=config["reservoir"]["num_qubits"],
            depth=depth,
            dev=dev,
            seed=config["seed"],
        )

        # fig_qc, ax = qml.draw_mpl(qc, level="device")(x_scaled[0])
        # fig_qc.suptitle(f"Random Layers = {depth}", fontsize="xx-large")

        x_qc = PCA(n_components=2).fit_transform(qc(x))
        x_qc_sc = PCA(n_components=2).fit_transform(qc(x_scaled))
        x_qc_norm = PCA(n_components=2).fit_transform(qc(x_norm_dist))

        axs[0, i + 1].set_title(f"QC, random layers={depth}")
        axs[0, i + 1].scatter(x_qc[:, 0], x_qc[:, 1], c=y)
        axs[1, i + 1].scatter(x_qc_sc[:, 0], x_qc_sc[:, 1], c=y)
        axs[2, i + 1].scatter(x_qc_norm[:, 0], x_qc_norm[:, 1], c=y)
    plt.show()


def show_histogram():
    hm = HiddenManifold()
    x, y = hm.get_train()

    scaler = pre.MinMaxScaler(feature_range=(-np.pi / 2, np.pi / 2))
    qt = pre.QuantileTransformer(output_distribution="normal")

    x_scaled = scaler.fit_transform(x)
    x_normal = scaler.fit_transform(qt.fit_transform(x_scaled))

    fig, axs = plt.subplots(2, 6, sharey=True, tight_layout=True)
    fig.suptitle("Histogram per feature")
    n_bins = 10
    for i in range(6):
        axs[0, i].set_title(f"Feature {i + 1}")
        axs[0, i].hist(x_scaled[:, i], bins=n_bins)
        axs[1, i].hist(x_normal[:, i], bins=n_bins)

    plt.show()


def compare_separability(config):
    # Get data
    hm = HiddenManifold()
    x, y = hm.get_train()
    x_test, y_test = hm.get_test()

    # Set preprocessors
    scaler = pre.MinMaxScaler(feature_range=(-np.pi / 2, np.pi / 2))
    x_scaled = scaler.fit_transform(x)
    x_test_scaled = scaler.transform(x_test)

    qt = pre.QuantileTransformer(output_distribution="normal")
    x_norm_scaled = scaler.fit_transform(qt.fit_transform(x))
    x_test_norm_scaled = scaler.transform(qt.fit_transform(x_test))

    # Build model
    # model = LinearSVC(random_state=42, max_iter=300, dual="auto")
    model = SVC(random_state=42, max_iter=1000)

    # Get classical predictions
    model.fit(x, y)
    acc = accuracy_score(y_test, model.predict(x_test))
    print(f"Raw data: {acc}")

    model.fit(x_scaled, y)
    acc = accuracy_score(y_test, model.predict(x_test_scaled))
    print(f"Scaled data: {acc}")

    model.fit(x_norm_scaled, y)
    acc = accuracy_score(y_test, model.predict(x_test_norm_scaled))
    print(f"Norm. distr. scaled data: {acc}")

    # Get quantum-classical predictions
    dev = qml.device("default.qubit", wires=config["reservoir"]["num_qubits"])
    for i, depth in enumerate(range(0, 25, 3)):
        qc = get_random_qc(
            n_features=config["reservoir"]["num_qubits"],
            depth=depth,
            dev=dev,
            seed=config["seed"],
        )

        print(
            f"\n----------------- Random Layers: {depth} --------------------------"
        )
        model.fit(qc(x), y)
        acc = accuracy_score(y_test, model.predict(qc(x_test)))
        print(f"Raw data: {acc}")

        model.fit(qc(x_scaled), y)
        acc = accuracy_score(y_test, model.predict(qc(x_test_scaled)))
        print(f"Scaled data: {acc}")

        model.fit(qc(x_norm_scaled), y)
        acc = accuracy_score(y_test, model.predict(qc(x_test_norm_scaled)))
        print(f"Norm. distr. scaled data: {acc}")


def compare_preds(config):
    # Get data
    hm = HiddenManifold()
    x, y = hm.get_train()
    x_test, y_test = hm.get_test()

    qc = get_random_qc(
        n_features=config["reservoir"]["num_qubits"],
        depth=12,
        dev=qml.device(
            "default.qubit", wires=config["reservoir"]["num_qubits"]
        ),
        seed=config["seed"],
    )

    model = SVC(random_state=42, max_iter=1000)

    model.fit(x, y)
    preds = model.predict(x_test)

    model.fit(qc(x), y)
    q_preds = model.predict(qc(x_test))

    for true_y, pred, qpred in zip(
        y_test[preds != q_preds],
        preds[preds != q_preds],
        q_preds[preds != q_preds],
    ):
        print(true_y, pred, qpred)


def test_qce(config):
    for depth in range(20):
        qc = get_random_qc(
            n_features=config["reservoir"]["num_qubits"],
            depth=depth,
            dev=qml.device(
                "default.qubit", wires=config["reservoir"]["num_qubits"]
            ),
            seed=config["seed"],
        )
        expr = quantum_circuit_expressibility(
            circuit=qc,
            param_ranges=[(-np.pi / 2, np.pi / 2)] * 6,
            num_samples=100,
        )
        print(f"depth: {depth}, sen: {expr}")


def test_qos(config):
    hm = HiddenManifold()
    x, y = hm.get_train()
    epsilon = np.random.normal(loc=0.0, scale=0.01, size=(len(x), 6))

    for depth in range(20):
        qc = get_random_qc(
            n_features=config["reservoir"]["num_qubits"],
            depth=depth,
            dev=qml.device(
                "default.qubit", wires=config["reservoir"]["num_qubits"]
            ),
            seed=config["seed"],
        )
        sen = quantum_output_sensitivity(
            fx=qc(x), fx_eps=qc(x + epsilon), eps=epsilon
        )
        print(f"depth: {depth}, sen: {sen}")
