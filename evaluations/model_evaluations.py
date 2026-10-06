import matplotlib.pyplot as plt
import pennylane as qml
from sklearn.decomposition import PCA

from data.hidden_manifold import HiddenManifold
from models.extractors.reservoir import get_random_qc


def plot_pca(n_components, features, labels):
    pca = PCA(n_components=n_components)
    pca_features = pca.fit_transform(features)
    plt.scatter(pca_features[:, 0], pca_features[:, 1], c=labels)
    plt.show()


def plot_pca_raw_data(config):
    conf_data = config["datasets"]["hidden-manifold"]
    hm = HiddenManifold(data_path=conf_data["data_path"], dim=conf_data["dim"])
    x, y = hm.get_train()
    plot_pca(2, x, y)


def plot_pca_qc_output(config):
    conf_data = config["datasets"]["hidden-manifold"]
    hm = HiddenManifold(data_path=conf_data["data_path"], dim=conf_data["dim"])
    x, y = hm.get_train()

    dev = qml.device("default.qubit", wires=config["reservoir"]["num_qubits"])
    qc = get_random_qc(
        n_features=config["reservoir"]["num_qubits"],
        depth=config["reservoir"]["depth"],
        dev=dev,
        seed=config["seed"],
    )
    x_qc = qc(x)

    plot_pca(2, x_qc, y)
