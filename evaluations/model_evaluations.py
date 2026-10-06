import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from data.hidden_manifold import HiddenManifold


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
