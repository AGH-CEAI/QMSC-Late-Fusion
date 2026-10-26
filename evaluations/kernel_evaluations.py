import math

import numpy as np
import pennylane as qml
from scipy.linalg import sqrtm
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

import models.extractors.kernels as kernel_functions
from data.hidden_manifold import HiddenManifold


#
def svm_precomputed_kernel_accuracy(data, kernel, random_state):
    x_train, y_train = data.get_train()
    x_test, y_test = data.get_test()

    svm = SVC(kernel="precomputed", random_state=random_state)

    kernel_matrix_train = qml.kernels.square_kernel_matrix(x_train, kernel)
    kernel_matrix_test = qml.kernels.kernel_matrix(x_test, x_train, kernel)

    svm.fit(kernel_matrix_train, y_train)
    preds = svm.predict(kernel_matrix_test)

    return accuracy_score(y_true=y_test, y_pred=preds)


def svm_callable_kernel_accuracy(data, kernel, random_state):
    x_train, y_train = data.get_train()
    x_test, y_test = data.get_test()

    svm = SVC(kernel=kernel, random_state=random_state)

    svm.fit(x_train, y_train)
    preds = svm.predict(x_test)

    return accuracy_score(y_true=y_test, y_pred=preds)


def compute_g(K_classical, K_quantum, eps=1e-7):
    """
    Compute geometric difference g between classical and quantum kernels.
    Formula: g = sqrt( || sqrt(K_quantum) @ inv(K_classical) @ sqrt(K_quantum) || )

    Source: https://pennylane.ai/demos/tutorial_huang_geometric_kernel_difference
    """
    N = K_classical.shape[0]

    # Regularize and invert K_classical
    Kc_reg = K_classical + eps * np.eye(N)
    Kc_inv = np.linalg.inv(Kc_reg)

    # Compute the square root of the quantum kernel
    sqrt_Kq = sqrtm(K_quantum)
    # Construct M = sqrt(Kq) @ Kc⁻¹ @ sqrt(Kq)
    M = sqrt_Kq @ Kc_inv @ sqrt_Kq

    # g = sqrt(max eigenvalue of M)
    max_eigval = np.max(np.linalg.eigvalsh(M))
    return np.sqrt(np.maximum(max_eigval, 0.0))


#############################################


def svm_accuracy_check():
    data_dim = 6
    random_state = 42
    data = HiddenManifold(dim=data_dim)

    kernels = kernel_functions.KernelFactory(data_dim=data_dim)

    for name, kernel in kernels.as_dict.items():
        print(
            name,
            svm_precomputed_kernel_accuracy(
                data=data, kernel=kernel, random_state=random_state
            ),
        )
