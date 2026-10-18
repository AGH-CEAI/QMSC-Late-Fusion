import math

import pennylane as qml
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

import models.extractors.kernels as kernel_functions
from data.hidden_manifold import HiddenManifold


def svm_precomputed_kernel_accuracy(data, kernel, random_state):
    x_train, y_train = data.get_train()
    x_test, y_test = data.get_test()

    svm = SVC(kernel="precomputed", random_state=random_state)

    kernel_matrix_train = qml.kernels.square_kernel_matrix(x_train, kernel)
    kernel_matrix_test = qml.kernels.square_kernel_matrix(x_test, kernel)

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


def svm_accuracy_check():
    data_dim = 6
    random_state = 42
    data = HiddenManifold(dim=data_dim)

    dev_angle = qml.device("default.qubit", wires=data_dim)

    n_wires_amp_emb = math.ceil(math.log(data_dim, 2))
    dev_amp = qml.device("default.qubit", wires=n_wires_amp_emb)

    angle_kernel = kernel_functions.circuit_to_kernel(
        dev_angle, qml.templates.AngleEmbedding
    )
    amplitude_kernel = kernel_functions.circuit_to_kernel(
        dev_amp, qml.templates.AmplitudeEmbedding
    )

    kernel_ = ["rbf", angle_kernel, amplitude_kernel]

    for kernel in kernel_:
        print(
            svm_callable_kernel_accuracy(
                data=data, kernel=kernel, random_state=random_state
            )
        )
