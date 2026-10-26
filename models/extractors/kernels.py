import math
from dataclasses import dataclass, field

import pennylane as qml


def circuit_to_kernel(dev, circuit, circuit_params={}):
    """
    Takes quantum device and circuit as arguments and returns kernel function.
    Source: https://pennylane.ai/demos/tutorial_kernels_module
    """

    @qml.qnode(dev)
    def kernel(x1, x2):
        circuit(x1, wires=dev.wires, **circuit_params)
        qml.adjoint(circuit)(x2, wires=dev.wires, **circuit_params)
        return qml.probs(wires=dev.wires)

    return lambda x1, x2: kernel(x1, x2)[0]


def repeated_amplitude_encoding(n_reapeats: int):
    def circuit(x, wires):
        for i in range(n_reapeats):
            qml.AmplitudeEmbedding(x, wires, pad_with=True, normalize=True)

    return circuit


@dataclass
class KernelFactory:
    data_dim: int
    dev_angle: qml.device = field(init=False)
    dev_amp: qml.device = field(init=False)

    def __post_init__(self):
        self.dev_angle = qml.device("lightning.qubit", wires=self.data_dim)
        n_wires_amp_emb = math.ceil(math.log(self.data_dim, 2))
        self.dev_amp = qml.device("lightning.qubit", wires=n_wires_amp_emb)

    @property
    def x_angle(self):
        return circuit_to_kernel(
            self.dev_angle, qml.templates.AngleEmbedding, {"rotation": "X"}
        )

    @property
    def y_angle(self):
        return circuit_to_kernel(
            self.dev_angle, qml.templates.AngleEmbedding, {"rotation": "Y"}
        )

    @property
    def amplitude(self):
        return circuit_to_kernel(
            self.dev_amp, repeated_amplitude_encoding(n_reapeats=1)
        )

    @property
    def r2_amplitude(self):
        return circuit_to_kernel(
            self.dev_amp, repeated_amplitude_encoding(n_reapeats=2)
        )

    @property
    def r3_amplitude(self):
        return circuit_to_kernel(
            self.dev_amp, repeated_amplitude_encoding(n_reapeats=3)
        )

    @property
    def iqp(self):
        return circuit_to_kernel(self.dev_angle, qml.IQPEmbedding)

    @property
    def r2_iqp(self):
        return circuit_to_kernel(
            self.dev_angle, qml.IQPEmbedding, {"n_repeats": 2}
        )

    @property
    def r3_iqp(self):
        return circuit_to_kernel(
            self.dev_angle, qml.IQPEmbedding, {"n_repeats": 3}
        )

    @property
    def as_dict(self):
        return {
            "x_angle": self.x_angle,
            "y_angle": self.y_angle,
            "amplitude": self.amplitude,
            "r2_amplitude": self.r2_amplitude,
            "r3_amplitude": self.r3_amplitude,
            "iqp": self.iqp,
            "r2_iqp": self.r2_iqp,
            "r3_iqp": self.r3_iqp,
        }
