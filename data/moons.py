import os
from typing import Tuple

import numpy as np
import numpy.typing as npt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

from data.base import BaseDataLoader


class Moons(BaseDataLoader):
    def __init__(
        self,
        data_path: str = "datasets/moons/",
        n_samples=200,
        noise=0.2,
        random_state=42,
    ):
        super().__init__()

        self.x_train, self.y_train, self.x_test, self.y_test = self._get_moons(
            data_path, n_samples, noise, random_state
        )

    def get_train(self) -> Tuple[npt.NDArray, npt.NDArray]:
        return self.x_train, self.y_train

    def get_test(self) -> Tuple[npt.NDArray, npt.NDArray]:
        return self.x_test, self.y_test

    # Private methods
    def _build_moons_datafile_name(
        self, n_samples: int, noise: float, random_state: int
    ) -> str:
        """
        Generates the filename for the Moons dataset.

        Args:
            n_samples (int): Total number of points generated.
            noise (float): Standard deviation of Gaussian noise added to the data.
            random_state (int): Seed for random number generation.

        Returns:
            str: The constructed filename.
        """
        noise_str = str(noise).replace(".", "_")
        return f"moons-{n_samples}-{noise_str}-{random_state}.npz"

    def _moons_load(
        self,
        full_filepath: str,
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
        """
        Loads the dataset splits from a local .npz file.

        Args:
            full_filepath (str): The absolute or relative path to the .npz file.

        Returns:
            tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
                A tuple containing (x_train, y_train, x_test, y_test).
        """
        data = np.load(full_filepath)
        return (
            data["x_train"],
            data["y_train"],
            data["x_test"],
            data["y_test"],
        )

    def _moons_generate(
        self, n_samples: int, noise: float, random_state: int
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
        """
        Generates the Moons dataset and splits it into train and test sets.

        Args:
            n_samples (int): Total number of points generated.
            noise (float): Standard deviation of Gaussian noise added to the data.
            random_state (int): Seed for random number generation.

        Returns:
            tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
                A tuple containing (x_train, y_train, x_test, y_test).
        """
        X, y = make_moons(
            n_samples=n_samples, noise=noise, random_state=random_state
        )
        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=0.33, random_state=random_state
        )
        return x_train, y_train, x_test, y_test

    def _get_moons(
        self, data_path: str, n_samples: int, noise: float, random_state: int
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
        """
        Retrieves the Moons dataset, caching it locally to avoid repeated generation.

        Args:
            data_path (str): Path to the directory where the dataset is stored.
            n_samples (int): Total number of points generated.
            noise (float): Standard deviation of Gaussian noise added to the data.
            random_state (int): Seed for random number generation.

        Returns:
            tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
                A tuple containing (x_train, y_train, x_test, y_test).
        """

        filename = self._build_moons_datafile_name(
            n_samples, noise, random_state
        )
        full_filepath = os.path.join(data_path, filename)

        if os.path.exists(full_filepath):
            x_train, y_train, x_test, y_test = self._moons_load(full_filepath)
        else:
            x_train, y_train, x_test, y_test = self._moons_generate(
                n_samples, noise, random_state
            )

            os.makedirs(os.path.dirname(full_filepath), exist_ok=True)
            np.savez(
                full_filepath,
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
            )
        return x_train, y_train, x_test, y_test
