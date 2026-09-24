import os
import numpy as np
import numpy.typing as npt
import pennylane as qp

from data.loaders.base import BaseDataLoader


class HiddenManifold(BaseDataLoader):
    def __init__(self, data_path: str, dim: int = 10, diff: bool = False):
        super().__init__()

        self.x_train, self.y_train, self.x_test, self.y_test = (
            self._get_manifold(data_path, dim, diff)
        )

    def split_to_multisource(
        self, n_sources: int = 2, n_feat_per_source: int = 4
    ):
        self.x_train_multisource = tuple(
            self.x_train[
                :, i * n_feat_per_source : (i + 1) * n_feat_per_source
            ]
            for i in range(n_sources)
        )

    # Private methods
    def _build_manifold_datafile_name(self, dim: int, diff: bool) -> str:
        """
        Generates the filename for the Hidden Manifold dataset.

        Args:
            dim (int): The varying dimension.
            diff (bool): Indicates if the manifold dimension (True) or input dimension (False) varies.

        Returns:
            str: The constructed filename.
        """
        if diff:
            name = f"hidden-manifold-diff-{dim}.npz"
        else:
            name = f"hidden-manifold-{dim}.npz"
        return name

    def _manifold_load(
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

    def _manifold_download(
        self, dim: int, diff: bool
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
        """
        Downloads the benchmark dataset "Hidden Manifold".

        Source: https://pennylane.ai/datasets/hidden-manifold

        Args:
            dim (int): The varying dimension.
            diff (bool): If True, input dimension d=10 is constant and `dim` sets manifold dimension m in [2, ..., 20].
                        If False, manifold dimension m=6 is constant and `dim` sets input dimension d in [2, ..., 20].

        Returns:
            tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
                A tuple containing (x_train, y_train, x_test, y_test).
        """
        [ds] = qp.data.load("other", name="hidden-manifold")
        train = ds.diff_train[str(dim)] if diff else ds.train[str(dim)]
        test = ds.diff_test[str(dim)] if diff else ds.test[str(dim)]
        return (
            train["inputs"],
            train["labels"],
            test["inputs"],
            test["labels"],
        )

    def _get_manifold(
        self, data_path: str, dim: int = 10, diff: bool = False
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
        """
        Retrieves the Hidden Manifold dataset, caching it locally to avoid repeated downloads.

        Args:
            dim (int, optional): The varying dimension (defaults to 10).
            diff (bool, optional): Determines whether manifold or input dimension varies (defaults to False).

        Raises:
            ValueError: If `dim` is not in the allowed range [2, 20].

        Returns:
            tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
                A tuple containing (x_train, y_train, x_test, y_test).
        """
        if dim not in range(2, 21):
            raise ValueError(
                f"dim needs to be in range 2–20, but instead it is {dim}."
            )
        filename = self._build_manifold_datafile_name(dim=dim, diff=diff)
        full_filepath = "datasets/hidden-manifold/" + filename
        if os.path.exists(full_filepath):
            x_train, y_train, x_test, y_test = self._manifold_load(
                full_filepath
            )
        else:
            x_train, y_train, x_test, y_test = self._manifold_download(
                dim, diff
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
