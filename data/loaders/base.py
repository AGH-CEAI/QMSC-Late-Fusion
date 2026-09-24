from abc import ABC, abstractmethod
import numpy.typing as npt


class BaseDataLoader(ABC):
    def __init__(self):
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.x_train_multisource: tuple[npt.NDArray, ...] = tuple()

    @abstractmethod
    def split_to_multisource(self, n_sources: int, n_feat_per_source: int):
        return NotImplemented
