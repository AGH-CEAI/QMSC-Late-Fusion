from abc import ABC, abstractmethod

import numpy.typing as npt


class BaseDataLoader(ABC):
    def __init__(self):
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

    @abstractmethod
    def get_train(self):
        raise NotImplementedError
