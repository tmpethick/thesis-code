from abc import ABCMeta, abstractmethod

from src.experiment.config_helpers import ConfigMixin


class DataSet(ConfigMixin, metaclass=ABCMeta):
    @property
    @abstractmethod
    def X_train(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def Y_train(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def X_test(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def Y_test(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def X_val(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def Y_val(self):
        raise NotImplementedError

    @property
    def input_dim(self):
        return self.X_train.shape[-1]
