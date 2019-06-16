

# Y is always the mid-quote (column 2 in the data set).


# Strike, tau, S, r, q, volume, openinterest, date-lastradedate, ask, bid
from abc import abstractmethod, ABCMeta

import pandas as pd
from sklearn.model_selection import train_test_split

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


class SPXOptions(DataSet):
    X_train = None
    Y_train = None
    X_val = None
    Y_val = None
    X_test = None
    Y_test = None

    Y_label = 'midquote'
    prioritized_X_labels = [
        'strike',
        'tau',
        'S',
        'r',
        'q',
        'volume',
        'OpenInterest',
        'LastTradeDate',
        'ask',
        'bid',
    ]

    def __init__(self, D=1, subset_size=None):
        self.D = D

        import scipy.io
        mat = scipy.io.loadmat('data/OptionData_0619/optionsSPXweekly_96_17.mat')
        data = mat['optionsSPX']

        COLS = [
            "strike",
            "midquote",
            "tau",
            "r",
            "q",
            "bid",
            "ask",
            "IV",
            "volume",
            "OpenInterest",
            "Delta",
            "Gamma",
            "Vega",
            "Theta",
            "LastTradeDate",
            "Callput",
            "Date",
            "S"]

        df = pd.DataFrame(data=dict(zip(COLS, data.T)))[:subset_size]
        self.X = df[self.prioritized_X_labels[:self.D]].values
        self.Y = df[[self.Y_label]].values

        self.X_train, self.X_test, self.Y_train, self.Y_test = \
            train_test_split(self.X, self.Y, test_size=0.20, shuffle=True, random_state=42)
        self.X_train, self.X_val, self.Y_train, self.Y_val = \
            train_test_split(self.X_train, self.Y_train, test_size=0.20, shuffle=True, random_state=42)


#%%

__all__ = ['DataSet', 'SPXOptions']
