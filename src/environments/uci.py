import numpy as np
import math

import scipy.io
from sklearn.model_selection import train_test_split, KFold

from src.environments import DataSet

def percentage(X, p=1):
    size = int(len(X) * p)
    return X[:size]


class Precipitation(DataSet):
    X_train = None
    Y_train = None
    X_val = None
    Y_val = None
    X_test = None
    Y_test = None

    def __init__(self):
        mat_hyp = scipy.io.loadmat('data/scalable/precipitation/precipitation3240-hyp.mat')
        mat = scipy.io.loadmat('data/scalable/precipitation/precipitation3240.mat')
        self.X_train = percentage(mat_hyp['Xhyp'])
        self.Y_train = percentage(mat_hyp['yhyp'].astype('float'))
        self.X_post_train = percentage(mat['X'])
        self.Y_post_train = percentage(mat['y'].astype('float'))
        self.X_test = percentage(mat['Xtest'])
        self.Y_test = percentage(mat['ytest'].astype('float'))


class NaturalSound(DataSet):
    X_train = None
    Y_train = None
    X_val = None
    Y_val = None
    X_test = None
    Y_test = None

    def __init__(self, subset_size=None):
        mat = scipy.io.loadmat('data/scalable/sound/audio_data.mat')
        self.X_train = mat['xtrain'].astype('float')
        self.Y_train = mat['ytrain'].astype('float')
        self.X_test = mat['xtest'].astype('float')
        self.Y_test = mat['ytest'].astype('float')

        if subset_size is not None:
            print(subset_size)
            indexes = np.sort(np.random.choice(len(self.X_train), subset_size, replace=False))
            self.X_train = self.X_train[indexes]
            self.Y_train = self.Y_train[indexes]
        # xtest, xtestplot, xtrain, xtrainplot, yfull, ymu, ymug, ymuplotfitc, ymuplotg, ys2, ytest, ytestplot, ytrain, ytrainplot


class UCI(DataSet):
    X_train = None
    Y_train = None
    X_val = None
    Y_val = None
    X_test = None
    Y_test = None

    test_percentage = 0.0
    val_percentage = 0.1

    DATASETS = [
        '3droad',
        'airfoil',
        'autompg',
        'autos',
        'bike',
        'breastcancer',
        'buzz',
        'challenger',
        'concrete',
        'concreteslump',
        'elevators',
        'energy',
        'fertility',
        'forest',
        'gas',
        'houseelectric',
        'housing',
        'keggdirected',
        'keggundirected',
        'kin40k',
        'machine',
        'parkinsons',
        'pendulum',
        'pol',
        'protein',
        'pumadyn32nm',
        'servo',
        'skillcraft',
        'slice',
        'sml',
        'solar',
        'song',
        'stock',
        'tamielectric',
        'wine',
        'yacht',
    ]

    @classmethod
    def max_train_size(cls, name):
        df = cls.load_raw_df(name)
        size = df.shape[0]
        size = int(math.ceil(size * (1 - cls.test_percentage)))
        size = int(math.ceil(size * (1 - cls.val_percentage)))
        return size

    @classmethod
    def load_raw_df(cls, name):
        assert name in cls.DATASETS, "This is not one of the support UCI datasets."
        mat = scipy.io.loadmat('data/uci/{0}/{0}.mat'.format(name))
        return mat['data']

    def __init__(self, name, subset_size=None, kfold=None, kfolds=5):
        """
        :param name:
        :param subset_size: cut-off training set size.
        :param kfold: zero indexed k-fold to use.
        :param kfolds: number of total folds.
        """

        data = self.__class__.load_raw_df(name)
        self.X = data[:, :-1]
        self.Y = data[:, -1:]

        if kfold is not None:
            assert kfold < kfolds
            kf = KFold(n_splits=kfolds, random_state=42, shuffle=True)
            train, test = get_nth(kf.split(self.X), kfold)
            self.X_train, self.X_test, self.Y_train, self.Y_test = \
                self.X[train], self.X[test], self.Y[train], self.Y[test]
        else:
            self.X_train, self.X_test, self.Y_train, self.Y_test = \
                train_test_split(self.X, self.Y, test_size=self.test_percentage, shuffle=True, random_state=42)
            self.X_train, self.X_val, self.Y_train, self.Y_val = \
                train_test_split(self.X_train, self.Y_train, test_size=self.val_percentage, shuffle=True,
                                random_state=42)

        self.X_train, self.Y_train = self.X_train[:subset_size], self.Y_train[:subset_size]


def get_nth(iterable, n):
    for i, x in enumerate(iterable):
        if i == n:
            return x


__all__ = ['UCI', 'NaturalSound', 'Precipitation']
