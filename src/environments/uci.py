import math

from sklearn.model_selection import train_test_split

from src.environments import DataSet


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
        import scipy.io
        assert name in cls.DATASETS, "This is not one of the support UCI datasets."
        mat = scipy.io.loadmat('data/uci/{0}/{0}.mat'.format(name))
        return mat['data']

    def __init__(self, name, subset_size=None):
        """
        Keyword Arguments:
            D {int} -- dimensionality of the input space (default: {1})
            subset_size {int} -- Size of the training set (default: {None})
        """

        data = self.__class__.load_raw_df(name)
        self.X = data[:, :-1]
        self.Y = data[:, -1:]

        self.X_train, self.X_test, self.Y_train, self.Y_test = \
            train_test_split(self.X, self.Y, test_size=self.test_percentage, shuffle=True, random_state=42)
        self.X_train, self.X_val, self.Y_train, self.Y_val = \
            train_test_split(self.X_train, self.Y_train, test_size=self.val_percentage, shuffle=True,
                             random_state=42)
        self.X_train, self.Y_train = self.X_train[:subset_size], self.Y_train[:subset_size]


__all__ = ['UCI']
