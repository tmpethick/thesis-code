import math
from abc import abstractmethod, ABCMeta

import pandas as pd
from sklearn.model_selection import train_test_split

from src.experiment.config_helpers import ConfigMixin
from src.utils import random_hypercube_samples

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
    test_percentage = 0.2
    val_percentage = 0.2

    @classmethod
    def max_train_size(cls):
        df = cls.load_raw_df()
        size = df.shape[0]
        size = int(math.ceil(size * (1-cls.test_percentage)))
        size = int(math.ceil(size * (1-cls.val_percentage)))
        return size

    @classmethod
    def load_raw_df(cls):
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

        df = pd.DataFrame(data=dict(zip(COLS, data.T)))
        return df

    def __init__(self, D=1, subset_size=None):
        """
        Keyword Arguments:
            D {int} -- dimensionality of the input space (default: {1})
            subset_size {int} -- Size of the training set (default: {None})
        """
        self.D = D

        df = self.__class__.load_raw_df()
        self.X = df[self.prioritized_X_labels[:self.D]].values
        self.Y = df[[self.Y_label]].values

        self.X_train, self.X_test, self.Y_train, self.Y_test = \
            train_test_split(self.X, self.Y, test_size=cls.test_percentage, shuffle=True, random_state=42)
        self.X_train, self.X_val, self.Y_train, self.Y_val = \
            train_test_split(self.X_train, self.Y_train, test_size=cls.val_percentage, shuffle=True, random_state=42)
        self.X_train, self.Y_train = self.X_train[:subset_size], self.Y_train[:subset_size]


import numpy as np



class GrowthModel(object):
    def __init__(self, model, **kwargs):
        from src.growth_model_GPR.econ import Econ
        from src.growth_model_GPR.ipopt_wrapper import IPOptWrapper
        from src.growth_model_GPR.parameters import Parameters
        from src.growth_model_GPR.nonlinear_solver import NonlinearSolver  # solves opt. problems for terminal VF
        from src.growth_model_GPR.interpolation import Interpolation  # interface to sparse grid library/terminal VF
        from src.growth_model_GPR.postprocessing import \
            PostProcessing  # computes the L2 and Linfinity error of the mode

        self.params = Parameters(**kwargs)
        self.bounds = np.array([[self.params.k_bar, self.params.k_up]] * self.params.n_agents)
        self.input_dim = self.params.n_agents

        self.econ = Econ(self.params)
        self.ipopt = IPOptWrapper(self.params, self.econ)
        self.nonlinear_solver = NonlinearSolver(self.params, self.ipopt)

        self.interpolation = Interpolation(self.params, self.nonlinear_solver)
        self.post = PostProcessing(self.params)

        self.model = model

    def loop(self):
        for i in range(self.params.numstart, self.params.numits):
        # terminal value function
            Xtraining = np.random.uniform(self.params.k_bar, self.params.k_up, (self.params.No_samples, self.input_dim))
            if (i==1):
                print("start with Value Function Iteration")
                self.interpolation.GPR_init(i, Xtraining, self.model)

            else:
                print("Now, we are in Value Function Iteration step", i)
                self.interpolation.GPR_iter(i, Xtraining, self.model)

        #======================================================================
        print("===============================================================")
        print(" ")
        print(" Computation of a growth model of dimension ", self.params.n_agents ," finished after ", self.params.numits, " steps")
        print(" ")
        print("===============================================================")
        #======================================================================

        # compute errors
        # avg_err=self.post.ls_error(self.params.n_agents, self.params.numstart, self.params.numits, self.params.No_samples_postprocess)

        #======================================================================
        print("===============================================================")
        print(" ")
        #print " Errors are computed -- see error.txt"
        print(" ")
        print("===============================================================")
        #======================================================================

    #def _loop(model, env, N=1000, T=10):
    #    for i in range(T):
    #        X = random_hypercube_samples(1000, env.bounds)
    #        Y = env(X, model)
    #        model.init(X, Y)

    def __call__(self, X, prob_model):
        """OBS: every call will iterate.
        We assume that self.prob_model is updated with new observations (discarding the old.)
        """
        pass


__all__ = ['DataSet', 'SPXOptions']
