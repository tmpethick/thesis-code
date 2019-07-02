import math
import numpy as np
from src.environments.dataset import DataSet

import pandas as pd
from ..models.core_models import BaseModel
from sklearn.model_selection import train_test_split

from src.experiment.config_helpers import ConfigMixin

from .option_pricer.MonteCarlo import MonteCarlo


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
            train_test_split(self.X, self.Y, test_size=self.test_percentage, shuffle=True, random_state=42)
        self.X_train, self.X_val, self.Y_train, self.Y_val = \
            train_test_split(self.X_train, self.Y_train, test_size=self.val_percentage, shuffle=True, random_state=42)
        self.X_train, self.Y_train = self.X_train[:subset_size], self.Y_train[:subset_size]


class GrowthModel(ConfigMixin):
    def __init__(self, **kwargs):
        from src.growth_model_GPR.econ import Econ
        from src.growth_model_GPR.ipopt_wrapper import IPOptWrapper
        from src.growth_model_GPR.parameters import Parameters
        from src.growth_model_GPR.nonlinear_solver import NonlinearSolver  # solves opt. problems for terminal VF
        from src.growth_model_GPR.interpolation import Interpolation  # interface to sparse grid library/terminal VF
        from src.growth_model_GPR.postprocessing import \
            PostProcessing  # computes the L2 and Linfinity error of the mode

        self.params = Parameters(**kwargs)
        
        # To mimic the interface of BaseEnvironment
        self.bounds = np.array([[self.params.k_bar, self.params.k_up]] * self.params.n_agents)
        self.input_dim = self.params.n_agents

        self.econ = Econ(self.params)
        self.ipopt = IPOptWrapper(self.params, self.econ)
        self.nonlinear_solver = NonlinearSolver(self.params, self.ipopt)

        self.interpolation = Interpolation(self.params, self.nonlinear_solver)
        self.post = PostProcessing(self.params)

    def sample(self, size):
        return np.random.uniform(self.params.k_bar, self.params.k_up, (size, self.input_dim))

    def evaluate(self, Xtraining, model_prev=None):
        np.random.seed(666)
        #generate sample aPoints
        y = np.zeros(self.params.No_samples, float) # training targets

        if model_prev is None:
            # solve bellman equations at training points
            for i in range(len(Xtraining)):
                print("solving bellman", i)
                y[i] = self.nonlinear_solver.initial(Xtraining[i], self.params.n_agents)[0]
        else:
            for i in range(len(Xtraining)):
                y[i] = self.nonlinear_solver.iterate(Xtraining[i], self.params.n_agents, model_prev)[0]

        # Fit to data using Maximum Likelihood Estimation of the parameters
        y = y[:, np.newaxis]
        return y

    def loop(self, model: BaseModel, callback=lambda i, growth_model, model: None):
        for i in range(self.params.numstart, self.params.numits):
        # terminal value function
            Xtraining = self.sample(size=self.params.No_samples)

            if (i==1):
                print("start with Value Function Iteration")
                y = self.evaluate(Xtraining)
            else:
                print("Now, we are in Value Function Iteration step", i)
                y = self.evaluate(Xtraining, model)

            model.init(Xtraining, y)
            callback(i, self, model)


class GrowthModelCallback(object):
    def __init__(self, growth_model):
        np.random.seed(0)
        self.N = growth_model.params.No_samples_postprocess
        numits = growth_model.params.numits

        self.X_test = growth_model.sample(self.N)
        self.y_pred_prev = np.zeros(self.N)

        self.max_err = np.empty(numits)
        self.RMSE = np.empty(numits)
        self.MAE = np.empty(numits)

    def __call__(self, i, growth_model, model):
        y_pred, _ = model.get_statistics(self.X_test, full_cov=False)

        if i != 0:
            diff = self.y_pred_prev - y_pred
            self.max_err[i-1] = np.amax(np.fabs(diff))
            self.MAE[i-1] = np.average(np.fabs(diff)) / self.N
            self.RMSE[i-1] = np.sqrt(np.sum(np.square(diff)) / self.N)

        self.y_pred_prev = y_pred


from .core import BaseEnvironment

class HestonOptionPricer(BaseEnvironment):
    bounds = None 
    is_expensive = True

    def __init__(self,
        strike = 0.9,
        n_trials = 5000,
        n_steps = 12,
        **kwargs,
        ):

        super().__init__(**kwargs)

        min_vol = 0.08
        max_vol = 1
        min_maturity = 0.1
        max_maturity = 0.9
        self.bounds = np.array([[min_vol, max_vol], [min_maturity, max_maturity]])

        #interval = 0.2,
        #vol = np.arange(0.1, 0.8+interval-10**-10, interval)
        # From 1 month to 6 months
        #time_to_maturity = [0.08,0.17,0.25,0.34,0.42,0.5,1]
        # 0.1,0.3,0.7
        
        # Mean reversion speed
        self.kappa = 3.00 - 0.3*4
        self.strike = strike
        self.n_trials = n_trials
        self.n_steps = n_steps
        self.is_american = False
        self.o_type = 'c'

        # for t in time_to_maturity:
        # [X,y,x1,x2,xxx,YT] = heston_option_pricing_2d(time_to_maturity,strike,n_trials,n_steps,vol,vol[0],vol[-1],'c',False,kappa, self.bounds, grid_test=False)
        # print(X.shape,y.shape,x1.shape,x2.shape,xxx.shape,YT.shape)

    def _call(self, X):
        risk_free_rate = 0.1
        dividend = 0.3
        # Only for BlackScholes model
        volatility = 0
        # stock price
        stock_price = 1

        # Theta =  Gamma(in the paper)
        theta = 3/self.kappa*0.03
        rho = -0.6

        # volatility of volatility
        xi = 0.6

        Y=[]
        for x in X:
            [t,v] = x
            # np.random.seed(1)
            V0 = v**2
            mc = MonteCarlo(S0=stock_price,K=self.strike,T=t,r=risk_free_rate,q=dividend,sigma=volatility,
                        kappa=self.kappa,theta=theta,xi=xi,rho=rho,V0=V0,underlying_process="Heston model")
            price_matrix = mc.simulate(n_trials=self.n_trials,n_steps=self.n_steps,boundaryScheme="Higham and Mao")
            if (self.is_american):
                mc.LSM(option_type=self.o_type,func_list=[lambda x: x**0, lambda x: x],onlyITM=False,buy_cost=0.0,sell_cost=0.0)
            price = mc.MCPricer(option_type=self.o_type, isAmerican=self.is_american) 
            Y.append(price)
        Y = np.array(Y)[:,np.newaxis]
        return Y


    # def plot_model(self, model):
    #     assert self.X1_test is not None, "Plotting only possible for `grid_test=True`"

    #     x1, x2 = np.meshgrid(self.X1_test, self.X2_test)

    #     model.init(self.X_train, self.Y_train)
    #     y_pred, sigma = model.get_statistics(self.X_test)
    #     y_pred = y_pred.reshape(len(self.X1_test),len(self.X2_test))

    #     fig = plt.figure()
    #     ax = fig.gca(projection='3d')
    #     surf = ax.plot_surface(x1, x2, y_pred, cmap=plt.cm.coolwarm,
    #                            linewidth=0, antialiased=False)
    #     plt.xlabel('volatility')
    #     plt.ylabel('time to maturity')
    #     # Add a color bar which maps values to colors.
    #     fig.colorbar(surf, shrink=0.5, aspect=5)
    #     print("MAE, RMSE, MAX =", errors(y_pred.flatten(), YT.flatten()))
    #     return fig

    # def plot(self):
    #     assert self.X1_test is not None, "Plotting only possible for `grid_test=True`"

    #     YT = self.Y_test.reshape(len(self.X1_test),len(self.X2_test))
    #     x1, x2 = np.meshgrid(self.X1_test, self.X2_test)

    #     fig = plt.figure()
    #     ax = fig.gca()
    #     ax = fig.gca(projection='3d')
    #     surf = ax.plot_surface(x1, x2, YT, cmap=plt.cm.coolwarm,
    #                         linewidth=0, antialiased=False)
    #     #surf = ax.contourf(x1, x2, YT, 1000)
    #     plt.xlabel('volatility')
    #     plt.ylabel('time to maturity')
    #     # Add a color bar which maps values to colors.
    #     fig.colorbar(surf, shrink=0.5, aspect=5)
    #     return fig


class AAPL(DataSet):
    X_train = None
    Y_train = None
    X_val = None
    Y_val = None
    X_test = None
    Y_test = None

    prioritized_X_labels = ['days', 'delta']
    Y_label = 'impl_volatility'
    test_percentage = 0.1
    val_percentage = 0.0

    def __init__(self, D=2, subset_size=None):
        """
        Keyword Arguments:
            D {int} -- dimensionality of the input space (default: {1})
            subset_size {int} -- Size of the training set (default: {None})
        """
        self.D = D

        df = pd.read_csv('data/AAPL/fdata.csv')
        self.X = df[self.prioritized_X_labels[:self.D]].values
        self.Y = df[[self.Y_label]].values

        self.X_train, self.X_test, self.Y_train, self.Y_test = \
            train_test_split(self.X, self.Y, test_size=self.test_percentage, shuffle=True, random_state=42)
        self.X_train, self.X_val, self.Y_train, self.Y_val = \
            train_test_split(self.X_train, self.Y_train, test_size=self.val_percentage, shuffle=True, random_state=42)
        self.X_train, self.Y_train = self.X_train[:subset_size], self.Y_train[:subset_size]


class EconomicModel(DataSet):
    X_train = None
    Y_train = None
    X_val = None
    Y_val = None
    X_test = None
    Y_test = None

    prioritized_X_labels = ['days', 'delta']
    Y_label = 'impl_volatility'
    test_percentage = 0.1
    val_percentage = 0.0

    def __init__(self, D=2, output_policy=0, subset_size=None):
        """
        Keyword Arguments:
            D {int} -- dimensionality of the input space (default: {1})
            subset_size {int} -- Size of the training set (default: {None})
        """
        self.D = D

        assert D in [2, 4, 8, 12, 16, 20], "Invalid dimension for EconomicModel"

        if D == 2:
            df = pd.read_csv('data/economic_policies/Output.plt', header=None, delim_whitespace=True)
            self.X = df.loc[:, 0:1].values
            policies = df.loc[:, 3::2].values
        else:
            df = pd.read_csv(f'data/economic_policies/GPR_training-{D}d.txt', header=None, delim_whitespace=True)
            self.X = df.loc[:,3:3+D-1].values
            policies = df.loc[:,3+D+1:].values
        
        self.Y = policies[:,output_policy:output_policy+1]

        self.X_train, self.X_test, self.Y_train, self.Y_test = \
            train_test_split(self.X, self.Y, test_size=self.test_percentage, shuffle=True, random_state=42)
        self.X_train, self.X_val, self.Y_train, self.Y_val = \
            train_test_split(self.X_train, self.Y_train, test_size=self.val_percentage, shuffle=True, random_state=42)
        self.X_train, self.Y_train = self.X_train[:subset_size], self.Y_train[:subset_size]


__all__ = ['SPXOptions', 'HestonOptionPricer', 'AAPL', 'GrowthModel', 'GrowthModelCallback', 'EconomicModel']
