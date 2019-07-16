import math
import numpy as np
import pickle
import copy

import os
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from src.models.core_models import BaseModel, SaveMixin
from src.models import DKLGPModel
from src.experiment.config_helpers import ConfigMixin
from src.utils import average_at_locations
from src.environments.dataset import DataSet
from src.utils import construct_2D_grid, call_function_on_grid
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

    def sample(self, size, rnd=np.random):
        return rnd.uniform(self.params.k_bar, self.params.k_up, (size, self.input_dim))

    def evaluate(self, Xtraining, model_prev=None):
        Y = np.zeros((self.params.No_samples, 1), float) # training targets

        if model_prev is None:
            # solve bellman equations at training points
            for i in range(len(Xtraining)):
                Y[i] = self.nonlinear_solver.initial(Xtraining[i], self.params.n_agents)[0]
        else:
            for i in range(len(Xtraining)):
                Y[i] = self.nonlinear_solver.iterate(Xtraining[i], self.params.n_agents, model_prev)[0]

        return Y

    def loop(self, model: BaseModel, callback=lambda i, growth_model, model: None):
        for i in range(self.params.numstart, self.params.numits):
            rnd = np.random.RandomState(666)
            Xtraining = self.sample(size=self.params.No_samples, rnd=rnd)

            if (i==1):
                print("start with Value Function Iteration")
                Y = self.evaluate(Xtraining)
            else:
                print("Now, we are in Value Function Iteration step", i)
                Y = self.evaluate(Xtraining, model)

            # TODO: should be ensure that the model hyperparameters are reinstanciated?
            model.init(Xtraining, Y)
            model.save(self.params.model_dir + str(i))
            callback(i, self, model)

        self.post.ls_error()


class GrowthModelDistributed(GrowthModel):
    """A MPI enabled variant of the GrowthModel.
    """
    def loop(self, mother_model: BaseModel, callback=lambda i, growth_model, model: None):
        from mpi4py import MPI
        
        self.MPI = MPI
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        mother_path = os.path.join(self.params.output_dir, 'mother_model')
        mother_model.save(mother_path)

        for i in range(self.params.numstart, self.params.numits):
            if (i==1):
                print("start with Value Function Iteration")
                X, Y = self.mpi_evaluate(model_prev=None)
            else:
                print("Now, we are in Value Function Iteration step", i)
                X, Y = self.mpi_evaluate(model_prev=model)

            # Distribute the model across all nodes.
            if self.rank == 0:
                model = SaveMixin.load(mother_path)
                model.init(X, Y)

                model.save(self.params.model_dir + str(i))

                callback(i, self, model)

            self.comm.Barrier()

            if self.rank != 0:
                model = SaveMixin.load(self.params.model_dir + str(i))
        
        if self.rank == 0:
            self.post.ls_error()

    def mpi_evaluate(self, model_prev=None):
        comm = self.comm
        rank = self.rank
        size = self.size
        
        aPoints=0
        iNumP1_buf=np.zeros(1, int)
        iNumP1=iNumP1_buf[0]
        aVals_gathered=0
        
        if rank==0:
            k_range=np.array([self.params.k_bar, self.params.k_up])

            ranges=np.empty((self.params.n_agents, 2))


            for i in range(self.params.n_agents):
                ranges[i]=k_range

            rnd = np.random.RandomState(666)
            aPoints = self.sample(size=self.params.No_samples, rnd=rnd)
            
            iNumP1=aPoints.shape[0]
            iNumP1_buf[0]=iNumP1
            aVals_gathered=np.empty((iNumP1, 1))
        
        comm.Barrier()
        comm.Bcast(iNumP1_buf, root=0)
        iNumP1=iNumP1_buf[0]
        
        nump=iNumP1//size
        r=iNumP1 % size
        
        if rank<r:
            nump+=1
        
        displs_scat=np.empty(size)
        sendcounts_scat=np.empty(size)
        
        displs_gath=np.empty(size)
        sendcounts_gath=np.empty(size)
        
        for i in range(r):
            displs_scat[i]=i*(1+iNumP1//size)*self.params.n_agents
            sendcounts_scat[i]=(1+iNumP1//size)*self.params.n_agents
            
            displs_gath[i]=i*(1+iNumP1//size)
            sendcounts_gath[i]=(1+iNumP1//size)
            
        for i in range(r, size):
            displs_scat[i]=(r+i*(iNumP1//size))*self.params.n_agents
            sendcounts_scat[i]=(iNumP1//size)*self.params.n_agents
            
            displs_gath[i]=r+i*(iNumP1//size)
            sendcounts_gath[i]=(iNumP1//size)

        local_aPoints=np.empty((nump, self.params.n_agents))
        
        comm.Scatterv([aPoints, sendcounts_scat, displs_scat, self.MPI.DOUBLE], local_aPoints)
        
        local_aVals = np.empty([nump, 1])
        
        #with open("comparison1.txt", 'w') as file:
        if model_prev is None:
            print(f"Node {rank} processing {nump} points")
            for iI in range(nump):
                local_aVals[iI]=self.nonlinear_solver.initial(local_aPoints[iI], self.params.n_agents)[0]
                # v_and_rank=np.array([[local_aVals[iI], rank]])
                # to_print=np.hstack((local_aPoints[iI].reshape(1,self.params.n_agents), v_and_rank))
                # np.savetxt(file, to_print, fmt='%2.16f')
        else:
            print(f"Node {rank} processing {nump} points")
            for iI in range(nump):
                local_aVals[iI]=self.nonlinear_solver.iterate(local_aPoints[iI], self.params.n_agents, model_prev)[0]
                # v_and_rank=np.array([[local_aVals[iI], rank]])
                # to_print=np.hstack((local_aPoints[iI].reshape(1,self.params.n_agents), v_and_rank))
                # np.savetxt(file, to_print, fmt='%2.16f')
                
        comm.Gatherv(local_aVals, [aVals_gathered, sendcounts_gath, displs_gath, self.MPI.DOUBLE])

        return aPoints, aVals_gathered


class GrowthModelCallback(object):
    def __init__(self, growth_model, verbose=False):
        np.random.seed(0)
        self.N = growth_model.params.No_samples_postprocess
        numits = growth_model.params.numits

        self.X_test = growth_model.sample(self.N)
        self.y_pred_prev = np.zeros(self.N)

        self.max_err = np.empty(numits)
        self.RMSE = np.empty(numits)
        self.MAE = np.empty(numits)
        
        self.verbose = verbose

    def __call__(self, i, growth_model, model):
        y_pred, _ = model.get_statistics(self.X_test, full_cov=False)

        # TODO: not meaningful before y_pred_prev is populated using growth_model.evaluate()
        # make sure that the model is not updated.

        if i != 0:
            diff = self.y_pred_prev - y_pred
            self.max_err[i-1] = np.amax(np.fabs(diff))
            self.MAE[i-1] = np.average(np.fabs(diff)) / self.N
            self.RMSE[i-1] = np.sqrt(np.sum(np.square(diff)) / self.N)

            if self.verbose:
                print('max_err:', self.max_err[i-1])
                print('MAE:', self.MAE[i-1])
                print('RMSE:', self.RMSE[i-1])
        
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

    def plot(self, N=200):
        XY, X, Y = construct_2D_grid(self.bounds, N=N)
        Z = call_function_on_grid(self.noiseless, XY)

        fig = plt.figure()
        ax = fig.gca()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(X, Y, Z[...,0], cmap=plt.cm.coolwarm,
                            linewidth=0, antialiased=False)
        #surf = ax.contourf(x1, x2, YT, 1000)
        plt.xlabel('volatility')
        plt.ylabel('time to maturity')
        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)
        return fig


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

        # Assumes task is to predict the mean well.
        self.Y_test = average_at_locations(self.X_test, self.Y_test)
        self.X_test = self.X_test[:2500]
        self.Y_test = self.Y_test[:2500]


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


__all__ = ['SPXOptions', 'HestonOptionPricer', 'AAPL', 'GrowthModelDistributed', 'GrowthModel', 'GrowthModelCallback', 'EconomicModel']
