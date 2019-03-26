
#%%
%load_ext autoreload
%autoreload 2

from GPyOpt.acquisitions import AcquisitionBase
from GPyOpt.util.general import get_quantiles

class AcquisitionCB(AcquisitionBase):
    """
    GP-Confidence Bound acquisition function

    :param model: GPyOpt class of model
    :param space: GPyOpt class of domain
    :param optimizer: optimizer of the acquisition. Should be a GPyOpt optimizer
    :param cost_withGradients: function
    :param jitter: positive value to make the acquisition more explorative

    .. Note:: does not allow to be used with cost

    """

    analytical_gradient_prediction = False

    def __init__(self, model, space, optimizer=None, cost_withGradients=None, exploration_weight=2):
        self.optimizer = optimizer
        super(AcquisitionCB, self).__init__(model, space, optimizer)
        if cost_withGradients is not None:
            print('The set cost function is ignored! LCB acquisition does not make sense with cost.')  

    def _compute_acq(self, x):
        """
        Computes the GP-Lower Confidence Bound 
        """
        m, s = self.model.predict(x)   
        f_acqu = s
        return f_acqu

    def _compute_acq_withGradients(self, x):
        """
        Computes the GP-Lower Confidence Bound and its derivative
        """
        m, s, dmdx, dsdx = self.model.predict_withGradients(x) 
        f_acqu = s       
        df_acqu = dsdx
        return f_acqu, df_acqu


# --- Load GPyOpt
import GPyOpt
import GPy
from GPyOpt.methods import BayesianOptimization
from GPyOpt.models import GPModel
from GPyOpt.optimization.acquisition_optimizer import AcquisitionOptimizer
from GPyOpt.core.task.space import Design_space
import numpy as np

# --- Define your problem
#def f(x): return (6*x-2)**2*np.sin(12*x-4)
f_true = GPyOpt.objective_examples.experiments2d.sixhumpcamel()
f = f_true.f
bounds =np.array(f_true.bounds)
bounds_gpyopt =[{'name': 'var_1', 'type': 'continuous', 'domain': f_true.bounds[0]},
         {'name': 'var_2', 'type': 'continuous', 'domain': f_true.bounds[1]}]

#%%
# Finance problem
def f(x):
   y = 1 / (np.abs(0.5 - x[...,0] ** 4 - x[...,1] ** 4) + 0.1)
   return y[...,None]

bounds = np.array([[0,1],[0,1]])
bounds_gpyopt =[{'name': 'var_1', 'type': 'continuous', 'domain': bounds[0]},
         {'name': 'var_2', 'type': 'continuous', 'domain': bounds[1]}]

#%% ------------ GPyOpt without hyperopt
space = Design_space(bounds_gpyopt, None)
kern = GPy.kern.Matern52(2)
model = GPModel(max_iters=0, kernel=kern, exact_feval=True)
acquisition_optimizer = AcquisitionOptimizer(space, 'lbfgs', model=model)
acq = AcquisitionCB(model, space, optimizer=acquisition_optimizer)
myBopt = BayesianOptimization(f=f, domain=bounds_gpyopt, model=model, acquisition=acq)
myBopt.run_optimization(max_iter=100)
myBopt.plot_acquisition()

#%% ------------ GPyOpt with hyperopt
space = Design_space(bounds_gpyopt, None)
kern = GPy.kern.Matern52(2)
model = GPModel(max_iters=5, kernel=kern, exact_feval=True, verbose=False)
acquisition_optimizer = AcquisitionOptimizer(space, 'lbfgs', model=model)
acq = AcquisitionCB(model, space, optimizer=acquisition_optimizer)
myBopt = BayesianOptimization(f=f, domain=bounds_gpyopt, model=model, acquisition=acq, verbosity=False)
myBopt.run_optimization(max_iter=100)
myBopt.plot_acquisition()

#%% ------------ Home-backed
from src.acquisition_functions import QuadratureAcquisition
from src.algorithms import AcquisitionAlgorithm
from src.models import GPModel

kernel = GPy.kern.Matern32(2)
noise_prior = 0.01
model = GPModel(kernel=kernel, noise_prior=noise_prior, do_optimize=True, num_mcmc=0)

acq = QuadratureAcquisition
bq = AcquisitionAlgorithm(f, [model], acq, bounds=bounds, n_init=2, n_iter=100, n_acq_max_starts=5)
bq.run()
import seaborn as sns
sns.scatterplot(bq.X[...,0], bq.X[...,1])


#%% ------------ Home-backed Model Mismatch!
from src.acquisition_functions import AcquisitionModelMismatch
from src.algorithms import AcquisitionAlgorithm
from src.models import GPModel

kernel = GPy.kern.Matern32(2)
noise_prior = 0.01
model = GPModel(kernel=kernel, noise_prior=noise_prior, do_optimize=True, num_mcmc=0)

exp_kernel = GPy.kern.Exponential(2)
linear_comparison_model = GPModel(kernel=exp_kernel, noise_prior=0.01)

models = [model, linear_comparison_model]

acq = AcquisitionModelMismatch
bq = AcquisitionAlgorithm(f, models, acq, bounds=bounds, n_init=2, n_iter=100, n_acq_max_starts=2)
bq.run()
import seaborn as sns
sns.scatterplot(bq.X[...,0], bq.X[...,1])
