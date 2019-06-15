
#%%
%load_ext autoreload
%autoreload 2

# --- Load GPyOpt
import GPyOpt
import GPy
from GPyOpt.methods import BayesianOptimization
from GPyOpt.models import GPModel
from GPyOpt.optimization.acquisition_optimizer import AcquisitionOptimizer
from GPyOpt.core.task.space import Design_space
from src.acquisition_functions import GPyOptQuadratureAcquisition
import numpy as np

from src.environments import Sixhumpcamel, to_gpyopt_bounds, Kink2D

#%%

# --- Define your problem
#def f(x): return (6*x-2)**2*np.sin(12*x-4)
f = Sixhumpcamel()
bounds_gpyopt = to_gpyopt_bounds(f.bounds)

#%%

f = Kink2D()
bounds_gpyopt = to_gpyopt_bounds(f.bounds)

#%% ------------ GPyOpt with hyperopt
space = Design_space(bounds_gpyopt, None)
kern = GPy.kern.Matern52(2)
model = GPModel(max_iters=5, kernel=kern, exact_feval=True, verbose=True)
acquisition_optimizer = AcquisitionOptimizer(space, 'lbfgs', model=model)
acq = GPyOptQuadratureAcquisition(model, space, optimizer=acquisition_optimizer)
myBopt = BayesianOptimization(f=f, domain=bounds_gpyopt, model=model, acquisition=acq, verbosity=True)
myBopt.run_optimization(max_iter=100)
myBopt.plot_acquisition()


#%%------------ GPyOpt with hyperopt and model mismatch

space = Design_space(bounds_gpyopt, None)
kern = GPy.kern.Matern52(2)
model = GPModel(max_iters=5, kernel=kern, exact_feval=True, verbose=False)
acquisition_optimizer = AcquisitionOptimizer(space, 'lbfgs', model=model)
acq = GPyOptQuadratureAcquisition(model, space, optimizer=acquisition_optimizer)
myBopt = BayesianOptimization(f=f, domain=bounds_gpyopt, model=model, acquisition=acq, verbosity=False)
myBopt.run_optimization(max_iter=100, verbosity=True)
myBopt.plot_acquisition()


#%% ------------ Home-backed
from src.acquisition_functions import QuadratureAcquisition
from src.algorithms import AcquisitionAlgorithm, bo_plot_callback
from src.models.core_models import GPModel

kernel = GPy.kern.Matern32(2)
noise_prior = 0.01
model = GPModel(kernel=kernel, noise_prior=noise_prior, do_optimize=True, num_mcmc=0)

acq = QuadratureAcquisition
bq = AcquisitionAlgorithm(f, [model], acq, bounds=f.bounds, n_init=2, n_iter=100, n_acq_max_starts=5)
bq.run(callback=bo_plot_callback)
import seaborn as sns
sns.scatterplot(bq.X[...,0], bq.X[...,1])


#%% ------------ Home-backed Model Mismatch!
from src.acquisition_functions import AcquisitionModelMismatch
from src.algorithms import AcquisitionAlgorithm
from src.models.core_models import GPModel

kernel = GPy.kern.Matern32(2)
noise_prior = 0.01
model = GPModel(kernel=kernel, noise_prior=noise_prior, do_optimize=True, num_mcmc=0)

exp_kernel = GPy.kern.Exponential(2)
linear_comparison_model = GPModel(kernel=exp_kernel, noise_prior=0.01)

models = [model, linear_comparison_model]

acq = AcquisitionModelMismatch
bq = AcquisitionAlgorithm(f, models, acq, bounds=f.bounds, n_init=2, n_iter=100, n_acq_max_starts=2)
bq.run(callback=bo_plot_callback)
import seaborn as sns
sns.scatterplot(bq.X[...,0], bq.X[...,1])

# take 1D tværsnit
#

