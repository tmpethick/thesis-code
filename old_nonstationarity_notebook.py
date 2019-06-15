#%%
%load_ext autoreload
%autoreload 2

#%%
import numpy as np
from src.plot_utils import plot2D, plot1D, plot_function
from src.algorithms import AcquisitionAlgorithm, bo_plot_callback
from src.utils import random_hypercube_samples, root_mean_square_error
from src.environments import Kink1D, Kink2D, BaseEnvironment
from src.models.core_models import BaseModel
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt

import seaborn as sns
sns.set_style("darkgrid")

#%%
from src.models import DKLGPModel

f = Kink1D()
model = DKLGPModel(gp_dim=2)
test_gp_model(f, model, n_samples=50)
plt.show()

#%%
from src.models.dkl_gp import DKLGPModel

f = Kink2D()
model = DKLGPModel(gp_dim=2, n_iter=50)
test_gp_model(f, model)
plt.show()

#%%
from src.models.lls_gp import LocalLengthScaleGPModel, LocalLengthScaleGPBaselineModel

f = Kink1D()
model = LocalLengthScaleGPModel()
test_gp_model(f, model, n_samples=10)
plt.show()

plot_function(f, model.get_lengthscale, title="Lengthscale", points=model.lls_kernel.X_)
plt.show()

#%%
from src.models.lls_gp import LocalLengthScaleGPModel, LocalLengthScaleGPBaselineModel

f = Kink2D()
model = LocalLengthScaleGPModel(l_samples=30)
test_gp_model(f, model, n_samples=7)
plt.show()

plot_function(f, model.get_lengthscale, title="Lengthscale", points=model.lls_kernel.X_)
plt.show()

#%%
from src.models.lls_gp import LocalLengthScaleGPModel, LocalLengthScaleGPBaselineModel

f = Kink2D()
model = LocalLengthScaleGPBaselineModel()
test_gp_model(f, model)
plt.show()

plot_function(f, model.get_lengthscale, title="Lengthscale")
plt.show()


#%% BO 2D vanilla strategy (scatter plot)

import GPy
from src.acquisition_functions import QuadratureAcquisition
from src.algorithms import AcquisitionAlgorithm
from src.models.core_models import GPModel

f = Kink2D()

kernel = GPy.kern.Matern32(f.input_dim)
noise_prior = 0.01
model = GPModel(kernel=kernel, noise_prior=noise_prior, do_optimize=True, num_mcmc=0)

acq = QuadratureAcquisition
bq = AcquisitionAlgorithm(f, [model], acq, bounds=f.bounds, n_init=2, n_iter=50, n_acq_max_starts=2)
bq.run(callback=bo_plot_callback)

#%% BO

from src.acquisition_functions import QuadratureAcquisition
from src.algorithms import AcquisitionAlgorithm
from src.models.lls_gp import LocalLengthScaleGPModel, LocalLengthScaleGPBaselineModel
from src.models.dkl_gp import DKLGPModel


# Make complete by: 1) constructing BO 2) get config from BO.
# Create ConfigMixin (with uniqueness criteria)

# Construct BO based on config
#   convert strings to models
#   Instansiate models
# Get config from BO (string version)

# Execute on server from python (pickle/unpickle config)


f = Kink1D()
# model = LocalLengthScaleGPModel()
model = DKLGPModel(gp_dim=2, n_iter=1)

acq = QuadratureAcquisition
# Requires at least n_init samples
bq = AcquisitionAlgorithm(f, [model], acq, bounds=f.bounds, n_init=5, n_iter=50, n_acq_max_starts=2)
bq.run(callback=bo_plot_callback)


# config is JSON serilizable
# How to convert ('Class', {})
