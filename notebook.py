#%%
%load_ext autoreload
%autoreload 2

# %%

import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import GPy

from src.algorithms import AcquisitionAlgorithm, random_hypercube_samples
from src.models import GPModel, FourierFeatureModel
from src.acquisition_functions import QuadratureAcquisition

#%%

def f(x):
   return np.sinc(x)

#%%

kernel = GPy.kern.RBF(1, ARD=False)
kernel.lengthscale.set_prior(GPy.priors.LogGaussian(0, 1))
kernel.variance.set_prior(GPy.priors.LogGaussian(0, 1))
noise_prior = GPy.priors.LogGaussian(0, 0.00001)

bounds = np.array([[-2,2]])
model = GPModel(kernel=kernel, noise_prior=noise_prior, do_optimize=True, num_mcmc=0)
acq = QuadratureAcquisition
bq = AcquisitionAlgorithm(f, model, acq, bounds=bounds, n_init=5, n_iter=5)

#%%
bq.run()

#%%
################### Plot ####################

bq.plot()


#%%
# Testing RFF

def f(x):
   return np.sinc(x)

for n_features in [10, 100, 500]:
   X = random_hypercube_samples(10, bounds)
   Y = f(X)
   bounds = np.array([[-2,2]])
   model = FourierFeatureModel(n_features=n_features)
   X_line = np.linspace(bounds[0,0], bounds[0,1], 500)[:, None]
   model.fit(X, Y)
   model.plot(X_line)

   mean, covar = model.get_statistics(X_line)
   plt.matshow(model.kernel(X,X))
   plt.matshow(covar)
   plt.show()

#%%

