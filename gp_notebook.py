#%%
%load_ext autoreload
%autoreload 2

# %%

import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import GPy

from src.algorithms import AcquisitionAlgorithm, random_hypercube_samples
from src.models.models import GPModel, RandomFourierFeaturesModel
from src.acquisition_functions import QuadratureAcquisition

# Plotting
import seaborn as sns
sns.set_style("darkgrid")

#%%

def f(x):
   return np.sinc(x)

kernel = GPy.kern.RBF(1, ARD=False)
kernel.lengthscale.set_prior(GPy.priors.LogGaussian(0, 1))
kernel.variance.set_prior(GPy.priors.LogGaussian(0, 1))
noise_prior = 0.01 # GPy.priors.LogGaussian(0, 0.00001)

bounds = np.array([[-2,2]])
model = GPModel(kernel=kernel, noise_prior=noise_prior, do_optimize=True, num_mcmc=0)
acq = QuadratureAcquisition
bq = AcquisitionAlgorithm(f, model, acq, bounds=bounds, n_init=5, n_iter=5)
bq.run()

#%%
################### Plot ####################

bq.plot()

#%% GPy linear kernel

def f(x):
   return np.sinc(x)

bounds = np.array([[0,1]])
Y = f(X)

kernel = GPy.kern.Linear(1) # + GPy.kern.Bias(1)
model = GPModel(kernel=kernel, noise_prior=0.01, do_optimize=False)
X_line = np.linspace(bounds[0,0], bounds[0,1], 500)[:, None]
model.fit(X, Y)
model.plot(X_line)
plt.plot(X_line, f(X_line))

mean, covar = model.get_statistics(X_line)
mean, covar = mean[0], covar[0,:,:,0]
plt.matshow(model.kernel.K(X_line, X_line))
X = random_hypercube_samples(15, bounds)
plt.matshow(covar)
plt.show()

#%% Testing LinearGP
from src.models.models import GPVanillaLinearModel

def f(x):
   return np.sinc(x)

bounds = np.array([[0,1]])
X = random_hypercube_samples(15, bounds)
Y = f(X)
d = X.shape[-1]
model = GPVanillaLinearModel(noise=0.01)
X_line = np.linspace(bounds[0,0], bounds[0,1], 500)[:, None]
model.fit(X, Y)
model.plot(X_line)
plt.plot(X_line, f(X_line))

plt.matshow(model.kernel(X_line, X_line))
mean, covar = model.get_statistics(X_line)
plt.matshow(covar)
plt.show()
   
#%%
# Testing EfficientLinearGP (uses low rank)
# (should be equivalent to GPVanillaLinearModel)
from src.models.models import EfficientLinearModel

def f(x):
   return np.sinc(x)

bounds = np.array([[0,1]])
X = random_hypercube_samples(15, bounds)
Y = f(X)
d = X.shape[-1]
model = EfficientLinearModel(noise=0.01, n_features=d)
X_line = np.linspace(bounds[0,0], bounds[0,1], 500)[:, None]
model.fit(X, Y)
model.plot(X_line)
plt.plot(X_line, f(X_line))

mean, covar = model.get_statistics(X_line)
plt.matshow(model.kernel(X_line, X_line))
plt.matshow(covar)
plt.show()
   

#%% GPy
from src.models.models import GPVanillaModel

def f(x):
   return np.sinc(x)

bounds = np.array([[0,1]])
X = random_hypercube_samples(15, bounds)
Y = f(X)

model = GPModel(kernel=GPy.kern.RBF(1, lengthscale=0.2), noise_prior=0.01, do_optimize=False)
X_line = np.linspace(bounds[0,0], bounds[0,1], 500)[:, None]
model.fit(X, Y)
model.plot(X_line)
plt.plot(X_line, f(X_line))

mean, covar = model.get_statistics(X_line)
mean, covar = mean[0], covar[0,:,:,0]
plt.matshow(model.kernel.K(X_line, X_line))
plt.matshow(covar)
plt.show()


#%% Home backed GP
from src.models.models import GPVanillaModel

def f(x):
   return np.sinc(x)

bounds = np.array([[0,1]])
X = random_hypercube_samples(15, bounds)
Y = f(X)

model = GPVanillaModel(gamma=0.2, noise=0.01)
X_line = np.linspace(bounds[0,0], bounds[0,1], 500)[:, None]
model.fit(X, Y)
model.plot(X_line)
plt.plot(X_line, f(X_line))

mean, covar = model.get_statistics(X_line)
plt.matshow(model.kernel(X_line, X_line))
plt.matshow(covar)
plt.show()

#%%
# Testing RFF

def f(x):
   return np.sinc(x)

bounds = np.array([[0,1]])
X = random_hypercube_samples(15, bounds)
Y = f(X)
for n_features in [1000]:
   model = RandomFourierFeaturesModel(gamma=0.2, n_features=n_features)
   X_line = np.linspace(bounds[0,0], bounds[0,1], 500)[:, None]
   model.fit(X, Y)
   model.plot(X_line)
   plt.plot(X_line, f(X_line))

   mean, covar = model.get_statistics(X_line)
   plt.matshow(model.kernel(X_line))
   plt.matshow(covar)
   plt.show()

#%%
# Testing QFF
from src.models.models import QuadratureFourierFeaturesModel

def f(x):
   return np.sinc(x)

bounds = np.array([[-2,2]])
X = random_hypercube_samples(15, bounds)
Y = f(X)
for n_features in [300]:
   model = QuadratureFourierFeaturesModel(gamma=0.2, n_features=n_features)
   X_line = np.linspace(bounds[0,0], bounds[0,1], 500)[:, None]
   model.fit(X, Y)
   model.plot(X_line)
   plt.plot(X_line, f(X_line))

   mean, covar = model.get_statistics(X_line)
   plt.matshow(model.kernel(X_line,X_line))
   plt.matshow(covar)
   plt.show()

#%% 
# Test exponential
def f(x):
   return np.sinc(x)

# kernel = GPy.kern.RBF(1, ARD=False)
# kernel.lengthscale.set_prior(GPy.priors.LogGaussian(0, 1))
# kernel.variance.set_prior(GPy.priors.LogGaussian(0, 1))
kernel = GPy.kern.Exponential(1, variance=1.0)

bounds = np.array([[-2,2]])
X = random_hypercube_samples(15, bounds)
Y = f(X)
model = GPy.models.GPRegression(X, Y, kernel=kernel)
model.Gaussian_noise.fix(0)
model.plot()

#%% Sample paths from exponential
X_test = np.linspace(0, 1, 100).reshape(-1, 1)
Y_post_test = model.posterior_samples_f(X_test, full_cov=True, size=3)
simY, simMse = model.predict(X_test)

plt.plot(X_test, Y_post_test)
plt.plot(X, Y, 'xk', markersize=4)
plt.plot(X_test, simY - 3 * simMse ** 0.5, '--g')
plt.plot(X_test, simY + 3 * simMse ** 0.5, '--g')

#%% 
# Test Matern
def f(x):
   return np.sinc(x)

# kernel = GPy.kern.RBF(1, ARD=False)
# kernel.lengthscale.set_prior(GPy.priors.LogGaussian(0, 1))
# kernel.variance.set_prior(GPy.priors.LogGaussian(0, 1))
kernel = GPy.kern.Matern32(1, variance=1.0)

bounds = np.array([[-2,2]])
X = random_hypercube_samples(15, bounds)
Y = f(X)
model = GPy.models.GPRegression(X, Y, kernel=kernel)
model.Gaussian_noise.fix(0)
model.plot()

