#%%
%load_ext autoreload
%autoreload 2

# %%

import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import GPy

from src.algorithms import AcquisitionAlgorithm, random_hypercube_samples
from src.models import GPModel, RandomFourierFeaturesModel 
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
X = random_hypercube_samples(15, bounds)
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
plt.matshow(covar)
plt.show()

#%% Testing LinearGP
from src.models import GPVanillaLinearModel

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
from src.models import EfficientLinearModel

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
from src.models import GPVanillaModel

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
from src.models import GPVanillaModel

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
from src.models import QuadratureFourierFeaturesModel

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

#%%
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

#%%
# Test Finance function 
from mpl_toolkits import mplot3d
fig = plt.figure()
ax = plt.axes(projection='3d')

def f(x, y):
   return 1 / (np.abs(0.5 - x ** 4 - y ** 4) + 0.1)

X = np.linspace(0, 1, 100)
Y = np.linspace(0, 1, 100)
X, Y = np.meshgrid(X, Y)

ax.contour3D(X,Y,f(X,Y), 50, cmap='binary')

#%% Huuuuge kink (testing kernels)
def f(x):
   return 1 / (10 ** (-4) + x ** 2)

X = np.linspace(-2, 2, 100)
plt.plot(X, f(X))

bounds = np.array([[-2,2]])
X = np.array([[-0.14622061],
       [-1.93486064],
       [ 0.70511943],
       [-0.77367659],
       [-1.40736587],
       [ 1.87965451],
       [-0.07431302],
       [-1.29844724],
       [-1.42601055],
       [ 0.89177627],
       [ 1.86789145],
       [-0.51414124],
       [ 0.15496186],
       [-1.25582743],
       [ 0.73451204]])
Y = f(X)
for kernel in [
   GPy.kern.Exponential(1, variance=200.0),
   GPy.kern.Matern32(1, variance=200.0),
   GPy.kern.Matern52(1, variance=200.0),
   GPy.kern.RBF(1, variance=200.0),]:
   model = GPy.models.GPRegression(X, Y, kernel=kernel)
   model.Gaussian_noise.fix(0)
   model.plot()

#%% Linear kernel

k = GPy.kern.Linear(1) + GPy.kern.White(1, variance=10)
k.plot()
X = np.linspace(0.,1.,500)
X = X[:,None]
C = k.K(X,X)
plt.imshow(C, interpolation='nearest')
plt.show()
mu = np.zeros((500))
Z = np.random.multivariate_normal(mu,C,20)
for i in range(20):
      plt.plot(X[:],Z[i,:])

#%% Plotting the quadratic behavior of the "flipped" diagonal
A = np.zeros(C.shape[0])
for i in range(C.shape[0]):
   j = (C.shape[0] - 1) - i
   A [i] = C[i,j]
plt.plot(A)

#%% Plot any kernel
import numpy as np
import pylab as plt
import GPy
import re

def get_equation(kern):
    match = re.search(r'(math::)(\r\n|\r|\n)*(?P<equation>.*)(\r\n|\r|\n)*', kern.__doc__)
    return '' if match is None else match.group('equation').strip()


# Try plotting sample paths here
k = GPy.kern.LinearFull(input_dim=1, rank=1, kappa=np.array([1000]), W=1000*np.ones((1, 1)))

X = np.linspace(0.,1.,500) # define X to be 500 points evenly spaced over [0,1]
X = X[:,None] # reshape X to make it n*p --- we try to use 'design matrices' in GPy 

mu = np.zeros((500))# vector of the means --- we could use a mean function here, but here it is just zero.
C = k.K(X,X) # compute the covariance matrix associated with inputs X

# Generate 20 separate samples paths from a Gaussian with mean mu and covariance C
Z = np.random.multivariate_normal(mu,C,20)

            
kernel_equation = get_equation(k)
#print kernel_equation
from IPython.display import Math, display
display(Math(kernel_equation))

fig = plt.figure()     # open a new plotting window
plt.subplot(121)
for i in range(3):
      plt.plot(X[:],Z[i,:])

plt.title('{} samples'.format(kernel_name))

plt.subplot(122)
plt.imshow(C, interpolation='nearest')
plt.title('{} covariance'.format(kernel_name))

#%%
