#%%
%load_ext autoreload
%autoreload 2

# %%

import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import GPy
import scipy

from src.algorithms import AcquisitionAlgorithm, random_hypercube_samples
from src.models import GPModel, RandomFourierFeaturesModel 
from src.acquisition_functions import QuadratureAcquisition

import seaborn as sns
sns.set_style("darkgrid")


def mean_square_error(bq):
    X_line = np.linspace(bq.bounds[0,0], bq.bounds[0,1], 500)[:,None]
    Y = f(X_line)
    Y_hat, covar = bq.models[0].get_statistics(X_line)
    Y_hat = np.mean(Y_hat, axis=0) # average over hyperparameters
    mse = np.sqrt(np.sum(np.square(Y - Y_hat)))
    print("MSE:", mse)
    return mse 


#%% 1D Huuuuge kink (testing kernels)
def f(x):
   return 1 / (10 ** (-4) + x ** 2)

X = np.linspace(-2, 2, 100)
plt.plot(X, f(X))

# Fix the test points
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

# Explore what kernel works
for kernel in [
   GPy.kern.Exponential(1, variance=200.0),
   GPy.kern.Matern32(1, variance=200.0),
   GPy.kern.Matern52(1, variance=200.0),
   GPy.kern.RBF(1, variance=200.0),]:
   model = GPy.models.GPRegression(X, Y, kernel=kernel)
   model.Gaussian_noise.fix(0)
   model.plot()

#%% Uniform sampling
from src.acquisition_functions import AcquisitionModelMismatch
from src.algorithms import AcquisitionAlgorithm
from src.models import GPModel

bounds = np.array([[-2,2]])
def f(x):
   return 1 / (10 ** (-4) + x ** 2)

kernel = GPy.kern.Matern32(1)
noise_prior = 0.01
model = GPModel(kernel=kernel, noise_prior=noise_prior, do_optimize=True, num_mcmc=0)

acq = QuadratureAcquisition
bq = AcquisitionAlgorithm(f, [model], acq, bounds=bounds, n_init=2, n_iter=100, n_acq_max_starts=2)
#bq.run(callback=lambda bq: bq.plot())
bq.run()
bq.plot()

plt.hist(bq.models[0].X)
mse_vanilla = mean_square_error(bq)

#%% Model Mismatch sampling approach
from src.acquisition_functions import AcquisitionModelMismatch
from src.algorithms import AcquisitionAlgorithm
from src.models import GPModel

bounds = np.array([[-2,2]])
def f(x):
   return 1 / (10 ** (-4) + x ** 2)

kernel = GPy.kern.Matern32(1)
noise_prior = 0.01
model = GPModel(kernel=kernel, noise_prior=noise_prior, do_optimize=True, num_mcmc=0)

exp_kernel = GPy.kern.Exponential(1)
linear_comparison_model = GPModel(kernel=exp_kernel, noise_prior=0.01)

models = [model, linear_comparison_model]

acq = AcquisitionModelMismatch(*models, beta=0)
bq = AcquisitionAlgorithm(f, models, acq, bounds=bounds, n_init=2, n_iter=100, n_acq_max_starts=2)
#bq.run(callback=lambda bq: bq.plot())
bq.run()
bq.plot()

mse_model_mismatch = mean_square_error(bq)

#%%

print("MSE model mismatch:", mse_model_mismatch) # 169.9740896452874
print("MSE vanilla:", mse_vanilla) # 10075.7768671319

#%%
# Test 2D Finance function 
from mpl_toolkits import mplot3d
fig = plt.figure()
ax = plt.axes(projection='3d')

def f(x):
   return 1 / (np.abs(0.5 - x[...,0] ** 4 - x[...,1] ** 4) + 0.1)

X = np.linspace(0, 1, 300)
Y = np.linspace(0, 1, 300)
X, Y = np.meshgrid(X, Y)
Z = f(np.stack((X,Y), axis=-1))

ax.contour3D(X,Y,Z, 50, cmap='binary')

#%% 2D vanilla strategy (scatter plot)
from src.acquisition_functions import AcquisitionModelMismatch
from src.algorithms import AcquisitionAlgorithm
from src.models import GPModel

bounds = np.array([[0,1],[0,1]])
def f(x):
   y = 1 / (np.abs(0.5 - x[...,0] ** 4 - x[...,1] ** 4) + 0.1)
   return y[...,None]

kernel = GPy.kern.Matern32(2)
noise_prior = 0.01
model = GPModel(kernel=kernel, noise_prior=noise_prior, do_optimize=True, num_mcmc=0)

acq = QuadratureAcquisition
bq = AcquisitionAlgorithm(f, [model], acq, bounds=bounds, n_init=2, n_iter=100, n_acq_max_starts=2)
#bq.run(callback=lambda bq: bq.plot())
bq.run()


#%%
bq.plot()

#%% 2D model mismatch strategy (scatter plot)
# Nb: Points seem to be concentrated around two areas/points.
from src.acquisition_functions import AcquisitionModelMismatch
from src.algorithms import AcquisitionAlgorithm
from src.models import GPModel

bounds = np.array([[0,1],[0,1]])
def f(x):
   y = 1 / (np.abs(0.5 - x[...,0] ** 4 - x[...,1] ** 4) + 0.1)
   return y[...,None]

kernel = GPy.kern.RBF(2)
noise_prior = 0.01
model = GPModel(kernel=kernel, noise_prior=noise_prior, do_optimize=True, num_mcmc=0)

exp_kernel = GPy.kern.Exponential(2)
linear_comparison_model = GPModel(kernel=exp_kernel, noise_prior=0.01)

models = [model, linear_comparison_model]

acq = AcquisitionModelMismatch(*models, beta=0)
bq = AcquisitionAlgorithm(f, models, acq, bounds=bounds, n_init=2, n_iter=100, n_acq_max_starts=5)
#bq.run(callback=lambda bq: bq.plot())
bq.run()

#%%
bq.plot()

#%%
import seaborn as sns
sns.scatterplot(bq.X[...,0], bq.X[...,1])


# Sample near kinks
   # We need to somehow specify that we are bad at kinks. 
   # Gradients does not capture this. (and wont work for kinks)
   # Sample at curvature. Will sample at "arupt" changes. This might work.
   # Try Home backed Acquisition function.
