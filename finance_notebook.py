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

#%%
from src.acquisition_functions import AcquisitionRelative
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

acq = AcquisitionRelative
bq = AcquisitionAlgorithm(f, models, acq, bounds=bounds, n_init=5, n_iter=5, n_acq_max_starts=2)
bq.run(callback=lambda bq: bq.plot())

# Incorporate gradients (in kernel)
# Model selection comparison (in particular linear)
    # Acquisition function: |mu(x) - L(x)| + beta sigma(x)
    # see sigma regularization ensuring uniform exploration

# See if it works with approximation
# (Implement hyperparameter opt)
# (Implement evaluation metrics: L2)

# Test on 2D
# Test on high-dim manifold (see finance github)

#%%
# Test 2D Finance function 
from mpl_toolkits import mplot3d
fig = plt.figure()
ax = plt.axes(projection='3d')

def f(x, y):
   return 1 / (np.abs(0.5 - x ** 4 - y ** 4) + 0.1)

X = np.linspace(0, 1, 100)
Y = np.linspace(0, 1, 100)
X, Y = np.meshgrid(X, Y)

ax.contour3D(X,Y,f(X,Y), 50, cmap='binary')
