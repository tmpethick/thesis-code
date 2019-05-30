#%%
%load_ext autoreload
%autoreload 2

from runner import notebook_run, notebook_run_CLI, notebook_run_server, execute

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")

from src.utils import *
from src.plot_utils import *
from src.kernels import *
from src.models.models import *
from src.models.dkl_gp import *
from src.models.lls_gp import *
from src.environments import *
from src.acquisition_functions import *
from src.algorithms import *

# AS + active sampling (learning inverse mapping A^T as well)
# DKL (learn inverse mapping as well)
# DKL will QFF help if we can have more samples?

#%%


from src.plot_utils import plot1D
from src.models.models import QuadratureFourierFeaturesModel, RandomFourierFeaturesModel
from src.utils import random_hypercube_samples
from src.environments import Kink1D

f = Kink1D()
n_features = 300

X1 = np.linspace(-1,-0.5, 3)
X2 = np.linspace(-0.5, 0.5, 20)
X3 = np.linspace(0.5, 1, 3)
X = np.concatenate([X1, X2, X3])[:,None]
Y = f(X)


#%%

class Step(BaseEnvironment):
    bounds = np.array([[0,1]])

    def __call__(self, X):
        noise = np.random.normal(0, 0.1, X.shape)
        return np.where(X > 0.5, 0, 5) + noise

#%%
from src.models.models import RandomFourierFeaturesModel, RFFRBF, GPModel

# f = Step()
# X = np.random.uniform(f.bounds[0,0],f.bounds[0,1], 100)[:, None]
f = Sinc()
X = np.random.uniform(f.bounds[0,0],f.bounds[0,1], 20)[:, None]

Y = f(X)

noise = 0.001
variance = 0.2
opt = False
kernel = RFFRBF(lengthscale=0.8, variance=variance)
model = RandomFourierFeaturesModel(kernel, noise=noise, n_features=500, do_optimize=opt)
kernel = GPyRBF(1, lengthscale=0.8, variance=variance)
model.init(X,Y)
plot_model(model, f)
print(model.kernel_.theta)
plt.show()

model = GPModel(kernel, noise_prior=noise, do_optimize=opt)
model.init(X,Y)
plot_model(model, f)
plt.show()

#%% Noise makes a big difference. (1e-5 forces it to go through point which makes it oscillated.)
# Oscillation is still a problem for RFF with noise though.
# These VFF guys have similar problem: http://gpss.cc/gpa17/slides/VFF.pdf
from src.models.models import GPModel
from src.kernels import GPyRBF, RFFMatern, RFFRBF
from src.models.models import RandomFourierFeaturesModel

noise = 0.1
kernel = RFFRBF(lengthscale=0.1)
model = RandomFourierFeaturesModel(kernel, noise=noise, n_features=n_features)
model.init(X, Y)
plot1D(model, f)
plt.title("{} with $\sigma^2=${}".format(model, noise))

noise = 0.1
kernel = GPyRBF(1, lengthscale=0.1)
model = GPModel(kernel, noise_prior=noise)
model.init(X, Y)
plot1D(model, f)
plt.title("{} with $\sigma^2=${}".format(model, noise))

noise = 1e-5
kernel = RFFRBF(lengthscale=0.1)
model = RandomFourierFeaturesModel(kernel, noise=noise, n_features=n_features)
model.init(X, Y)
plot1D(model, f)
plt.title("{} with $\sigma^2=${}".format(model, noise))

noise = 1e-5
kernel = GPyRBF(1, lengthscale=0.1)
model = GPModel(kernel, noise_prior=noise)
model.init(X, Y)
plot1D(model, f)
plt.title("{} with $\sigma^2=${}".format(model, noise))


#%% Fixing RFF oscillation (would increased variance help?)
# For ExactGP small variance fairs better when noise is (very!)small.
n_features = 500

for var in [0.1, 1, 100, 1000]:
    # kernel = RFFRBF(lengthscale=0.1, variance=var)
    # model = RandomFourierFeaturesModel(kernel, noise=1, n_features=n_features)
    kernel = GPyRBF(1, lengthscale=0.1, variance=var)
    model = GPModel(kernel, noise_prior=1e-5)

    model.init(X, Y)
    plot1D(model, f)
    plt.show()

#%% Fixing RFF oscillation (would increased variance help?)
# Variance will even make the mean vary more wildly for RFF.
# Can this be explained by the expression for the mean?
n_features = 500

for var in [1, 1000]:
    kernel = RFFRBF(lengthscale=0.1, variance=var)
    model = RandomFourierFeaturesModel(kernel, noise=0.1, n_features=n_features)

    model.init(X, Y)
    plot1D(model, f)
    plt.show()

#%% Show oscillation outside observation with RFF on smooth function.
# sin(x) with changing lengthscale.

from src.environments import IncreasingOscillation

f = IncreasingOscillation()
half_way = f.bounds[0,0] + (f.bounds[0,1] - f.bounds[0,0]) / 2
X1 = np.random.uniform(f.bounds[0,0], half_way, 5)
X2 = np.random.uniform(half_way, f.bounds[0,1], 50)
X = np.concatenate([X1, X2])[:, None]
Y = f(X)

n_features = 500

kernel = GPyRBF(1, lengthscale=0.008, variance=1)
model = GPModel(kernel, noise_prior=1e-5, do_optimize=True)

model.init(X, Y)
plot1D(model, f)
plt.show()

kernel = RFFRBF(lengthscale=0.008, variance=1)
model = RandomFourierFeaturesModel(kernel, noise=1e-5, n_features=n_features)

model.init(X, Y)
plot1D(model, f)
plt.show()


#%%--------------------- Using true Hessian -------------------------

#%% Generate Hessian of Kink2D

from sympy import *
from sympy.utilities.lambdify import lambdify, implemented_function,lambdastr
x, y, z = symbols('x y z', real=True)
z = 1 / (abs(Rational(1,2) - x ** 4 - y ** 4) + Rational(1,10))
z.diff(x)
hess = [simplify(z.diff(x0).diff(x1)) for x0 in [x,y] for x1 in [x,y]]

# Get Hessian
for h in hess:
    print(lambdastr(x, h))

h = lambdify(x, hess[0])

# Get fro norm
H = Matrix(hess)
H_fro = simplify(H.norm())
H_fro

#%% We cannot even integrate the function value...

# z = 1 / (abs(Rational(1,2) - x ** 4 - y ** 4) + Rational(1,10))
# X = integrate(z, (x, 0,1))
# X = integrate(X, (y, 0,1))
# X

#%% Sample according to Hess with MCMC 
# Explain why we can't integrate easily (normalize)

import numpy as np
import emcee

from src.environments import Kink2D

f = Kink2D()

def lnprob(x):
    # if x[0]<0 or x[0]>1 or x[1]<0 or x[1]>1:
    #     return 0

    X = np.array([x])
    hess = f.hessian(X)
    hess_norm = np.linalg.norm(hess, ord='fro', axis=(-2, -1))
    return np.log(hess_norm)

ndim, nwalkers = 2, 100
p0 = [np.random.rand(ndim) for i in range(nwalkers)]

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)
sampler.run_mcmc(p0, 1000)

#%%
# TODO: Bounds
import seaborn as sns

X = sampler.flatchain[:,0]
Y = sampler.flatchain[:,1]

plt.figure()
sns.kdeplot(X, Y, cmap="Reds", shade=True, bw='silverman')
plt.show()

X_bounded = X[(X >= 0) & (X <= 1) & (Y >= 0) & (Y <= 1)]
Y_bounded = Y[(X >= 0) & (X <= 1) & (Y >= 0) & (Y <= 1)]
#%%

plt.figure()
sns.kdeplot(X_bounded, Y_bounded, cmap="Reds", shade=True, bw='silverman', clip=[[0,1],[0,1]])
plt.show()
print(X_bounded.shape)

#%% Choose subset of data an ensure that it is still approx the same dist

# Reshape to (N, D)
data = np.stack([X_bounded, Y_bounded], axis=-1)
N, D = data.shape

# We have 31k points (pick random subset)
n = 4000
data_sub = data[np.random.choice(N, size=n, replace=False)]

# Plot to validate that it still represents the distribution
plt.figure()
sns.kdeplot(data_sub[:, 0], data_sub[:,1], cmap="Reds", shade=True, bw='silverman', clip=[[0,1],[0,1]])
plt.show()

#%% Prepare duck typing for RMSE testing 

from src.utils import root_mean_square_error
from src.models import GPModel

class DuckTypeModel(object):
    def __init__(self, model):
        self.model = model

    def _get_statistics(self, X, full_cov=True):
        mean, var = self.model.predict(X, full_cov=False)
        return mean, var


#%% Plot with sparseGP using 1000 inducing points
# What out. The reason it might do well in undersampled region is because *the prior* is correct (=0). Because of the high lengthscale it would not be able to do well in that area. 
# So we have to be careful when evaluating the performance based on Kink2D. 
# Two distinct problems:
# 1) For stationary undersampled regions rely on good prior. (since samples will only change local behaviour if lengthscale is short)
# 2) RFF oscillation in undersample regions (still around prior mean)
import GPy

kernel = GPy.kern.Matern32(2)

n = 4000
data_sub = data[np.random.choice(N, size=n, replace=False)]
#data_sub = random_hypercube_samples(n, f.bounds)

# uniform vs curvature
# noise vs noiseless
# kernel..

X = data_sub
Y = f(X)

#num_inducing = 1000
#model = GPy.models.SparseGPRegression(X, Y, kernel=kernel, num_inducing=num_inducing)
model = GPy.models.GPRegression(X, Y, kernel=kernel)
model.Gaussian_noise.fix(1e-5)
model.randomize()
model.optimize()
model.plot()
plt.show()
print(kernel.lengthscale)

model_wrapper = DuckTypeModel(model)
RMSE = root_mean_square_error(model_wrapper, f, rand=True)
print(RMSE)

# What to answer?
# Is non-stationary kernel required?
# Does it help scaling to many observations?
# Would active sampling be beneficial?


#%% ExactGP
# (Kink2D + 20) is a problem for small lengthscale + few samples in high lengthscale area.
# Normalizing would help. But wouldn't solve the inherint problem with stationarity.


from src.environments import Kink2DShifted
from src.plot_utils import plot2D
from src.models.models import GPModel
from src.kernels import GPyRBF
from src.utils import random_hypercube_samples

f = Kink2DShifted()

n = 1000

data_sub = data[np.random.choice(N, size=n, replace=False)]
#data_sub = random_hypercube_samples(n, f.bounds)
X = data_sub
Y = f(X)

# Hyperparams taken from Opt of ExactGP.
kernel = GPyRBF(2, lengthscale=0.05)
model = GPModel(kernel, noise_prior=1e-5)
model.init(X, Y)
plot2D(model, f)

# Calc error
RMSE = root_mean_square_error(model, f, rand=True)
print(RMSE)


#%% Test the error
# More data does not seem to help much. (sparse Matern 1k-inducing with 31k points does as well as 4k). The inducing points might be the bottleneck. Both have test-error N(2.5, ~0.1).
# RBF fails completely (even with 31k points). error of ~190. With optimization it can even explode to 50k...

model_wrapper = DuckTypeModel(model)
RMSE = root_mean_square_error(model_wrapper, f, rand=True)
print(RMSE)

#%% RFF does reasonable on uniform sample (but fails horribly if not uniform!!)
# It is extremely sensitive to lengthscale/lengthscale: [0.03, ..0.06]

from src.plot_utils import plot2D
from src.models.models import QuadratureFourierFeaturesModel, RandomFourierFeaturesModel, RFFMatern, RFFRBF
from src.utils import random_hypercube_samples

n_features = 300
n = 10000

#data_sub = data[np.random.choice(N, size=n, replace=False)]
data_sub = random_hypercube_samples(n, f.bounds)
X = data_sub
Y = f(X)

# TODO: fix for very low noise
# TODO: find lengthscale (MLE?)

# Hyperparams taken from Opt of ExactGP.
kernel = RFFRBF(lengthscale=0.05)
#kernel = RFFMatern(lengthscale=0.86, nu=3/2)
model = RandomFourierFeaturesModel(kernel, noise=1e-5, n_features=n_features)
#model = QuadratureFourierFeaturesModel(lengthscale=0.2, noise=1e-2, n_features=n_features)
model.init(X, Y)
plot2D(model, f)

# Calc error
RMSE = root_mean_square_error(model, f, rand=True)
print(RMSE)


#%% Test QFF
from src.plot_utils import plot2D
from src.models.models import QuadratureFourierFeaturesModel, RandomFourierFeaturesModel, RFFMatern, RFFRBF
from src.utils import random_hypercube_samples

n_features = 300
n = 10000

#data_sub = data[np.random.choice(N, size=n, replace=False)]
data_sub = random_hypercube_samples(n, f.bounds)
X = data_sub
Y = f(X)

model = QuadratureFourierFeaturesModel(lengthscale=0.05, noise=1e-5, n_features=n_features)
model.init(X, Y)
plot2D(model, f)

# Calc error
RMSE = root_mean_square_error(model, f, rand=True)
print(RMSE)

#%% Plot with RFF (using lengthscale learned with scalableGP...)

# (Use learned Matern lengthscale as RBF seems to fit badly. Maybe its not that bad actually?)
# Can we "fix" the wrong model (wrong legnthscale as well) by simply overrun it with observations?

from src.plot_utils import plot2D
from src.models.models import QuadratureFourierFeaturesModel, RandomFourierFeaturesModel, RFFMatern, RFFRBF
from src.utils import random_hypercube_samples

n_features = 1000
n = 1000

#data_sub = data[np.random.choice(N, size=n, replace=False)]
data_sub = random_hypercube_samples(n, f.bounds)
X = data_sub
Y = f(X)

# TODO: fix for very low noise
# TODO: find lengthscale (MLE?)

# Hyperparams taken from Opt of ExactGP.
kernel = RFFRBF(lengthscale=0.05)
#kernel = RFFMatern(lengthscale=0.86, nu=3/2)
model = RandomFourierFeaturesModel(kernel, noise=1e-5, n_features=n_features)
#model = QuadratureFourierFeaturesModel(lengthscale=0.05, noise=1e-5, n_features=n_features)
model.init(X, Y)
plot2D(model, f)

# Calc error
RMSE = root_mean_square_error(model, f, rand=True)
print(RMSE)


#%% Try KISS

import math
import torch
import gpytorch
from matplotlib import pyplot as plt
from src.models.dkl_gp import DKLGPModel

class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, grid_bounds=None):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        
        #grid_size = gpytorch.utils.grid.choose_grid_size(train_x)
        grid_size = 100
        
        kernel = gpytorch.kernels.RBFKernel()
        
        # TODO: Remove! This are app specific priors
        kernel.initialize(lengthscale=0.05)
        likelihood.initialize(noise=1e-5)

        self.mean_module = gpytorch.means.ConstantMean()
        #self.covar_module = gpytorch.kernels.ProductStructureKernel(
        self.covar_module = gpytorch.kernels.GridInterpolationKernel(
                gpytorch.kernels.ScaleKernel(
                    kernel,
                ), 
                grid_size=grid_size, 
                grid_bounds=grid_bounds,
                num_dims=2
            )
        #, num_dims=2)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class KISSGP(DKLGPModel):
    def __init__(self, grid_bounds=None, **kwargs):
        self.grid_bounds = grid_bounds
        return super(KISSGP, self).__init__(**kwargs)

    def _fit(self, X, Y, Y_dir=None, is_initial=True):
        n, d = X.shape

        self.X_torch = torch.Tensor(X)
        self.Y_torch = torch.Tensor(Y[:, 0])

        # noise=torch.ones(train_x.shape[0]) * 1e-5
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.model = GPRegressionModel(self.X_torch, self.Y_torch, self.likelihood, )#grid_bounds=self.grid_bounds)

        self.model.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam([
            {'params': self.model.parameters()},
        ], lr=0.1)

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        def train():
            training_iterations = 100
            for i in range(training_iterations):
                optimizer.zero_grad()
                output = self.model(self.X_torch)
                loss = -mll(output, self.Y_torch)
                loss.backward()
                print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
                optimizer.step()

        # See dkl_mnist.ipynb for explanation of this flag
        with gpytorch.settings.use_toeplitz(True):
            train()

from src.utils import random_hypercube_samples
from src.plot_utils import plot2D
from src.utils import root_mean_square_error

# What interpolation strategy are we using?

n = 10000
data_sub = data[np.random.choice(N, size=n, replace=False)]
#data_sub = random_hypercube_samples(n, f.bounds)
X = data_sub
Y = f(X)

model = KISSGP(grid_bounds=f.bounds)
model.init(X, Y)

#%%
plot2D(model, f)

# Calc error
RMSE = root_mean_square_error(model, f, rand=True)
print(RMSE)
print(model.model.covar_module.base_kernel.base_kernel.lengthscale)


#%%
import GPy

kernel = GPy.kern.Matern32(2)
num_inducing = 1000
n = 1000

X = data[np.random.choice(N, size=n, replace=False)]
Y = f(X)

#model = GPy.models.SparseGPRegression(X, Y, kernel=kernel, num_inducing=num_inducing)
model = GPy.models.GPRegression(X, Y, kernel=kernel)
model.Gaussian_noise.fix(1e-5)

# Matern
# 0.7505329011996106, 0.8688686224350041 (bad, with randomize())
# 0.5510624061828673, 1.168184301821048 (good)
#kernel.variance = 1
#kernel.lengthscale = 0.86

#RBF
kernel.lengtscale = 0.04945546
#model.randomize()
model.optimize()
model.plot()
plt.show()
print(kernel.lengthscale)

model_wrapper = DuckTypeModel(model)
RMSE = root_mean_square_error(model_wrapper, f, rand=True)
print(RMSE)


#%% ------------------ Kink2D (uniform) --------------------
# Neither shifting nor RBF/Matern seems to make a difference for n=3000.
# In all instances ~10.

from src.plot_utils import plot2D
from src.utils import random_hypercube_samples
from src.models.models import GPModel
from src.environments import Kink2D, Kink2DShifted
from src.kernels import GPyMatern32, GPyRBF

f = Kink2DShifted()

n = 3000

data_sub = random_hypercube_samples(n, f.bounds)
X = data_sub
Y = f(X)

# Hyperparams taken from Opt of ExactGP.
#kernel = GPyRBF(2, lengthscale=0.05)       
kernel = GPyMatern32(2, lengthscale=0.05)  
model = GPModel(kernel, noise_prior=1e-5)
model.init(X, Y)
plot2D(model, f)

# Calc error
RMSE = root_mean_square_error(model, f, rand=True)
print(RMSE)

#%% DKLModel
from src.plot_utils import plot2D
from src.utils import random_hypercube_samples
from src.models.dkl_gp import DKLGPModel
from src.environments import Kink2D, Kink2DShifted, Kink1D
from src.kernels import GPyMatern32, GPyRBF
from src.plot_utils import plot_model

#f = Kink1D()
#f = Kink2DShifted()
f = Kink2D()
n = 3000

data_sub = random_hypercube_samples(n, f.bounds)
X = data_sub
Y = f(X)

model = DKLGPModel(n_iter=50, nn_kwargs={'layers': (1000, 500, 50, 1)})
model.init(X, Y)
plot_model(model, f)

# Calc error
RMSE = root_mean_square_error(model, f, rand=True)
print(RMSE)

model.plot_features(f)

#%% ---------------------- Hessian -------------------------

#%% Hessian of Kink2D

from src.environments import Kink2D

f = Kink2D()
f.plot()
#f.plot_derivative()
X = np.random.uniform(0,1,(10,2))
f.plot_curvature()

#%% Using samples from Hessian as dist

from src.environments import Kink2D

f = Kink2D()
X = np.random.uniform(0,1,(10,2))
f.hessian


#%% One sample on Kink1D (MLE is not "optimal"!)
from src.models import zero_mean_unit_var_normalization

model = run.interactive_stash['model']
acq = run.interactive_stash['acq']
f = run.interactive_stash['f']

# Does as intended with lengthscale = 0.05 (but this is not ML optimal!)
model.do_optimize = False
model.kernel.lengthscale = 0.05
model.kernel.variance = 3

X = np.linspace(0,1, 10)[:, None]
Y = np.ones(10)[:, None]
Y[5,0] = 10
Y, _, _ = zero_mean_unit_var_normalization(Y)

model.init(X, Y)
X_line = np.linspace(0,1,100)[:,None]
model.plot(X_line)


#%%--------------------- Other -------------------------

#%% Lengthscale effect on Curvature acq for two regions. (model setup)

run = execute(config_updates={
    'obj_func': {
        'name': 'IncreasingOscillation',
    },
    'model': {
        'name': 'GPModel',
        'kwargs': {
            'kernel': {
                'name': 'GPyRBF',
                'kwargs': {
                    'lengthscale': 1
                }
            },
            'noise_prior': None,
            'do_optimize': True,
            'num_mcmc': 0,
            'subsample_interval': 10,
            'step_size': 1e-1,
            'leapfrog_steps': 20,
            'n_burnin': 100,
        }
    },
    'acquisition_function': {
        'name': 'CurvatureAcquisition',
    },
    'gp_samples': 10
})

#%% Lengthscale effect on Curvature acq for two regions.
import numpy as np
import matplotlib.pyplot as plt

model = run.interactive_stash['model']
acq = run.interactive_stash['acq']
f = run.interactive_stash['f']

X = np.array([[0.28808442],
              [0.71803477],
              [0.70700961],
              [0.45427829],
              [0.89139108],
              [0.1685069 ],
              [0.79268465],
              [0.49095078],
              [0.93522034],
              [0.85808317]])
Y = f(X)


# Independant of beta high curvature maximizes... 
# What if we had many more points...

def plot_lengthscale(l, plot_acq=True):
    acq.use_var = True

    if l is None:
        model.do_optimize = True
    else:
        model.do_optimize = False
        model.kernel.lengthscale = l

    model.gpy_model.Gaussian_noise.fix(1e-20)
    model.init(X, Y)

    X_line = np.linspace(0, 1, 100)[:, None]
    plt.plot(X_line, f(X_line))
    model.plot(X_line)
    if plot_acq:
        plt.plot(X_line, acq(X_line))
    plt.show()


for l in [0.001, 0.01, 0.1, 1, 10, 100]:
    acq.beta = 100
    plot_lengthscale(l)


#%% We have to ensure we do not get stuck in high-curvature region.
# Adding points moves it to eventually explore

acq.beta = 100

plot_lengthscale(0.01)

X1 = np.random.uniform(0.6, 1, 30)[:, None]
X = np.concatenate((X, X1), axis=0)
Y = f(X)

plot_lengthscale(0.01)

#%%

X_line = np.linspace(0,1, 100)
f_X = X_line * np.sin(30*X_line)
f_Y = np.array([])
#f_X = np.sin(3*X_line[:len(X_line)//2])
#f_Y = np.sin(30*X_line[len(X_line)//2:])

plt.plot(X_line, np.concatenate((f_X, f_Y)))

#%%

run = execute(config_updates={
    'obj_func': {
        'name': 'IncreasingAmplitude',
    },
    'model': {
        'name': 'GPModel',
        'kwargs': {
            'kernel': {
                'name': 'GPyRBF',
                'kwargs': {
                    'lengthscale': 1
                }
            },
            'noise_prior': None,
            'do_optimize': True,
            'num_mcmc': 0,
        }
    },
    'acquisition_function': {
        'name': 'CurvatureAcquisition',
    },
    'gp_samples': 30
})

#%% Lengthscale of a sin


run = execute(config_updates={
    'obj_func': {
        'name': 'Sin',
    },
    'model': {
        'name': 'GPModel',
        'kwargs': {
            'kernel': {
                'name': 'GPyRBF',
                'kwargs': {
                    'lengthscale': 0.1,
                }
            },
            'noise_prior': 1e-20,
            'do_optimize': True,
            'num_mcmc': 0,
        }
    },
    'acquisition_function': {
        'name': 'CurvatureAcquisition',
    },
    'gp_samples': 100
})
print(run.interactive_stash['model'].kernel.lengthscale)

#%%

run = execute(config_updates={
    'obj_func': {
        'name': 'Kink1D',
    },
    'model': {
        'name': 'GPModel',
        'kwargs': {
            'kernel': {
                'name': 'GPyRBF',
                'kwargs': {
                    'lengthscale': 0.001,
                }
            },
            'noise_prior': 1e-20,
            'do_optimize': False,
            'num_mcmc': 0,
        }
    },
    'acquisition_function': {
        'name': 'CurvatureAcquisition',
    },
    'gp_samples': 100
})
print(run.interactive_stash['model'].kernel.lengthscale)


#%% Problem with small lengthscale. 
# 1) Mean peaks even in flat area (leading to high curvature)
# 2) Regression L2 would do terrible in X region (high lengthscale region).

acq.beta = 0
plot_lengthscale(0.01, plot_acq=True)
plot_lengthscale(0.01, plot_acq=False)
plot_lengthscale(None, plot_acq=False)

#%%
# --------------- TODO: ----------------

# Testing sampling pattern:
# 2D sin (with hessian)
# See that sampling matches hessian
    # Run BO and plot samples against hessian. (kde? requires too many samples...)

# Methods:
    # sample w.r.t. to the true hessian
    # Sample based on GP hessian.
    # Use derivative info to fit GP. (how does it improve hessian estimate?)
    # Use hessian variance?

# Ultimately:
    # Hessian for sampling
    # DKL vs. AS + Lengthscale
    # QFF/KISS for scaling both
    # Use derivatives
# Requires:
    # Implement hessian for DKL and lengthscale

# TODO:
# Compare: uniform, GP variance, hessian
# Find functions for which the sampling stategy is especially effective
# Find models for which the sampling strategy is especially effective
# Think about cool applications

# Implement for DKL
# Implement for Lengthscale
# How do we verify behavior for high-dim? (construct?)

#%%
# First see that hessian is correctly implementing by
# plotting Hessian of GP.
import numpy as np
import matplotlib.pyplot as plt

from src.environments import Sinc
from src.models import GPModel
from src.kernels import GPyRBF
from src.utils import random_hypercube_samples

f = Sinc()
X = random_hypercube_samples(50, f.bounds)
Y = f(X)

kernel = GPyRBF(f.input_dim)
model = GPModel(kernel, do_optimize=True)
model.init(X,Y)
X_line = np.linspace(f.bounds[0,0], f.bounds[0,1], 100)[:,None]
mean, var = model.get_statistics(X_line)
jac, jac_var = model.predict_jacobian(X_line)
hess, hess_var = model.predict_hessian(X_line)

model.plot(X_line)
plt.plot(X_line, mean[0])
plt.legend()
plt.show()

plt.plot(X_line, f.derivative(X_line), label="True jac")

plt.plot(X_line, jac[:,0], label="Jac")

# jac_simple = np.zeros(len(X_line))
# for i in range(len(X_line)):
#     p, v = model.predict_jacobian_1sample(np.array([X_line[i]]))
#     jac_simple[i] = p

# Scaling issue (no biggy)
# plt.plot(X_line, jac_simple, label="Jac 1 sample")
plt.legend()
plt.show()

plt.plot(X_line, hess[:, 0, 0], label="hessian")
plt.plot(X_line, f.hessian(X_line), label="True hessian")
plt.legend()
plt.show()


#%% Curvature hyperparameter experiment (how does the distance R between points influence the variance at the midpoint)

import numpy as np
import matplotlib.pyplot as plt

from src.models import GPModel
from src.kernels import GPyRBF


for l in [1, 10, 1000]:
    # l = 10
    kern = GPyRBF(1, lengthscale=l)
    model = GPModel(kernel=kern, noise_prior=1e-10)
    
    N = 100
    D = np.zeros(N)
    D_var = np.zeros(N)

    for i, R in enumerate(np.arange(1, N+1)):
        x1 = np.array([0])
        x2 = np.array([1])
        y1 = R * x1
        y2 = R * x2
        model.init(np.array([y1, y2]), np.array([[0], [0]]))
        d = np.array([y1 + 0.5 * (y2 - y1)])
        mu, var = model.get_statistics(d, full_cov=False)

        if i % 10 == 0:
            X_line = np.linspace(y1[0], y2[0], 100)[:, None]
            # model.plot(X_line)
            # plt.show()

        D[i] = R
        D_var[i] = var[0,0,0]

    plt.plot(D, D_var)
    plt.ylabel("$\sigma_Y(1/2 |y_2 - y_1|)$")
    plt.xlabel("$R=|y_1 - y_2|$ distance")
    plt.show()

#%%
run = execute("print_config", config_updates={
    'obj_func': {
        'name': 'Sinc',
    },
})

#%%

run = execute(config_updates={
    'obj_func': {
        'name': 'Sinc',
    },
    'model': {
        'name': 'GPModel',
        'kwargs': {
            'kernel': {
                'name': 'GPyRBF',
                'kwargs': {
                    'lengthscale': 1
                }
            },
            'noise_prior': None,
            'do_optimize': True,
            'num_mcmc': 0,
            'subsample_interval': 10,
            'step_size': 1e-1,
            'leapfrog_steps': 20,
            'n_burnin': 100,
        }
    },
    'acquisition_function': {
        'name': 'CurvatureAcquisition',
    },
    'bo': {
        'name': 'AcquisitionAlgorithm',
        'kwargs': {
            'n_init': 5,
            'n_iter': 60,
            'n_acq_max_starts': 10,
        }
    },
})


#%% See that sampling strategy indeed matches curvature

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

f = run.interactive_stash['f']
model = run.interactive_stash['model']
plt.plot(model.X)
plt.scatter(np.arange(len(model.X)), model.X)
plt.show()

f._plot(lambda X: np.abs(f.hessian(X)))
plt.hist(model.X, bins=50, normed=True)
#sns.distplot(model.X)
plt.show()


#%% Run uniform sampling, variance sampling, curvature sampling
N_SAMPLES = 65
FUNC = 'Kink2D'

run = notebook_run(through_CLI=False, config_updates={
    'obj_func': {
        'name': FUNC,
    },
    'model': {
        'name': 'GPModel',
        'kwargs': {
            'kernel': {
                'name': 'GPyRBF',
                'kwargs': {
                    'lengthscale': 1
                }
            },
            'noise_prior': None,
            'do_optimize': True,
            'num_mcmc': 0,
            'subsample_interval': 10,
            'step_size': 1e-1,
            'leapfrog_steps': 20,
            'n_burnin': 100,
        }
    },
    'gp_samples': N_SAMPLES,
})

run = notebook_run(through_CLI=False, config_updates={
    'obj_func': {
        'name': FUNC,
    },
    'model': {
        'name': 'GPModel',
        'kwargs': {
            'kernel': {
                'name': 'GPyRBF',
                'kwargs': {
                    'lengthscale': 1
                }
            },
            'noise_prior': None,
            'do_optimize': True,
            'num_mcmc': 0,
            'subsample_interval': 10,
            'step_size': 1e-1,
            'leapfrog_steps': 20,
            'n_burnin': 100,
        }
    },
    'acquisition_function': {
        'name': 'QuadratureAcquisition',
    },
    'bo': {
        'name': 'AcquisitionAlgorithm',
        'kwargs': {
            'n_init': 5,
            'n_iter': N_SAMPLES - 5,
            'n_acq_max_starts': 10,
        }
    },
})

run = notebook_run(through_CLI=False, config_updates={
    'obj_func': {
        'name': FUNC,
    },
    'model': {
        'name': 'GPModel',
        'kwargs': {
            'kernel': {
                'name': 'GPyRBF',
                'kwargs': {
                    'lengthscale': 1
                }
            },
            'noise_prior': None,
            'do_optimize': True,
            'num_mcmc': 0,
            'subsample_interval': 10,
            'step_size': 1e-1,
            'leapfrog_steps': 20,
            'n_burnin': 100,
        }
    },
    'acquisition_function': {
        'name': 'CurvatureAcquisition',
    },
    'bo': {
        'name': 'AcquisitionAlgorithm',
        'kwargs': {
            'n_init': 5,
            'n_iter': N_SAMPLES - 5,
            'n_acq_max_starts': 10,
        }
    },
})


#%% Run only CurvatureAcquisition 2D
run = notebook_run_server(config_updates={
    'obj_func': {
        'name': 'BigSinc',
    },
    'model': {
        'name': 'GPModel',
        'kwargs': {
            'kernel': {
                'name': 'GPyRBF',
                'kwargs': {
                    'lengthscale': 0.01
                }
            },
            'noise_prior': None,
            'do_optimize': True,
            'num_mcmc': 0,
            'subsample_interval': 10,
            'step_size': 1e-1,
            'leapfrog_steps': 20,
            'n_burnin': 100,
        }
    },
    'acquisition_function': {
        'name': 'CurvatureAcquisition',
        'kwargs': {
            'beta': 0,
            'use_var': False,
        }
    },
    # Scale the curvature
    'gp_samples': 50,
    #  'bo': {
    #      'name': 'AcquisitionAlgorithm',
    #      'kwargs': {
    #          'n_init': 5,
    #          'n_iter': 50,
    #          'n_acq_max_starts': 10,
    #      }
    #  },
})


#%%

# What function do we expect it to be good on?
# If model is specified correctly we will sample "uniformly" (given stationary kernel)
# So only if mis-specified model.
# We know however the kernel is not stationary. (it has kinks)
# We _do_ need to sample these high frequency areas more.

# Curvature a tendency to get trapped. (in Kink2D)
    # It will expand the region around high oscillation (where variance * hess is biggest.)
    # If GP hyperparams mode collapses leading to point mass at observation (=> extremely high gradient around points)
    
    # Why is acquisition function small around observations but still exploits?
    # How did we get a diagonal? (seems like two modes are being connected)

#%%

run = notebook_run(through_CLI=False, config_updates={
    'obj_func': {
        'name': "Kink2D",
    },
    'model': {
        'name': 'GPModel',
        'kwargs': {
            'kernel': {
                'name': 'GPyRBF',
                'kwargs': {
                    'lengthscale': 1
                }
            },
            'noise_prior': None,
            'do_optimize': True,
            'num_mcmc': 0,
            'subsample_interval': 10,
            'step_size': 1e-1,
            'leapfrog_steps': 20,
            'n_burnin': 100,
        }
    },
    'acquisition_function': {
        'name': 'CurvatureAcquisition',
        'kwargs': {'use_var': False}
    },
    'bo': {
        'name': 'AcquisitionAlgorithm',
        'kwargs': {
            'n_init': 50,
            'n_iter': 0,
            'n_acq_max_starts': 10,
        }
    },
})
#%%
bo = run.interactive_stash['bo']
x = bo._next_x()
print(x)
bo._add_observations(x)
bo.plot()

#%% Debugging MCMC... (depends on a BO constructed with a run)

run = notebook_run(config_updates={
    'obj_func': {
        'name': 'Sinc',
    },
    'model': {
        'name': 'GPModel',
        'kwargs': {
            'kernel': {
                'name': 'GPyRBF',
                'kwargs': {
                    'lengthscale': 0.1
                }
            },
            'noise_prior': None,
            'do_optimize': True,
            'num_mcmc': 0,
            'subsample_interval': 10,
            'step_size': 1e-1,
            'leapfrog_steps': 20,
            'n_burnin': 100,
        }
    },
    'acquisition_function': {
        'name': 'CurvatureAcquisitionDistribution',
        'kwargs': {
            'beta': 0
        }
    },
    'bo': {
        'name': 'SampleAlgorithm',
        'kwargs': {
            'n_init': 50,
            'n_iter': 0,
            'n_acq_max_starts': 10,
        }
    },
})


import matplotlib.pyplot as plt
import numpy as np
import emcee

bo = run.interactive_stash['bo']
bo.models[0].init(bo.X, bo.Y)
bo.plot()
plt.show()
hess_samples = np.array([bo._next_x() for i in range(100)])
plt.hist(hess_samples[:, 0])
plt.show()

bo.plot()
plt.show()

def acq_one(x):
    return bo.acquisition_function(np.array([x]))[0]

walkers = 20
sampler = emcee.EnsembleSampler(walkers, 1, acq_one)
sampler.run_mcmc([np.array([0]) for i in range(walkers)], 100, storechain=False)
sampler.run_mcmc(None, 100)
plt.hist(sampler.flatchain)
plt.show()
