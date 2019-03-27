#%%
%load_ext autoreload
%autoreload 2

import math

import numpy as np
import torch
import gpytorch
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

import seaborn as sns
sns.set_style("darkgrid")

class LargeFeatureExtractor(torch.nn.Sequential):
    def __init__(self, data_dim, output_dim):
        super(LargeFeatureExtractor, self).__init__()
        self.add_module('linear1', torch.nn.Linear(data_dim, 50))
        self.add_module('relu1', torch.nn.ReLU())
        self.add_module('linear2', torch.nn.Linear(50, 25))
        self.add_module('relu2', torch.nn.ReLU())
        self.add_module('linear3', torch.nn.Linear(25, 10))
        self.add_module('relu3', torch.nn.ReLU())
        self.add_module('linear4', torch.nn.Linear(10, output_dim))
        self.output_dim = output_dim

class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, feature_extractor):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.GridInterpolationKernel(
            gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=feature_extractor.output_dim)),
            num_dims=feature_extractor.output_dim, grid_size=100
        )
        self.feature_extractor = feature_extractor

    def forward(self, x):
        # We're scaling the features so that they're nice values
        projected_x = self.feature_extractor(x)
        projected_x = projected_x - projected_x.min(0)[0]
        projected_x = 2 * (projected_x / projected_x.max(0)[0]) - 1

        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def train(train_x, train_y, data_dim, gp_dim=2):
    feature_extractor = LargeFeatureExtractor(data_dim, gp_dim).to(device)
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    model = GPRegressionModel(train_x, train_y, likelihood, feature_extractor).to(device)

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam([
        {'params': model.feature_extractor.parameters()},
        {'params': model.covar_module.parameters()},
        {'params': model.mean_module.parameters()},
        {'params': model.likelihood.parameters()},
    ], lr=0.01)

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    training_iterations = 60
    def _train():
        for i in range(training_iterations):
            # Zero backprop gradients
            optimizer.zero_grad()
            # Get output from model
            output = model(train_x)
            # Calc loss and backprop derivatives
            loss = -mll(output, train_y)
            loss.backward()
            print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
            optimizer.step()

    # See dkl_mnist.ipynb for explanation of this flag
    with gpytorch.settings.use_toeplitz(True):
        _train()
    
    return model, likelihood

def predict(model, XY):
    test_x = torch.Tensor(XY).contiguous().to(device)
    test_z = torch.Tensor(Z).contiguous().to(device)

    with torch.no_grad(), gpytorch.settings.use_toeplitz(False), gpytorch.settings.fast_pred_var():
        multivariate_normal = model(test_x)
    
    return multivariate_normal


#%% --------------------- 2D -----------------------
# Financial 2D
from src.algorithms import construct_2D_grid, call_function_on_grid
from src.algorithms import random_hypercube_samples

data_dim = 2

def f(x):
   y = 1 / (np.abs(0.5 - x[...,0] ** 4 - x[...,1] ** 4) + 0.1)
   return y[...,None]

bounds = np.array([[0,1],[0,1]])

X = random_hypercube_samples(15 ** data_dim, bounds)
Y = f(X)[:,0]

train_x = torch.Tensor(X).contiguous().to(device)
train_y = torch.Tensor(Y).contiguous().to(device)

model, likelihood = train(train_x, train_y, data_dim, gp_dim=3)

#%% Test
model.eval()
likelihood.eval()

XY, X, Y = construct_2D_grid(bounds)
Z = call_function_on_grid(f, XY)
Z_pred_mean = call_function_on_grid(lambda XY: predict(model, XY).mean.numpy(), XY)
Z_pred_var = call_function_on_grid(lambda XY: predict(model, XY).variance.numpy(), XY)

fig = plt.figure()
ax = fig.add_subplot(221)
ax.contour(X, Y, Z, 50)
ax = fig.add_subplot(222)
ax.contour(X, Y, Z_pred_mean, 50)
ax = fig.add_subplot(223)
ax.contour(X, Y, Z_pred_var, 50)
ax = fig.add_subplot(224, projection='3d')
ax.contour3D(X,Y,np.abs(Z_pred_mean-Z), 50, cmap='binary')

#%% -------------------- 1D ---------------------
def f(x):
   return 1 / (10 ** (-4) + x ** 2)

data_dim = 1
bounds = np.array([[-2,2]])

X = np.random.uniform(bounds[0,0], bounds[0,1], 30)[:, None]
Y = f(X)[:,0]

train_x = torch.Tensor(X).contiguous().to(device)
train_y = torch.Tensor(Y).contiguous().to(device)

model, likelihood = train(train_x, train_y, data_dim, gp_dim=2)


#%% Test
model.eval()
likelihood.eval()

X_line = np.linspace(bounds[0,0], bounds[0,1], 100)[:,None]
Y_line = f(X_line)[:,0]

multivariate_normal_pred = predict(model, X_line)
mean = multivariate_normal_pred.mean.numpy()
var = multivariate_normal_pred.variance.numpy()

plt.scatter(X.reshape(-1), Y)
plt.fill_between(X_line.reshape(-1), (mean + 2 * np.sqrt(var)).reshape(-1), (mean - 2 * np.sqrt(var)))
plt.plot(X_line, mean)
plt.plot(X_line, Y_line)

#%%

# Isolate models
# Test as GP
# Test as BO

# Plot lengthscale change
# Compare with base-line (standard GP)
