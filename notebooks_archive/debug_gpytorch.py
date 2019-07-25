import matplotlib.pyplot as plt
import numpy as np
from src.environments.financial import SPXOptions
from src.models import NormalizerModel
from src.plot_utils import plot_model_unknown_bounds

# create model
data = SPXOptions(D=1, subset_size=15000)

import math
import torch
import gpytorch
from matplotlib import pyplot as plt

# Make plots inline

train_x = torch.Tensor(data.X_train[:,0]).contiguous().to(torch.device("cpu"))
train_y = torch.Tensor(data.Y_train[:,0]).contiguous().to(torch.device("cpu"))

class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)

        # SKI requires a grid size hyperparameter. This util can help with that. Here we are using a grid that has the same number of points as the training data (a ratio of 1.0). Performance can be sensitive to this parameter, so you may want to adjust it for your own problem on a validation set.

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.GridInterpolationKernel(
            gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()),
            grid_size=10000, num_dims=1,
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


likelihood = gpytorch.likelihoods.GaussianLikelihood().to(torch.device("cpu"))
model = GPRegressionModel(train_x, train_y, likelihood).to(torch.device("cpu"))

# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam([
    {'params': model.parameters()},  # Includes GaussianLikelihood parameters
], lr=0.1)

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

with gpytorch.settings.use_toeplitz(True), \
            gpytorch.settings.fast_computations(True, True, True):
    training_iterations = 30
    for i in range(training_iterations):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
        optimizer.step()
