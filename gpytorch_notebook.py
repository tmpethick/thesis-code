#%%
import numpy as np
import math
import torch
import gpytorch
from matplotlib import pyplot as plt

%matplotlib inline
%load_ext autoreload
%autoreload 2

from src.environments.smooth import Sinc

# Training data is 11 points in [0,1] inclusive regularly spaced
train_x = np.linspace(-20, 20, 100)
# True function is sin(2*pi*x) with Gaussian noise
train_y = torch.Tensor(Sinc()(train_x)) #+ np.random.normal(0, 0.1, size=100))
train_x = torch.Tensor(train_x)

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.RBFKernel()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# initialize likelihood and model
#likelihood = gpytorch.likelihoods.GaussianLikelihood() #noise_prior=gpytorch.priors.GammaPrior(0.2,0.01))
likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=torch.ones(train_x.shape[0]) * 0.01)
model = ExactGPModel(train_x, train_y, likelihood)
#model.rbf_covar_module.initialize(lengthscale=2e-3)
#model.mean_module.initialize(constant=0)

#%%
# Find optimal model hyperparameters
model.train()
likelihood.train()

#likelihood.initialize(noise=0.1)

# Use the adam optimizer
optimizer = torch.optim.Adam([
    {'params': model.parameters()}, 
    {'params': likelihood.parameters()}, 
], lr=0.01, )

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

training_iter = 100
for i in range(training_iter):
    # Zero gradients from previous iteration
    optimizer.zero_grad()
    # Output from model
    output = model(train_x)
    # Calc loss and backprop gradients
    loss = -mll(output, train_y)
    loss.backward()
    print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
        i + 1, training_iter, loss.item(),
        model.covar_module.lengthscale[0].item(),
        model.likelihood.noise[0].item()
    ))
    optimizer.step()

# Get into evaluation (predictive posterior) mode
model.eval()
likelihood.eval()

# Test points are regularly spaced along [0,1]
# Make predictions by feeding model through likelihood
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    test_x = torch.linspace(-20, 20, 20)
    observed_pred = model(test_x)

#%%
with torch.no_grad():
    # Initialize plot
    f, ax = plt.subplots(1, 1, figsize=(4, 3))

    # Get upper and lower confidence bounds
    lower, upper = observed_pred.confidence_region()
    # Plot training data as black stars
    ax.plot(train_x.numpy(), train_y.numpy(), 'k*')
    # Plot predictive means as blue line
    ax.plot(test_x.numpy(), observed_pred.mean.numpy(), 'b')
    # Shade between the lower and upper confidence bounds
    ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
    #ax.set_ylim([-20])
    ax.legend(['Observed Data', 'Mean', 'Confidence'])

#%%
