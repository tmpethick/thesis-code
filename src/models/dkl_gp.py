import torch
import gpytorch

from src.models.models import BaseModel

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


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


class DKLGPModel(BaseModel):
    def __init__(self, gp_dim=2):
        self.gp_dim = gp_dim
        self.model = None
        self.likelihood = None

    def fit(self, X, Y, is_initial=True):
        super(DKLGPModel, self).fit(X, Y, is_initial=is_initial)
        
        n, d = X.shape

        train_x = torch.Tensor(X).contiguous().to(device)
        train_y = torch.Tensor(Y).contiguous().to(device)

        feature_extractor = LargeFeatureExtractor(d, self.gp_dim).to(device)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
        self.model = GPRegressionModel(train_x, train_y, self.likelihood, feature_extractor).to(device)

        # Go into training mode
        self.model.train()
        self.likelihood.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam([
            {'params': self.model.feature_extractor.parameters()},
            {'params': self.model.covar_module.parameters()},
            {'params': self.model.mean_module.parameters()},
            {'params': self.model.likelihood.parameters()},
        ], lr=0.01)

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        training_iterations = 60

        def _train():
            for i in range(training_iterations):
                # Zero backprop gradients
                optimizer.zero_grad()
                # Get output from model
                output = self.model(train_x)
                # Calc loss and backprop derivatives
                loss = -mll(output, train_y)
                loss.backward()
                print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
                optimizer.step()

        # See dkl_mnist.ipynb for explanation of this flag
        with gpytorch.settings.use_toeplitz(True):
            _train()

    def get_statistics(self, X, full_cov=True):
        assert self.model is not None, "Call `self.fit` before predicting."

        # Go into prediction mode
        self.model.eval()
        self.likelihood.eval()

        test_x = torch.Tensor(X).contiguous().to(device)

        with torch.no_grad(), gpytorch.settings.use_toeplitz(False), gpytorch.settings.fast_pred_var():
            multivariate_normal = self.model(test_x)

        mean = multivariate_normal.mean.numpy()

        if full_cov:
            return mean, multivariate_normal.covariance_matrix.numpy()
        else:
            return mean, multivariate_normal.variance.numpy()

    def plot(self, ax=None):
        n, d = self.X.shape

        if d == 1:
            pass
        elif d == 2:
            pass
        else:
            raise ValueError("Input dim can be at most 2 but is {}.".format(d))
