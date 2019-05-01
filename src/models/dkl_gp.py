import numpy as np
import torch
import gpytorch

from src.models.models import BaseModel

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class LargeFeatureExtractor(torch.nn.Sequential):
    def __init__(self, data_dim, layers=(50, 25, 10, 2)):
        super(LargeFeatureExtractor, self).__init__()

        assert len(layers) >= 1, "You need to specify at least and output layer size."
        layers = (data_dim,) + layers

        i = 0
        self.add_module('linear{}'.format(i), torch.nn.Linear(layers[i], layers[i + 1]))

        for i in range(1, len(layers) - 1):
            in_ = layers[i]
            out = layers[i + 1]
            self.add_module('relu{}'.format(i - 1), torch.nn.ReLU())
            self.add_module('linear{}'.format(i), torch.nn.Linear(in_, out))

        self.output_dim = layers[-1]


class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, feature_extractor):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        # self.covar_module = gpytorch.kernels.GridInterpolationKernel(
        #     gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=feature_extractor.output_dim)),
        #     num_dims=feature_extractor.output_dim, grid_size=30
        # )
        self.covar_module = gpytorch.kernels.RBFKernel(ard_num_dims=feature_extractor.output_dim)
        self.feature_extractor = feature_extractor

    def forward(self, x):
        # We're scaling the features so that they're nice values
        projected_x = self.feature_extractor(x)
        projected_x = projected_x - projected_x.min(0)[0]
        projected_x = 2 * (projected_x / projected_x.max(0)[0]) - 1

        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def default_training_callback(model, i, loss):
    print('Iter %d/%d - Loss: %.3f' % (i + 1, model.n_iter, loss))


class DKLGPModel(BaseModel):
    def __init__(self, n_iter=50, gp_kwargs=None, nn_kwargs=None, training_callback=default_training_callback, **kwargs):
        super(DKLGPModel, self).__init__(**kwargs)

        self.gp_kwargs = gp_kwargs if gp_kwargs is not None else {}
        self.nn_kwargs = nn_kwargs if nn_kwargs is not None else {}

        self.model = None
        self.feature_extractor = None
        self.likelihood = None
        self.n_iter = n_iter

        self.X_torch = None
        self.Y_torch = None

        self.training_callback = training_callback

    def _fit(self, X, Y, Y_dir=None, is_initial=True):
        n, d = X.shape

        self.X_torch = torch.Tensor(X).contiguous().to(device)
        self.Y_torch = torch.Tensor(Y[:, 0]).contiguous().to(device)

        self.feature_extractor = LargeFeatureExtractor(d, **self.nn_kwargs).to(device)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
        self.model = GPRegressionModel(self.X_torch, self.Y_torch, self.likelihood, self.feature_extractor, **self.gp_kwargs).to(device)

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

        def _train():
            for i in range(self.n_iter):
                # Zero backprop gradients
                optimizer.zero_grad()
                # Get output from model
                output = self.model(self.X_torch)
                # Calc loss and backprop derivatives
                loss = -mll(output, self.Y_torch)
                loss.backward()

                self.training_callback(self, i, loss.item())
                optimizer.step()

        # See dkl_mnist.ipynb for explanation of this flag
        with gpytorch.settings.use_toeplitz(True):
            _train()

    def get_features(self, X):
        self.feature_extractor.eval()
        
        test_x = torch.Tensor(X).contiguous().to(device)
        Z = self.feature_extractor(test_x)
        Z = Z.detach().numpy()
        return Z

    def get_statistics(self, X, full_cov=True):
        assert self.model is not None, "Call `self.fit` before predicting."

        # Go into prediction mode
        self.model.eval()
        self.likelihood.eval()

        # Hack to fix issue with making prediction for single inputs (n=1).
        # Only needed if approximate grid interpolation is used.
        #if X.shape[0] == 1:
        #    fake_X = np.zeros((1, X.shape[1]))
        #    X = np.concatenate((X, fake_X), axis=0)
        #    cut_tail = -1
        #else:
        cut_tail = None

        test_x = torch.Tensor(X).contiguous().to(device)

        # Use Toeplitz and LOVE
        with torch.no_grad(), gpytorch.settings.use_toeplitz(False), gpytorch.settings.fast_pred_var():
            multivariate_normal = self.model(test_x)

            mean = multivariate_normal.mean.numpy()[:cut_tail, None]

            if full_cov:
                return mean, multivariate_normal.covariance_matrix.numpy()[:cut_tail, None]
            else:
                return mean, multivariate_normal.variance.numpy()[:cut_tail, None]
