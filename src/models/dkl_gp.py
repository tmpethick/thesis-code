import numpy as np
import torch
import gpytorch

from src.models.models import BaseModel
from src.utils import construct_2D_grid, call_function_on_grid
import matplotlib.pyplot as plt


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class LargeFeatureExtractor(torch.nn.Sequential):
    def __init__(self, data_dim, layers=(50, 25, 10, 2)):
        super().__init__()

        assert len(layers) >= 1, "You need to specify at least and output layer size."
        layers = (data_dim,) + tuple(layers)

        i = 0
        self.add_module('linear{}'.format(i), torch.nn.Linear(layers[i], layers[i + 1]))

        for i in range(1, len(layers) - 1):
            in_ = layers[i]
            out = layers[i + 1]
            self.add_module('relu{}'.format(i - 1), torch.nn.ReLU())
            self.add_module('linear{}'.format(i), torch.nn.Linear(in_, out))

        self.output_dim = layers[-1]


class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, feature_extractor=None):
        if feature_extractor is not None:
            gp_input_dim = feature_extractor.output_dim
        else:
            gp_input_dim = train_x.shape[-1]

        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        # self.covar_module = gpytorch.kernels.GridInterpolationKernel(
        #     gpytorch.kernels.ScaleKernel(
        #         gpytorch.kernels.RBFKernel(
        #             ard_num_dims=gp_input_dim
        #         )
        #     ),
        #     num_dims=gp_input_dim, grid_size=10, grid_bounds=,
        # )
        self.covar_module = gpytorch.kernels.RBFKernel(ard_num_dims=gp_input_dim)
        self.feature_extractor = feature_extractor

    def forward(self, x):
        if self.feature_extractor is not None:
            projected_x = self.feature_extractor(x)
        else:
            projected_x = x

        # We're scaling the features so that they're nice values
        # TODO: is this not problematic since test time data differs from training data?
        projected_x = projected_x - projected_x.min(0)[0]
        projected_x = 2 * (projected_x / projected_x.max(0)[0]) - 1

        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def default_training_callback(model, i, loss):
    if i % 20 == 0:
        print('Iter %d/%d - Loss: %.3f' % (i + 1, model.n_iter, loss))


class DKLGPModel(BaseModel):
    def __init__(self, n_iter=50, noise=None, learning_rate=0.1, gp_kwargs=None, nn_kwargs=None, training_callback=default_training_callback, **kwargs):
        super().__init__(**kwargs)

        self.gp_kwargs = gp_kwargs if gp_kwargs is not None else {}
        self.nn_kwargs = nn_kwargs if nn_kwargs is not None else {}

        self.learning_rate = learning_rate
        self.noise = noise
        self.model = None
        self.feature_extractor = None
        self.likelihood = None
        self.n_iter = n_iter

        self.X_torch = None
        self.Y_torch = None

        self.has_feature_map = self.nn_kwargs['layers'] is not None

        self.training_callback = training_callback

    def _fit(self, X, Y, Y_dir=None, is_initial=True):
        n, d = X.shape

        self.X_torch = torch.Tensor(X).contiguous().to(device)
        self.Y_torch = torch.Tensor(Y[:, 0]).contiguous().to(device)

        if self.has_feature_map:
            self.feature_extractor = LargeFeatureExtractor(d, **self.nn_kwargs).to(device)

        if self.noise is not None:
            self.likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(torch.ones(self.X_torch.shape[0]) * self.noise)
        else:
            self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)

        self.model = GPRegressionModel(self.X_torch, self.Y_torch, self.likelihood, self.feature_extractor, **self.gp_kwargs).to(device)

        # Go into training mode
        self.model.train()
        self.likelihood.train()

        opt_parameter_list = [
            {'params': self.model.covar_module.parameters()},
            {'params': self.model.mean_module.parameters()},
        ]

        if self.has_feature_map:
            self.feature_extractor.eval()

            opt_parameter_list.append({'params': self.model.feature_extractor.parameters()})

        # Only add noise as hyperparameter if it is not fixed.
        if self.noise is None:
            opt_parameter_list.append({'params': self.model.likelihood.parameters()})

        # optimize with Adam
        optimizer = torch.optim.Adam(opt_parameter_list, lr=self.learning_rate)

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

        #with gpytorch.settings.use_toeplitz(True):
        #    _train()
        _train()

    def get_features(self, X):
        if not self.has_feature_map:
            return X

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
        
        if self.has_feature_map:
            self.feature_extractor.eval()

        # Hack to fix issue with making prediction for single inputs (n=1).
        # Only needed if approximate grid interpolation is used.
        #if X.shape[0] == 1:
        #    fake_X = np.zeros((1, X.shape[1]))
        #    X = np.concatenate((X, fake_X), axis=0)
        #    cut_tail = -1
        #else:
        cut_tail = None

        test_x = torch.Tensor(X).contiguous().to(device)

        # Use LOVE
        #with torch.no_grad(), gpytorch.settings.use_toeplitz(True), gpytorch.settings.fast_pred_var():
        with torch.no_grad():
            # Passing through likelihood is not needed if using fixed noise
            if self.noise is not None:
                multivariate_normal = self.model(test_x)
            else:
                multivariate_normal = self.likelihood(self.model(test_x))

            mean = multivariate_normal.mean.numpy()[:cut_tail, None]

            if full_cov:
                return mean, multivariate_normal.covariance_matrix.numpy()[:cut_tail, None]
            else:
                return mean, multivariate_normal.variance.numpy()[:cut_tail, None]

    def plot_features(self, f):
        if not self.X.shape[-1] in [1, 2]:
            return None

        if self.X.shape[-1] == 1:
            X_line = np.linspace(f.bounds[0, 0], f.bounds[0, 1], 100)[:,None]
            fig = plt.figure()

            ax = fig.add_subplot(121)
            ax.set_title("Feature mapping")

            Z = self.get_features(X_line)
            O = f(X_line)
            for j in range(Z.shape[1]):
                ax.plot(X_line, Z[:,j])


        elif self.X.shape[-1] == 2:
            XY, X, Y = construct_2D_grid(f.bounds)
            Z = call_function_on_grid(self.get_features, XY)
            O = call_function_on_grid(f, XY)

            fig = plt.figure()
            ax = fig.add_subplot(121, projection='3d')
            ax.set_title('Feature mapping')

            #palette = itertools.cycle(sns.color_palette(as_cmap=True))
            for j in range(Z.shape[-1]):
                ax.contourf(X, Y, Z[...,j], 50) #, cmap=next(palette))

        if Z.shape[-1] == 1:
            ax = fig.add_subplot(122)
            ax.set_title('f in feature space')
            # Collapse (if X was 2D)
            ax.plot(Z.flatten(), O.flatten())
        elif Z.shape[-1] == 2:
            # TODO: fix for 1D->2D case
            if self.X.shape[-1] == 2:
                #Z = np.reshape(Z, (-1, 2))
                ax = fig.add_subplot(122)
                ax.set_title('f in feature space')
                print(Z.shape, O.shape)
                ax.contourf(Z[...,0], Z[...,1], O[...,0], 50)


        plt.tight_layout()
        return fig
