import numpy as np
import torch
import gpytorch

from src.models.models import BaseModel
from src.utils import construct_2D_grid, call_function_on_grid, random_hypercube_samples
import matplotlib.pyplot as plt


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

from torch import nn

def default_training_callback(model, i, loss):
    if i % 20 == 0:
        print('Iter %d/%d - Loss: %.3f' % (i + 1, model.n_iter, loss))


class LargeFeatureExtractor(torch.nn.Sequential):
    def __init__(self, data_dim, layers=(50, 25, 10, 2)):
        super().__init__()

        assert len(layers) >= 1, "You need to specify at least and output layer size."
        layers = (data_dim,) + tuple(layers)

        for i in range(0, len(layers) - 1):
            in_ = layers[i]
            out = layers[i + 1]
            self.add_module('linear{}'.format(i), torch.nn.Linear(in_, out))
            self.add_module('relu{}'.format(i), torch.nn.ReLU())

        self.output_dim = layers[-1]


class LinearFromFeatureExtractor(BaseModel):
    """Not really DNGO as it does not use a bayesian linear regressor.
    """
    def __init__(self, 
        feature_extractor=None, 
        data_dim=None,
        layers=(50, 25, 10, 2),
        n_iter=50,
        learning_rate=0.1, 
        training_callback=default_training_callback,
        **kwargs):
        super().__init__(**kwargs)

        self.n_iter = n_iter

        if feature_extractor is not None:
            self.feature_extractor = feature_extractor
        else:
            assert layers is not None and data_dim is not None, "DNGO should either have feature_extrator passed or `data_dim` and `layers`."
            self.feature_extractor = LargeFeatureExtractor(data_dim, layers)

        self.model = nn.Sequential(
            self.feature_extractor, 
            nn.Linear(in_features=self.feature_extractor.output_dim, out_features=1, bias=True)
        )

        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        self.training_callback = training_callback


    def _fit(self, X, Y, Y_dir=None, is_initial=True):
        self.model.train()

        # Move tensors to the configured device
        X_torch = torch.Tensor(X).contiguous().to(device)
        Y_torch = torch.Tensor(Y).contiguous().to(device)

        # Train the model
        for i in range(self.n_iter):
            # Forward pass
            Y_pred = self.model(X_torch)
            loss = self.loss(Y_pred, Y_torch)
            
            # Backward and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.training_callback(self, i, loss.item())

    def _get_statistics(self, X, full_cov=True):
        self.model.eval()

        X = torch.from_numpy(X).float()
        X = X.to(device)

        with torch.no_grad():
            # add .cpu() before .numpy() if using GPU
            mean = self.model(X).numpy()

        N = mean.shape[0]

        if full_cov:
            return mean, np.zeros((N, N, 1))
        else:
            return mean, np.zeros((N, 1))


class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, feature_extractor=None, mean_prior=None, n_grid=None):
        if feature_extractor is not None:
            gp_input_dim = feature_extractor.output_dim
        else:
            gp_input_dim = train_x.shape[-1]

        super().__init__(train_x, train_y, likelihood)

        if mean_prior is not None:
            #mean = torch.mean(train_y)
            raise NotImplementedError
        else:
            mean = None
        
        self.mean_module = gpytorch.means.ConstantMean(mean)

        self.uses_grid_interpolation = n_grid is not None
        if n_grid is not None:
            self.covar_module = gpytorch.kernels.GridInterpolationKernel(
                gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.RBFKernel(
                        ard_num_dims=gp_input_dim
                    )
                ),
                num_dims=gp_input_dim, grid_size=n_grid, 
                grid_bounds=None, # TODO: should we set grid bounds?
            )
        else:
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=gp_input_dim)
            )
        self.feature_extractor = feature_extractor

    def forward(self, x):
        if self.feature_extractor is not None:
            projected_x = self.feature_extractor(x)
        else:
            projected_x = x

        # We're scaling the features so that they're nice values
        # TODO: is this not problematic since test time data differs from training data?
        # projected_x = projected_x - projected_x.min(0)[0]
        # projected_x = 2 * (projected_x / projected_x.max(0)[0]) - 1

        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class DKLGPModel(BaseModel):
    def __init__(self, 
        n_iter=50, 
        noise=None, 
        learning_rate=0.1, 
        gp_kwargs=None, 
        nn_kwargs=None, 
        do_pretrain=False,
        pretrain_n_iter=10000,
        training_callback=default_training_callback, 
        **kwargs):

        super().__init__(**kwargs)
        self.do_pretrain = do_pretrain
        self.pretrain_n_iter = pretrain_n_iter

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
            noise = torch.ones(self.X_torch.shape[0]) * self.noise
            self.likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=noise)
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
        #if self.noise is None:
        opt_parameter_list.append({'params': self.model.likelihood.parameters()})

        # optimize with Adam
        optimizer = torch.optim.Adam(opt_parameter_list, lr=self.learning_rate)

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        # Greedily do_pretrain using MSE with an additional layer to output domain
        if self.do_pretrain:
            pretrain_feature_extrator = LinearFromFeatureExtractor(feature_extractor=self.feature_extractor, learning_rate=self.learning_rate, n_iter=self.pretrain_n_iter)
            pretrain_feature_extrator.init(X, Y)

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

        with gpytorch.settings.use_toeplitz(self.model.uses_grid_interpolation):
            _train()

    def get_features(self, X):
        if not self.has_feature_map:
            return X

        self.feature_extractor.eval()
        
        test_x = torch.Tensor(X).contiguous().to(device)
        Z = self.feature_extractor(test_x)
        Z = Z.detach().numpy()
        return Z

    def _get_statistics(self, X, full_cov=True):
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
        with torch.no_grad(), gpytorch.settings.use_toeplitz(self.model.uses_grid_interpolation), gpytorch.settings.fast_pred_var(self.model.uses_grid_interpolation):
            if self.noise is not None:
                noise = torch.ones(test_x.shape[0]) * self.noise
                multivariate_normal = self.likelihood(self.model(test_x), noise=noise)
            else:
                multivariate_normal = self.likelihood(self.model(test_x))

            mean = multivariate_normal.mean.detach().numpy()[:cut_tail, None]

            if full_cov:
                return mean, multivariate_normal.covariance_matrix.detach().numpy()[:cut_tail, None]
            else:
                return mean, multivariate_normal.variance.detach().numpy()[:cut_tail, None]

    def plot_features(self, f):
        if self.feature_extractor is None or not self.feature_extractor.output_dim in [1, 2]:
            return None

        if self.X.shape[-1] == 1:
            X_line = np.linspace(f.bounds[0, 0], f.bounds[0, 1], 100)[:,None]
            fig = plt.figure()

            ax = fig.add_subplot(121)
            ax.set_title("Feature mapping")

            Z = self.get_features(X_line)
            O = f.noiseless(X_line)
            O_hat = self.get_mean(X_line)
            for j in range(Z.shape[1]):
                ax.plot(X_line, Z[:,j])
        elif self.X.shape[-1] == 2:
            XY, X, Y = construct_2D_grid(f.bounds)
            Z = call_function_on_grid(self.get_features, XY)
            O = call_function_on_grid(f.noiseless, XY)

            fig = plt.figure()
            ax = fig.add_subplot(121, projection='3d')
            ax.set_title('Feature mapping')

            #palette = itertools.cycle(sns.color_palette(as_cmap=True))
            for j in range(Z.shape[-1]):
                ax.contourf(X, Y, Z[...,j], 50) #, cmap=next(palette))
        else:
            fig = plt.figure()
            # Base Z on scatter plot with samples randomly drawn from Z as fallback.
            X = random_hypercube_samples(500, f.bounds)
            Z = self.get_features(X)
            O = f.noiseless(X)

        if Z.shape[-1] == 1:
            ax = fig.add_subplot(122)
            ax.set_title('f in feature space')
            if self.X.shape[-1] == 1:
                ax.scatter(Z.flatten(), O.flatten(), label="ground truth")
                ax.scatter(Z.flatten(), O_hat.flatten(), label="prediction")
            else:
                O = np.reshape(O, (-1, 1))
                ax.scatter(Z.flatten(), O.flatten())

        elif Z.shape[-1] == 2:
            if self.X.shape[-1] == 2:
                ax = fig.add_subplot(122)
                ax.set_title('f in feature space')
                ax.contourf(Z[...,0], Z[...,1], O[...,0], 50)
            else:
                ax = fig.add_subplot(122, projection="3d")
                ax.set_title('f in feature space')
                O = np.reshape(O, (-1, 1))
                ax.scatter(Z[...,0], Z[...,1], O[...,0])

        plt.tight_layout()
        return fig
