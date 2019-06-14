import numpy as np
import torch
import gpytorch
import warnings

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


# def FeatureNorm(nn.Module):
#     __constants__ = ['eps', 'running_mean', 'running_var']

#     def __init__(self, num_features, eps=1e-5):
#         self.num_features = num_features
#         self.eps = eps

#         self.register_buffer('running_mean', torch.zeros(num_features))
#         self.register_buffer('running_var', torch.ones(num_features))

#     def reset_running_stats(self):
#         if self.track_running_stats:
#             self.running_mean.zero_()
#             self.running_var.fill_(1)

#     def reset_parameters(self):
#         self.reset_running_stats()


# training: record mean and std: eps=1e-6
        # mean = x.mean(-1)
        # std = x.std(-1)
# forward pass: add normalization
        # (x - mean) / (std + self.eps)



class LargeFeatureExtractor(torch.nn.Sequential):
    def __init__(self, data_dim=None, layers=(50, 25, 10, 2), normalize_output=True, relu_output=False):
        super().__init__()

        assert data_dim is not None, "data_dim needs to be specified"
        assert len(layers) >= 1, "You need to specify at least an output layer size."
        layers = (data_dim,) + tuple(layers)

        for i in range(0, len(layers) - 2):
            in_ = layers[i]
            out = layers[i + 1]
            self.add_module('linear{}'.format(i), torch.nn.Linear(in_, out))
            self.add_module('relu{}'.format(i), torch.nn.ReLU())
        
        self.output_dim = layers[-1]
        self.add_module('linear{}'.format(i+1), torch.nn.Linear(layers[-2], self.output_dim))
        if relu_output:
            self.add_module('relu{}'.format(i+1), torch.nn.ReLU())
        if normalize_output:
            self.add_module('normalization', nn.BatchNorm1d(self.output_dim, affine=False, momentum=1))



class FeatureModel(BaseModel):
    def get_features(self, X):
        if self.feature_extractor is None:
            return X

        self.feature_extractor.eval()
        
        test_x = torch.Tensor(X)

        # TODO: don't use attr that only subclass assigns.
        if hasattr(self, 'use_double_precision') and self.use_double_precision:
            test_x = test_x.double()

        test_x = test_x.contiguous().to(device)
        
        Z = self.feature_extractor(test_x)
        Z = Z.detach().numpy()
        return Z

    def plot_features(self, f):
        if self.feature_extractor is None or not self.feature_extractor.output_dim in [1, 2]:
            return None

        if self.X.shape[-1] == 1:
            X_line = np.linspace(f.bounds[0, 0], f.bounds[0, 1], 1000)[:,None]
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

                O = f.noiseless(X_line)
                O_hat, O_hat_var = self.get_statistics(X_line, full_cov=False)

                #ax.scatter(Z.flatten(), O_hat.flatten(), label="prediction")
                c = np.linspace(0.0, 1.0, Z.shape[0])
                #plt.errorbar(Z.flatten(), O_hat.flatten(), yerr=2 * np.sqrt(O_hat_var), fmt='.k', alpha=0.2, c=c, cmap='viridis', label='prediction')
                ax.scatter(Z.flatten(), O.flatten(), label="ground truth", marker="x")
                ax.scatter(Z.flatten(), O_hat.flatten(), c=c, cmap='viridis', label='prediction', marker=".")
                plt.legend()
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


class LinearFromFeatureExtractor(FeatureModel):
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
            assert layers is not None and data_dim is not None, "DNGO should either have feature_extractor passed or `data_dim` and `layers`."
            self.feature_extractor = LargeFeatureExtractor(data_dim, layers)

        self.model = nn.Sequential(
            self.feature_extractor, 
            nn.Linear(in_features=self.feature_extractor.output_dim, out_features=1, bias=True)
        )

        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        self.training_callback = training_callback

    def _fit(self, X, Y, Y_dir=None):

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
    def __init__(self, train_x, train_y, likelihood, feature_extractor=None, mean_prior=None, lengthscale_prior=None, outputscale_prior=None, n_grid=None,
    kernel='RBF'):
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
        
        self.feature_extractor = feature_extractor
        self.mean_module = gpytorch.means.ZeroMean()

        self.uses_grid_interpolation = n_grid is not None
        self.n_grid = n_grid

        if kernel == 'RBF':
            self.rbf_kernel = gpytorch.kernels.RBFKernel(lengthscale_prior=lengthscale_prior, ard_num_dims=gp_input_dim)
        elif kernel == 'Linear':
            self.rbf_kernel = gpytorch.kernels.RBFKernel(lengthscale_prior=lengthscale_prior, ard_num_dims=gp_input_dim)
        #self.rbf_kernel = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=4, ard_num_dims=gp_input_dim)
        kernel = self.rbf_kernel

        self.scale_kernel = gpytorch.kernels.ScaleKernel(kernel, outputscale_prior=outputscale_prior)
        kernel = self.scale_kernel
        
        if n_grid is not None:
            kernel = gpytorch.kernels.GridInterpolationKernel(
                kernel,
                num_dims=gp_input_dim,
                grid_size=n_grid, 
                grid_bounds=None, # TODO: should we set grid bounds?
            )
        
        self.covar_module = kernel

        # Initialize lengthscale and outputscale to mean of priors
        if lengthscale_prior is not None:
            self.rbf_kernel.lengthscale = lengthscale_prior.mean
        if outputscale_prior is not None:
            self.scale_kernel.outputscale = outputscale_prior.mean

    def initialize_proxy(self, lengthscale=None, outputscale=None, noise=None):
        if isinstance(self.likelihood, gpytorch.likelihoods.FixedNoiseGaussianLikelihood):
            assert noise is None, "Only init noise if not fixed."

        if lengthscale is not None:
            self.rbf_kernel.lengthscale = lengthscale
        if outputscale is not None:
            self.scale_kernel.outputscale = outputscale
        if noise is not None:
            self.likelihood.noise = noise
        return self

    def get_lengthscale(self):
        return self.rbf_kernel.lengthscale

    def get_outputscale(self):
        return self.scale_kernel.outputscale

    def forward(self, x):
        if self.feature_extractor is not None:
            projected_x = self.feature_extractor(x)
        else:
            projected_x = x

        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class DKLGPModel(FeatureModel):
    def __init__(self, 
        n_iter=50,
        noise=None,
        learning_rate=0.1, 
        gp_kwargs=None, 
        nn_kwargs=None, 
        do_pretrain=False,
        pretrain_n_iter=10000,
        training_callback=default_training_callback, 
        feature_extractor_constructor=LargeFeatureExtractor,
        initial_parameters=None,
        use_double_precision=False,
        covar_root_decomposition=True,
        use_cg=False,
        max_cg_iter=1000,
        precond_size=10,
        **kwargs):

        super().__init__(**kwargs)
        self.use_double_precision = use_double_precision
        self.covar_root_decomposition = covar_root_decomposition
        self.use_cg = use_cg
        self.max_cg_iter = max_cg_iter
        self.precond_size = precond_size

        self.feature_extractor_constructor = feature_extractor_constructor

        self.do_pretrain = do_pretrain
        self.pretrain_n_iter = pretrain_n_iter

        self.gp_kwargs = gp_kwargs if gp_kwargs is not None else {}
        self.nn_kwargs = nn_kwargs if nn_kwargs is not None else {}

        # self.gp_kwargs.update({
        #     'lengthscale_prior': gpytorch.priors.GammaPrior(3.0, 6.0),
        #     'outputscale_prior': gpytorch.priors.GammaPrior(2.0, 0.15),
        # })
        
        self.initial_parameters = initial_parameters

        self.learning_rate = learning_rate
        self.noise = noise
        self.model = None
        self.feature_extractor = None
        self.likelihood = None
        self.n_iter = n_iter
        self.training_loss = None

        self.X_torch = None
        self.Y_torch = None

        self.has_feature_map = self.nn_kwargs.get('layers') is not None

        self.training_callback = training_callback

        self.warnings = {}

    def get_common_hyperparameters(self):
        return {
            'outputscale': self.model.get_outputscale(),
            'lengthscale': self.model.get_lengthscale(),
            'noise': self.noise if self.noise is not None else self.likelihood.noise.item(),
        }

    def _fit(self, X, Y, Y_dir=None):
        # catch warning, save and stop (runner should store this warning)
        if self.do_pretrain:
            self._train(X, Y, fix_gp_params=True)
        self._train(X, Y, fix_gp_params=False)

    def _train(self, X, Y, fix_gp_params=False):
        n, d = X.shape

        self.X_torch = torch.Tensor(X)
        self.Y_torch = torch.Tensor(Y[:, 0])

        if self.use_double_precision:
            self.X_torch = self.X_torch.double()
            self.Y_torch = self.Y_torch.double()

        self.X_torch = self.X_torch.contiguous().to(device)
        self.Y_torch = self.Y_torch.contiguous().to(device)

        if self.has_feature_map:
            self.feature_extractor = LargeFeatureExtractor(d, **self.nn_kwargs).to(device)

            if self.use_double_precision:
                self.feature_extractor = self.feature_extractor.double()

        if self.noise is not None:
            noise = torch.ones(self.X_torch.shape[0]) * self.noise
            if self.use_double_precision:
                noise = noise.double()
            self.likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=noise)
        else:
            self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)

        if self.use_double_precision:
            self.likelihood = self.likelihood.double()

        self.model = GPRegressionModel(self.X_torch, self.Y_torch, self.likelihood, self.feature_extractor, **self.gp_kwargs).to(device)

        if self.use_double_precision:
            self.model = self.model.double()

        if self.initial_parameters is not None:
            print(self.initial_parameters)
            self.model.initialize_proxy(**self.initial_parameters)

        # Go into training mode
        self.model.train()
        self.likelihood.train()

        opt_parameter_list = []

        if not fix_gp_params:
            opt_parameter_list.append({'params': self.model.covar_module.parameters()})
            opt_parameter_list.append({'params': self.model.mean_module.parameters()})
            # Only add noise as hyperparameter if it is not fixed.
            #if self.noise is None:
            opt_parameter_list.append({'params': self.model.likelihood.parameters()})

        if self.has_feature_map:
            self.feature_extractor.train()
            opt_parameter_list.append({'params': self.model.feature_extractor.parameters()})

        # optimize with Adam
        optimizer = torch.optim.Adam(opt_parameter_list, lr=self.learning_rate)

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        #if self.do_pretrain:
            # Greedily do_pretrain using MSE with an additional layer to output domain
            # pretrain_feature_extrator = LinearFromFeatureExtractor(feature_extractor=self.feature_extractor, learning_rate=self.learning_rate, n_iter=self.pretrain_n_iter)
            # pretrain_feature_extrator.init(X, Y)



        def _train():
            self.training_loss = np.empty(self.n_iter)

            for i in range(self.n_iter):
                # Zero backprop gradients
                optimizer.zero_grad()
                # Get output from model
                output = self.model(self.X_torch)
                # Calc loss and backprop derivatives
                loss = -mll(output, self.Y_torch)
                loss.backward()

                training_loss = loss.item()
                if i % 30 == 0:
                    print('Current hyperparameters:', self.get_common_hyperparameters())
                self.training_loss[i] = training_loss
                self.training_callback(self, i, training_loss)
                optimizer.step()

        with gpytorch.settings.use_toeplitz(self.model.uses_grid_interpolation), \
            gpytorch.settings.fast_computations(covar_root_decomposition=self.covar_root_decomposition, log_prob=self.use_cg, solves=self.use_cg),\
            gpytorch.settings.max_cg_iterations(self.max_cg_iter),\
            gpytorch.settings.max_preconditioner_size(self.precond_size):

            # Catch CG warning
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")

                _train()

                self.store_CG_warning('training', w)

    def store_CG_warning(self, key, warnings):
        for w in warnings:
            if issubclass(w.category, UserWarning) \
                and hasattr(w.message, 'args') \
                and len(w.message.args) >= 1 \
                and "CG terminated in " in str(w.message.args[0]):

                self.warnings.update({key: True})

    def plot_loss(self):
        fig, ax = plt.subplots()
        ax.plot(self.training_loss)
        return fig

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

        test_x = torch.Tensor(X)

        if self.use_double_precision:
            test_x = test_x.double()

        test_x = test_x.contiguous().to(device)

        with torch.no_grad(), \
            gpytorch.settings.fast_computations(covar_root_decomposition=self.covar_root_decomposition, log_prob=self.use_cg, solves=self.use_cg), \
            gpytorch.settings.use_toeplitz(self.model.uses_grid_interpolation), \
            gpytorch.settings.fast_pred_var(self.use_cg), \
            gpytorch.settings.max_cg_iterations(self.max_cg_iter),\
            gpytorch.settings.max_preconditioner_size(self.precond_size):

            # Catch CG warning
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")

                if self.noise is not None:
                    noise = torch.ones(test_x.shape[0]) * self.noise
                    if self.use_double_precision:
                        noise = noise.double()

                    multivariate_normal = self.likelihood(self.model(test_x), noise=noise)
                else:
                    multivariate_normal = self.likelihood(self.model(test_x))

                mean = multivariate_normal.mean.detach().numpy()[:cut_tail, None]

                self.store_CG_warning('pred', w)

            if full_cov:
                return mean, multivariate_normal.covariance_matrix.detach().numpy()[:cut_tail, :cut_tail, None]
            else:
                return mean, multivariate_normal.variance.detach().numpy()[:cut_tail, None]

