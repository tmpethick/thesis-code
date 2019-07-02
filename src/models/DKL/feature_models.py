import warnings

import gpytorch
import numpy as np
from matplotlib import pyplot as plt
import torch

from src.experiment.config_helpers import ConfigMixin, lazy_construct_from_module, LazyConstructor
from ..core_models import BaseModel
from .gpr import GPRegressionModel
from .feature_extractors import LargeFeatureExtractor, RFFEmbedding
from src.utils import construct_2D_grid, call_function_on_grid, random_hypercube_samples


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def default_training_callback(model, i, loss):
    if i % 20 == 0:
        print('Iter %d/%d - Loss: %.3f' % (i + 1, model.n_iter, loss))


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

        self.model = torch.nn.Sequential(
            self.feature_extractor,
            torch.nn.Linear(in_features=self.feature_extractor.output_dim, out_features=1, bias=True)
        )

        self.loss = torch.nn.MSELoss()
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


class GPyTorchModel(ConfigMixin, FeatureModel):
    def __init__(self,
                 n_iter=50,
                 learning_rate=0.1,
                 noise=None,
                 noise_lower_bound=1e-4,
                 initial_parameters=None,

                 do_pretrain=False,
                 pretrain_n_iter=10000,

                 feature_extractor_constructor=LazyConstructor(LargeFeatureExtractor),
                 gp_constructor=LazyConstructor(GPRegressionModel),

                 use_double_precision=False,
                 covar_root_decomposition=True,
                 use_cg=False,
                 max_cg_iter=1000,
                 precond_size=10,
                 eval_cg_tolerance=1e-4,

                 training_callback=default_training_callback,
                 **kwargs):

        super().__init__(**kwargs)
        self.use_double_precision = use_double_precision
        self.covar_root_decomposition = covar_root_decomposition
        self.use_cg = use_cg
        self.max_cg_iter = max_cg_iter
        self.precond_size = precond_size
        self.use_toeplitz = False # Set in `_train` when input_dim is known.
        self.eval_cg_tolerance = eval_cg_tolerance

        self.has_feature_map = feature_extractor_constructor is not None
        self.feature_extractor_constructor = feature_extractor_constructor
        self.gp_constructor = gp_constructor

        self.do_pretrain = do_pretrain
        self.pretrain_n_iter = pretrain_n_iter
        self.initial_parameters = initial_parameters

        self.learning_rate = learning_rate
        self.noise = noise
        self.noise_lower_bound = noise_lower_bound
        self.model = None
        self.feature_extractor = None
        self.likelihood = None
        self.n_iter = n_iter
        self.training_loss = None

        self.X_torch = None
        self.Y_torch = None

        self.training_callback = training_callback

        self.warnings = {}

    @classmethod
    def process_config(cls, *, feature_extractor_constructor=None, gp_constructor=None, **kwargs):
        from src.models.DKL import feature_extractors
        from src.models.DKL import gpr
        d = dict(**kwargs)
        if feature_extractor_constructor is not None:
            d['feature_extractor_constructor'] = lazy_construct_from_module(feature_extractors, feature_extractor_constructor)
        if gp_constructor is not None:
            d['gp_constructor'] = lazy_construct_from_module(gpr, gp_constructor)
        return d

    def _fit(self, X, Y, Y_dir=None):
        # catch warning, save and stop (runner should store this warning)
        if self.do_pretrain:
            self._train(X, Y, fix_gp_params=True)
        self._train(X, Y, fix_gp_params=False)

    def make_double(self, tensor):
        if self.use_double_precision:
            return tensor.double()
        else:
            return tensor.float()

    def to_torch(self, X):
        X_torch = torch.Tensor(X)
        X_torch = self.make_double(X_torch)
        X_torch = X_torch.contiguous().to(device)
        return X_torch

    def _train(self, X, Y, fix_gp_params=False):
        n, d = X.shape

        self.X_torch = self.to_torch(X)
        self.Y_torch = self.to_torch(Y[:, 0])

        if self.has_feature_map:
            self.feature_extractor = self.feature_extractor_constructor(D=d).to(device)
            self.feature_extractor = self.make_double(self.feature_extractor)

        if self.noise is not None:
            noise = torch.ones(self.X_torch.shape[0]) * self.noise
            noise = self.make_double(noise)
            self.likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
                noise=noise,
                noise_constraint=gpytorch.constraints.GreaterThan(self.noise_lower_bound)
            )
        else:
            self.likelihood = gpytorch.likelihoods.GaussianLikelihood(
                noise_constraint=gpytorch.constraints.GreaterThan(self.noise_lower_bound)
            ).to(device)

        self.likelihood = self.make_double(self.likelihood)

        self.model = self.gp_constructor(self.X_torch, self.Y_torch, self.likelihood, self.feature_extractor).to(device)

        self.model = self.make_double(self.model)

        if self.model.uses_inducing_points:
            self.use_toeplitz = True
        else:
            self.use_toeplitz = False

        if self.initial_parameters is not None:
            print(self.initial_parameters)
            self.initialize_parameters(**self.initial_parameters)
        
        self.optimize(self.X_torch, self.Y_torch, fix_gp_params=fix_gp_params)

    def set_train_data(self, X, Y):
        if isinstance(X, np.ndarray):
            X = self.to_torch(X)
            Y = self.to_torch(Y[:, 0])
        elif isinstance(X, torch.Tensor):
            Y = Y[:, 0]
        else:
            raise ValueError("X and Y have to be numpy array or torch tensor.")
        self.model.set_train_data(X, Y, strict=False)
        
        self.X_torch = X
        self.Y_torch = Y

    def optimize(self, X, Y, fix_gp_params=False): 
        print('training on {} data points of dim {}'.format(X.shape[0], X.shape[-1])) 
        # Go into training mode
        self.model.train()
        self.likelihood.train()

        opt_parameter_list = []

        if not fix_gp_params:
            opt_parameter_list.append({'params': self.model.covar_module.parameters()})
            opt_parameter_list.append({'params': self.model.mean_module.parameters()})
            # Only add noise as hyperparameter if noise is not fixed and inducing points 
            # are not used (since the likelihood is added through the kernel in that case)
            if self.noise is None and self.model.inducing_points is None:
                opt_parameter_list.append({'params': self.model.likelihood.parameters()})

        if self.has_feature_map:
            self.feature_extractor.train()
            opt_parameter_list.append({'params': self.model.feature_extractor.parameters()})

        # optimize with Adam
        optimizer = torch.optim.Adam(opt_parameter_list, lr=self.learning_rate)

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        def _train():
            self.training_loss = np.empty(self.n_iter)

            for i in range(self.n_iter):
                # Zero backprop gradients
                optimizer.zero_grad()
                # Get output from model
                output = self.model(X)
                # Calc loss and backprop derivatives
                loss = -mll(output, Y)
                loss.backward()

                loss_ = loss.item()
                if i % 30 == 0:
                    print('Current hyperparameters:', self.get_common_hyperparameters())
                self.training_loss[i] = loss_
                self.training_callback(self, i, loss_)
                optimizer.step()

        with gpytorch.settings.use_toeplitz(self.use_toeplitz), \
            gpytorch.settings.fast_computations(covar_root_decomposition=self.covar_root_decomposition, log_prob=self.use_cg, solves=self.use_cg), \
            gpytorch.settings.max_cg_iterations(self.max_cg_iter), \
            gpytorch.settings.max_preconditioner_size(self.precond_size), \
            gpytorch.settings.eval_cg_tolerance(self.eval_cg_tolerance):

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
        print('predicting {} points using {} training points'.format(X.shape[0], self.X_torch.shape[0])) 
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
            gpytorch.settings.use_toeplitz(self.use_toeplitz), \
            gpytorch.settings.fast_pred_var(self.use_cg), \
            gpytorch.settings.max_cg_iterations(self.max_cg_iter),\
            gpytorch.settings.max_preconditioner_size(self.precond_size), \
            gpytorch.settings.eval_cg_tolerance(self.eval_cg_tolerance):

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

    def initialize_parameters(self, noise=None, **kwargs):
        """Overwrite this for custom easy initialization of custom kernels.
        """
        if isinstance(self.likelihood, gpytorch.likelihoods.FixedNoiseGaussianLikelihood):
          assert noise is None, "Only init noise if not fixed."
        kwargs.update({
            'likelihood.noise': noise
        })
        self.model.initialize(**kwargs)


class SSGP(GPyTorchModel):
    def __init__(self, *args, **kwargs):
        defaults = dict(
            covar_root_decomposition=True,
            feature_extractor_constructor=LazyConstructor(RFFEmbedding, M=100),
            gp_constructor=LazyConstructor(GPRegressionModel,
                                           n_grid=None,
                                           kernel=LazyConstructor(gpytorch.kernels.LinearKernel),
                                           has_scale_kernel=False
                                           )
        )
        defaults.update(kwargs)
        super().__init__(*args, **defaults)

    def initialize_parameters(self, lengthscale=None, variance=None, **kwargs):
        kwargs.update({
            'covar_module.variance': variance,
            'feature_extractor.lengthscale': lengthscale,
        })
        super().initialize_parameters(**kwargs)

    def get_common_hyperparameters(self):
        return {
            'lengthscale': self.model.feature_extractor.lengthscale.detach().numpy(),
            'variance': self.model.covar_module.variance,
            'noise': self.noise if self.noise is not None else self.likelihood.noise.item(),
        }


class DKLGPModel(GPyTorchModel):
    def __init__(self, *args, nn_kwargs=None, gp_kwargs=None, **kwargs):
        if nn_kwargs is None:
            nn_kwargs = {}
        if gp_kwargs is None:
            gp_kwargs = {}

        if nn_kwargs.get('layers') is not None:
            feature_extractor_constructor = LazyConstructor(LargeFeatureExtractor, **nn_kwargs)
        else:
            feature_extractor_constructor = None

        default_gp_kwargs = dict(
            kernel=LazyConstructor(gpytorch.kernels.RBFKernel, lengthscale_prior=None),
        )
        default_gp_kwargs.update(gp_kwargs)
        kwargs.update(
            feature_extractor_constructor=feature_extractor_constructor,
            gp_constructor=LazyConstructor(GPRegressionModel, **default_gp_kwargs)
        )
        super().__init__(*args, **kwargs)

    def initialize_parameters(self, **kwargs):
        if self.model.uses_inducing_points:
            kwargs.update({
                'covar_module.outputscale': kwargs.get('outputscale'),
                'covar_module.base_kernel.base_kernel.lengthscale': kwargs.get('lengthscale')
            })
        else:
            kwargs.update({
                'covar_module.outputscale': kwargs.get('outputscale'),
                'covar_module.base_kernel.lengthscale': kwargs.get('lengthscale')
            })
        kwargs.pop('outputscale', None)
        kwargs.pop('lengthscale', None)
        super().initialize_parameters(**kwargs)

    def get_common_hyperparameters(self):
        kernel = self.model.covar_module
        if self.model.uses_inducing_points:
            rbf_kernel = kernel.base_kernel.base_kernel
            scale_kernel = kernel
        else:
            rbf_kernel = kernel.base_kernel
            scale_kernel = kernel

        return {
            'outputscale': scale_kernel.outputscale.detach().numpy(),
            'lengthscale': rbf_kernel.lengthscale.detach().numpy(),
            'noise': self.noise if self.noise is not None else self.likelihood.noise.item(),
        }


class SGPR(DKLGPModel):
    """
    This class only exists to make the interface clear. 
    The inducing point kernel recides in gpr.py.

    https://gpytorch.readthedocs.io/en/latest/examples/05_Scalable_GP_Regression_Multidimensional/SGPR_Example_CUDA.html
    """
    def __init__(self, *args, gp_kwargs, **kwargs):
        default_gp_kwargs = dict(
            n_grid=None,
            inducing_points=100,
        )
        default_gp_kwargs.update(gp_kwargs)
        super().__init__(*args, gp_kwargs=default_gp_kwargs, **kwargs)
