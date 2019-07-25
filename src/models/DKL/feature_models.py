import warnings
import pickle
import os

import torch
import gpytorch
import numpy as np
from matplotlib import pyplot as plt
import pathlib

from src.experiment.config_helpers import ConfigMixin, lazy_construct_from_module, LazyConstructor
from ..core_models import BaseModel, MarginalLogLikelihoodMixin
from .gpr import GPRegressionModel
from .feature_extractors import LargeFeatureExtractor, RFFEmbedding
from src.utils import construct_2D_grid, call_function_on_grid, random_hypercube_samples


if torch.cuda.is_available():
    print("Using GPU")
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def default_training_callback(model, i, loss):
    if i % 20 == 0:
        print('Iter %d/%d - Loss: %.3f' % (i + 1, model.n_iter, loss))


class FeatureModel(BaseModel):
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
        Z = Z.detach().cpu().numpy()
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

            # Use the first subfigure for training data instead
            X_train = self.X
            Z_train = self.get_features(X_train)
            O_train = self.Y
            O_train_pred, _ = self.get_statistics(X_train, full_cov=False)

            ax = fig.add_subplot(121)
            ax.set_title('Training data in feature space')
            c = np.linspace(0.0, 1.0, Z_train.shape[0])
            ax.scatter(Z_train.flatten(), O_train.flatten(), label="ground truth", marker="x")
            ax.scatter(Z_train.flatten(), O_train_pred.flatten(), c=c, cmap='viridis', label='prediction', marker=".")
            ax.legend()


        if Z.shape[-1] == 1:
            ax = fig.add_subplot(122)
            ax.set_title('f in feature space')
            if self.X.shape[-1] == 1:
                O = f.noiseless(X_line)
                O_hat, O_hat_var = self.get_statistics(X_line, full_cov=False)

                #ax.scatter(Z.flatten(), O_hat.flatten(), label="prediction")
                #plt.errorbar(Z.flatten(), O_hat.flatten(), yerr=2 * np.sqrt(O_hat_var), fmt='.k', alpha=0.2, c=c, cmap='viridis', label='prediction')
            else:
                O = np.reshape(O, (-1, 1))
                O_hat, O_hat_var = self.get_statistics(X, full_cov=False)
            

            c = np.linspace(0.0, 1.0, Z.shape[0])
            ax.scatter(Z.flatten(), O.flatten(), label="ground truth", marker="x")
            ax.scatter(Z.flatten(), O_hat.flatten(), c=c, cmap='viridis', label='prediction', marker=".")
            ax.legend()

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


class LinearFromFeatureExtractor(ConfigMixin, FeatureModel):
    """Not really DNGO as it does not use a bayesian linear regressor.
    """
    def __init__(self,
        feature_extractor=None,
        nn_kwargs=None,
        feature_extractor_constructor=LazyConstructor(LargeFeatureExtractor),
        n_iter=50,
        learning_rate=0.1,
        weight_decay=0.0,
        use_double_precision=False,
        training_callback=default_training_callback,
        **kwargs):
        super().__init__(**kwargs)

        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.model = None
        self.feature_extractor = feature_extractor
        self.use_double_precision = use_double_precision
        self.weight_decay = weight_decay

        self.training_loss = None
        
        # Constructor is only used if feature_extractor is not specified.
        if nn_kwargs is not None:
            self.feature_extractor_constructor = LazyConstructor(LargeFeatureExtractor, **nn_kwargs)
        elif feature_extractor_constructor is not None:
            self.feature_extractor_constructor = feature_extractor_constructor
        else:
            raise ValueError("Either feature_extractor_constructor or nn_kwargs should be specified")

        self.loss = torch.nn.MSELoss()
        self.training_callback = training_callback

    def _fit(self, X, Y, Y_dir=None):
        D = X.shape[-1]

        if self.feature_extractor is None:
            self.feature_extractor = self.feature_extractor_constructor(D=D)
            self.feature_extractor = self.make_double(self.feature_extractor)

        if self.model is None:
            self.model = torch.nn.Sequential(
                self.feature_extractor,
                torch.nn.Linear(in_features=self.feature_extractor.output_dim, out_features=1, bias=True)
            )
            self.model = self.make_double(self.model)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
            self.model.train()


        # Move tensors to the configured device
        X_torch = self.to_torch(X)
        Y_torch = self.to_torch(Y)

        self.training_loss = np.zeros(self.n_iter + 1)
        # Train the model
        for i in range(self.n_iter):
            # Forward pass
            Y_pred = self.model(X_torch)
            loss = self.loss(Y_pred, Y_torch)

            # Backward and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_ = loss.item()
            self.training_loss[i+1] = loss_
            self.training_callback(self, i, loss_)

    def _get_statistics(self, X, full_cov=True):
        self.model.eval()

        X = self.to_torch(X)

        with torch.no_grad():
            mean = self.model(X).detach().cpu().numpy()

        N = mean.shape[0]

        if full_cov:
            return mean, np.zeros((N, N, 1))
        else:
            return mean, np.zeros((N, 1))


class GPyTorchModel(MarginalLogLikelihoodMixin, ConfigMixin, FeatureModel):
    def __init__(self,
                 n_iter=50,
                 learning_rate=0.1,
                 noise=None,
                 noise_lower_bound=1e-4,
                 initial_parameters=None,

                 do_pretrain=False,
                 pretrain_n_iter=100,
                 gp_n_iter=None,
                 pretrainer_constructor=LazyConstructor(LinearFromFeatureExtractor),

                 feature_extractor_constructor=LazyConstructor(LargeFeatureExtractor),
                 gp_constructor=LazyConstructor(GPRegressionModel),

                 use_double_precision=False,
                 covar_root_decomposition=True,
                 use_cg=False,
                 max_cg_iter=1000,
                 precond_size=10,
                 eval_cg_tolerance=1e-4,
                 train_eval_cg_tolerance=1.0,
                 grad_clip=None,
                 use_toeplitz=None,
                 weight_decay=0.0,

                 training_callback=default_training_callback,
                 verbose=True,
                 **kwargs):

        super().__init__(**kwargs)
        self.use_double_precision = use_double_precision
        self.covar_root_decomposition = covar_root_decomposition
        self.use_cg = use_cg
        self.max_cg_iter = max_cg_iter
        self.precond_size = precond_size
        self.use_toeplitz = use_toeplitz # infered by inducing points if not specified.
        self.weight_decay = weight_decay
        self.eval_cg_tolerance = eval_cg_tolerance
        self.train_eval_cg_tolerance = train_eval_cg_tolerance
        
        self.grad_clip = grad_clip

        self.has_feature_map = feature_extractor_constructor is not None
        self.feature_extractor_constructor = feature_extractor_constructor
        self.gp_constructor = gp_constructor

        self.do_pretrain = do_pretrain
        self.pretrain_n_iter = pretrain_n_iter
        self.gp_n_iter = gp_n_iter
        self.pretrainer_constructor = pretrainer_constructor

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
        self.verbose = verbose

        self.warnings = {}
        self.is_trained = False

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
        if np.isnan(X).any() or np.isnan(X).any():
            warnings.warn("Training data contains NaN! This might prevent GPyTorch from completing training.", UserWarning)
        # catch warning, save and stop (runner should store this warning)
        self._train(X, Y, fix_gp_params=False)

    def _train(self, X, Y, fix_gp_params=False):
        n, d = X.shape
        self.is_trained = True

        self.X_torch = self.to_torch(X)
        self.Y_torch = self.to_torch(Y[:, 0])

        if self.has_feature_map:
            self.feature_extractor = self.feature_extractor_constructor(D=d).to(device)
            self.feature_extractor = self.make_double(self.feature_extractor)

        if self.do_pretrain:
            pretrainer_network = self.pretrainer_constructor(feature_extractor=self.feature_extractor, n_iter=self.pretrain_n_iter, weight_decay=self.weight_decay, learning_rate=self.learning_rate)
            # TODO: reuse X_torch instead of copying again from numpy to torch.
            pretrainer_network.init(X, Y)

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

        if self.use_toeplitz is None:
            if self.model.uses_inducing_points:
                self.use_toeplitz = True
            else:
                self.use_toeplitz = False

        if self.initial_parameters is not None:
            print(self.initial_parameters)
            self.initialize_parameters(**self.initial_parameters)
        
        if self.gp_n_iter is not None:
            self.optimize(fix_feature_params=True, n_iter=self.gp_n_iter)
        self.optimize(fix_gp_params=fix_gp_params)

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

    def optimize(self, fix_gp_params=False, fix_feature_params=False, n_iter=None): 
        if n_iter is None:
            n_iter = self.n_iter

        X = self.X_torch
        Y = self.Y_torch
        if self.verbose:
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

        if self.has_feature_map and not fix_feature_params:
            self.feature_extractor.train()
            opt_parameter_list.append({
                'params': self.model.feature_extractor.parameters(),
                'weight_decay': self.weight_decay,
            })

        # optimize with Adam
        optimizer = torch.optim.Adam(opt_parameter_list, lr=self.learning_rate)

        # "Loss" for GPs - the marginal log likelihood
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        def _train():
            self.training_loss = np.zeros(n_iter + 1)
            min_loss = np.inf
            min_state = None

            for i in range(n_iter):
                # Zero backprop gradients
                optimizer.zero_grad()
                # Get output from model
                output = self.model(X)
                # Calc loss and backprop derivatives
                loss = -self.mll(output, Y)
                print(loss)
                loss.backward()

                loss_ = loss.item()

                if loss_ < min_loss:
                    min_state = self.model.state_dict()
                    min_loss = loss_

                if self.verbose:
                    if i % 30 == 0:
                        print(f'Loss {loss_} with hyperparameters: {self.get_common_hyperparameters()}')
                self.training_loss[i+1] = loss_
                self.training_callback(self, i, loss_)

                if self.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm(self.model.parameters(), self.grad_clip)
                optimizer.step()
            
            if min_state is not None:
                self.model.load_state_dict(min_state)
                if self.verbose:
                    print("Using parameters with loss:", min_loss)

        with gpytorch.settings.use_toeplitz(self.use_toeplitz), \
            gpytorch.settings.fast_computations(covar_root_decomposition=self.covar_root_decomposition, log_prob=self.use_cg, solves=self.use_cg), \
            gpytorch.settings.max_cg_iterations(self.max_cg_iter), \
            gpytorch.settings.max_preconditioner_size(self.precond_size), \
            gpytorch.settings.eval_cg_tolerance(self.train_eval_cg_tolerance):

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

    def _get_multivariate_normal(self, X):
        if self.verbose:
            print('predicting {} points using {} training points'.format(X.shape[0], self.X_torch.shape[0]))
        assert self.model is not None, "Call `self.fit` before predicting."

        # Go into prediction mode
        self.model.eval()
        self.likelihood.eval()

        if self.has_feature_map:
            self.feature_extractor.eval()

        test_x = self.to_torch(X)

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

                self.store_CG_warning('pred', w)
        return multivariate_normal

    def _get_statistics(self, X, full_cov=True):
        # # Hack to fix issue with making prediction for single inputs (n=1).
        # # Only needed if approximate grid interpolation is used.
        # if X.shape[0] == 1:
        #    fake_X = np.zeros((1, X.shape[1]))
        #    X = np.concatenate((X, fake_X), axis=0)
        #    cut_tail = -1
        # else:
        cut_tail = None
        
        multivariate_normal = self._get_multivariate_normal(X)

        mean = multivariate_normal.mean.detach().cpu().numpy()[:cut_tail, None]

        if full_cov:
            return mean, multivariate_normal.covariance_matrix.detach().cpu().numpy()[:cut_tail, :cut_tail, None]
        else:
            return mean, multivariate_normal.variance.detach().cpu().numpy()[:cut_tail, None]

    def get_marginal_log_likelihood(self, X, Y):
        X = self.to_torch(X)
        Y = self.to_torch(Y[:,0])

        output = self.model(X)
        return self.mll(output, Y).item()

    def sample(self, X, size=1, prior=True):
        with gpytorch.settings.prior_mode(prior):
            multivariate_normal = self._get_multivariate_normal(X)
            return multivariate_normal.sample(size).detach().cpu().numpy()

    def initialize_parameters(self, noise=None, **kwargs):
        """Overwrite this for custom easy initialization of custom kernels.
        """
        if isinstance(self.likelihood, gpytorch.likelihoods.FixedNoiseGaussianLikelihood):
          assert noise is None, "Only init noise if not fixed."
        kwargs.update({
            'likelihood.noise': noise
        })
        self.model.initialize(**kwargs)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["model"] = None
        state["feature_extractor"] = None
        state["likelihood"] = None
        state["mll"] = None
        state["X_torch"] = None
        state["Y_torch"] = None
        state["training_callback"] = default_training_callback
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def save(self, PATH):
        super().save(PATH)

        if self.is_trained:
            np.save(os.path.join(PATH, 'X.npy'), self.X)
            np.save(os.path.join(PATH, 'Y.npy'), self.Y)
        
            torch.save(self.model.state_dict(), os.path.join(PATH, 'parameters.pt'))

    def post_load_hook(self, PATH):
        if self.is_trained:
            X = np.load(os.path.join(PATH, 'X.npy'))
            Y = np.load(os.path.join(PATH, 'Y.npy'))

            # Construct model.model (and avoiding training/optimization)
            n_iter_backup = self.n_iter
            self.n_iter = 0
            do_pretrain_backup = self.do_pretrain
            self.do_pretrain = False
            self.init(X, Y)
            self.n_iter = n_iter_backup
            self.do_pretrain = do_pretrain_backup

            # This will also load model.model.likelihood and model.model.feature_extractor
            state_dict = torch.load(os.path.join(PATH, 'parameters.pt'))
            self.model.load_state_dict(state_dict)

            # Replace with versions that has parameters loaded.
            self.likelihood = self.model.likelihood
            self.feature_extractor = self.model.feature_extractor

        return self


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
            'lengthscale': self.model.feature_extractor.lengthscale.detach().cpu().numpy(),
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
        if self.model.n_grid is not None:
            kwargs.update({
                'covar_module.outputscale': kwargs.get('outputscale'),
                'covar_module.base_kernel.base_kernel.lengthscale': kwargs.get('lengthscale')
            })
        elif self.model.inducing_points is not None:
            kwargs.update({
                'covar_module.base_kernel.outputscale': kwargs.get('outputscale'),
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
        if self.model.n_grid is not None:
            rbf_kernel = kernel.base_kernel.base_kernel
            scale_kernel = kernel
        elif self.model.inducing_points is not None:
            rbf_kernel = kernel.base_kernel.base_kernel
            scale_kernel = kernel.base_kernel
        else:
            rbf_kernel = kernel.base_kernel
            scale_kernel = kernel

        return {
            'outputscale': scale_kernel.outputscale.detach().cpu().numpy(),
            'lengthscale': rbf_kernel.lengthscale.detach().cpu().numpy(),
            'noise': self.noise if self.noise is not None else self.likelihood.noise.item(),
        }

    @classmethod
    def process_config(cls, *, gp_kwargs=None, **kwargs):
        if isinstance(gp_kwargs, dict) and 'kernel' in gp_kwargs:
            kernel = gp_kwargs['kernel']
            gp_kwargs_updated = gp_kwargs.copy()
            gp_kwargs_updated['kernel'] = lazy_construct_from_module(gpytorch.kernels, kernel)
            return dict(
                gp_kwargs=gp_kwargs_updated,
                **kwargs)
        else:
            return dict(gp_kwargs=gp_kwargs, **kwargs)


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


class DNNBLR(DKLGPModel):
    def __init__(self, *args, gp_kwargs=None, **kwargs):
        defaults = dict(
            covar_root_decomposition=True
            )
        default_gp_kwargs = dict(
            n_grid=None,
            inducing_points=None,
            has_scale_kernel=False,
            kernel=LazyConstructor(gpytorch.kernels.LinearKernel)
        )
        defaults.update(kwargs)
        default_gp_kwargs.update(gp_kwargs or {})
        super().__init__(*args, gp_kwargs=default_gp_kwargs, **defaults)

    def initialize_parameters(self, lengthscale=None, variance=None, noise=None):
        return {
            'covar_module.variance': variance,
            'likelihood.noise': noise
        }

    def get_common_hyperparameters(self):
        return {
            'variance': self.model.covar_module.variance,
            'noise': self.noise if self.noise is not None else self.likelihood.noise.item(),
        }
