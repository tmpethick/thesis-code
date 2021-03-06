import time
import math 
import os

import numpy as np
from matplotlib import pyplot as plt

from src.algorithms import AcquisitionAlgorithm
from src.environments import BaseEnvironment, EnvironmentNormalizer
from src.environments.dataset import DataSet
from src.experiment import settings
from src.models.core_models import MarginalLogLikelihoodMixin
from src.models import NormalizerModel, DKLGPModel, GPModel, TransformerModel, LocalLengthScaleGPModel, FeatureModel, SaveMixin
from src.plot_utils import plot_model, plot_model_unknown_bounds, plot_function
from src.utils import random_hypercube_samples, calc_errors_model_compare_mean, calc_errors_model_compare_var, \
    calc_errors, errors
from src.models.ASG import ControlledLocationsModelMixin


class Runner(object):
    def __init__(self, context, _log, _run):
        self.context = context
        self._log = _log
        self._run = _run

    def run(self):
        f = self.context.obj_func
        bo = self.context.bo
        models = self.context.models

        if bo is not None:
            bo.run(callback=self.plot)
        elif isinstance(f, DataSet):
            if hasattr(f, 'X_post_train'):
                X_post_train = f.X_post_train
                Y_post_train = f.Y_post_train
            else:
                X_post_train = None
                Y_post_train = None
            self.run_models(models, f.X_train, f.Y_train, None, f.X_test, f.Y_test, X_post_train, Y_post_train)

            # Plot models
            for i, model in enumerate(self.context.models):
                # For natural sound
                if f.input_dim == 1:
                    # Plot only a subset of the data
                    n = 1000
                    stride = int(math.ceil(len(f.X_train) / n))
                    X, Y = f.X_train[::stride], f.Y_train[::stride]

                    X_line = np.linspace(f.bounds[0,0], f.bounds[0,1], n)
                    Y_hat, _ = model.get_statistics(X_line, full_cov=False)
                    fig = plt.figure()
                    ax = fig.add_subplot(211)
                    ax.plot(X, Y)
                    ax = fig.add_subplot(212)
                    ax.plot(X_line, Y_hat)
                # elif f.input_dim == 2:
                #     Y_hat, _ = model.get_statistics(f.X_test, full_cov=False)
                #     fig = plt.figure()
                #     ax = fig.add_subplot(121, projection="3d")
                #     ax.scatter(f.X_test[..., 0], f.X_test[..., 0], f.Y_test[..., 0])
                #     ax.scatter(f.X_test[..., 0], f.X_test[..., 0], Y_hat[..., 0])
                #     ax = fig.add_subplot(122, projection="3d")
                #     ax.scatter(f.X_test[..., 0], f.X_test[..., 0], np.abs(f.Y_test - Y_hat)[..., 0])
                else:
                    fig = None  

                if fig is not None:
                    self.save_fig(fig, settings.ARTIFACT_GP_FILENAME.format(model_idx=i))

        elif isinstance(f, BaseEnvironment):
            X_train, Y_train, Y_train_dir, X_test, Y_test, X_post_train, Y_post_train = self.get_data_f(f)
            self.run_models(models, X_train, Y_train, Y_train_dir, X_test, Y_test, X_post_train, Y_post_train)
            
            if not (hasattr(f, 'is_expensive') and f.is_expensive):
                self.plot_models(self.context)

        for i, model in enumerate(self.context.models):
            self.save_model(model, i)
    
    def save_model(self, model, i):
        if self.context.save_model and isinstance(model, SaveMixin):
            model.save(os.path.join(settings.MODEL_SNAPSHOTS_DIR, self.context.exp_hash, str(i)))


    def get_data_f(self, f: BaseEnvironment):
        # Training
        bounds = f.bounds
        input_dim = f.input_dim

        # TODO: currently only based on the first model.
        if isinstance(self.context.model, ControlledLocationsModelMixin):
            X_train = None
            Y_train = None
        else:
            assert isinstance(self.context.gp_samples, int), "gp_samples need to be an int"
            if self.context.use_sample_grid:
                X_train = np.mgrid[[slice(axis[0], axis[1], self.context.gp_samples*1j) for axis in bounds]]
                X_train = np.moveaxis(X_train, 0, -1)
                X_train = np.reshape(X_train, (-1, X_train.shape[-1]))
            else:
                if input_dim == 1:
                    X_train = np.random.uniform(bounds[0, 0], bounds[0, 1], (self.context.gp_samples, 1))
                else:
                    X_train = random_hypercube_samples(self.context.gp_samples, bounds)
            Y_train = f(X_train)

        if self.context.gp_use_derivatives:
            Y_train_dir = f.derivative(X_train)
        else:
            Y_train_dir = None

        # Testing
        X_test = random_hypercube_samples(self.context.gp_test_samples, bounds, rng=np.random.RandomState(1))
        Y_test = f.noiseless(X_test)

        return X_train, Y_train, Y_train_dir, X_test, Y_test, None, None


    def run_models(self, models, X_train, Y_train, Y_train_dir, X_test, Y_test, X_post_train=None, Y_post_train=None):
        f = self.context.obj_func
        for i, model in enumerate(models):
            start_time = time.clock()
            if isinstance(model, ControlledLocationsModelMixin):
                X_train, Y_train = model.fit(f)
                print('samples:', X_train.shape)
            else:
                self.log_info('Model{}: {} training on {} of dim {}'.format(i, model, len(X_train), X_train.shape[-1]))
                model.init(X_train, Y_train, Y_dir=Y_train_dir)
                if X_post_train is not None:
                    model.set_train_data(X_post_train, Y_post_train)
            training_time = time.clock() - start_time

            # Test
            self.log_info('Model{}: {} predicting on {} of dim {}'.format(i, model, len(X_test), X_test.shape[-1]))
            N = len(Y_test)
            start_time = time.clock()
            Y_hat, var = model.get_statistics(X_test, full_cov=False)
            pred_time = time.clock() - start_time

            err = errors(Y_hat, var, Y_test, np.mean(Y_train))

            if i == 0:
                prefix = ''
            else:
                prefix = f'model{i}.'
            
            self.log_scalar(f'{prefix}mae', err['mae'], 0)
            self.log_scalar(f'{prefix}max_err', err['max_err'], 0)
            self.log_scalar(f'{prefix}rmse', err['rmse'], 0)
            self.log_scalar(f'{prefix}mnlp', err['mnlp'], 0)
            self.log_scalar(f'{prefix}nmse', err['nmse'], 0)
            self.log_scalar(f'{prefix}time.training', training_time, 0)
            self.log_scalar(f'{prefix}time.pred', pred_time, 0)

            if isinstance(model, MarginalLogLikelihoodMixin):
                mll = model.get_marginal_log_likelihood(X_train, Y_train)
                self.log_scalar(f'{prefix}mll', mll, 0)
                pmll = model.get_marginal_log_likelihood(X_test, Y_test)
                self.log_scalar(f'{prefix}pmll', pmll, 0)
                self.log_info(f'Model{i}: {model} has mll={mll} pmll={pmll}')

            self.log_info('Model{}: {} has {}'.format(i, model, err))

            # Log DKLGPModel specifics (bypassing NormalizerModel)
            if isinstance(model, NormalizerModel):
                true_model = model.model
            else:
                true_model = model

            if isinstance(true_model, DKLGPModel) or isinstance(true_model, GPModel):
                hyperparameters = true_model.get_common_hyperparameters()
                self.log_info('Model{} has parameters: {}'.format(i, hyperparameters))
                self.update_result('hyperparameters', hyperparameters)

            if hasattr(true_model, 'warnings') and len(true_model.warnings) > 0:
                self.update_result('WARNING', true_model.warnings)

    def plot_models(self, context):
        f = context.obj_func
        X = context.model.X

        if context.model_compare:
            # TODO: For now only supports 2 models
            model1 = context.model
            model2 = context.model2
            rmse, max_err = calc_errors_model_compare_mean(model1, model2, f)
            self.log_scalar('model_compare.mean.rmse', rmse, 0)
            self.log_scalar('model_compare.mean.max_err', max_err, 0)

            rmse, max_err = calc_errors_model_compare_var(model1, model2, f)
            self.log_scalar('model_compare.var.rmse', rmse, 0)
            self.log_scalar('model_compare.var.max_err', max_err, 0)

        for i, model in enumerate(context.models):
            fig = plot_model(model, f)

            if fig is not None:
                self.save_fig(fig, settings.ARTIFACT_GP_FILENAME.format(model_idx=i))

            # Transformed model
            # TODO: Plot sampled
            if isinstance(model, NormalizerModel):
                true_model = model.model
                normalized_f = EnvironmentNormalizer(f, model.X_normalizer, model.Y_normalizer)

                if f.input_dim == 1:
                    fig = plot_model_unknown_bounds(true_model)
                    self.save_fig(fig, settings.ARTIFACT_UNNORMALIZED_FILENAME.format(model_idx=i))
            else:
                true_model = model
                normalized_f = f

            if isinstance(true_model, TransformerModel):
                try:
                    subspace_dim = true_model.transformer.output_dim
                except:
                    pass
                else:
                    if subspace_dim <= 2:
                        X_test = random_hypercube_samples(1000, normalized_f.bounds)
                        Y_test = normalized_f.noiseless(X_test)
                        X_trans = true_model.transformer.transform(X_test)
                        mean, _ = true_model.prob_model.get_statistics(X_trans)

                        # TODO: move (and include variance)
                        fig = plt.figure()
                        if X_trans.shape[-1] == 1:
                            ax = fig.add_subplot(111)
                            ax.scatter(X_trans, Y_test, s=2)
                            ax.scatter(X_trans, mean, marker="1", s=2)
                        if X_trans.shape[-1] == 2:
                            ax = fig.add_subplot(111, projection='3d')
                            ax.scatter(X_trans[:, 0], X_trans[:, 1], mean, marker="1", color="red", s=2)
                            ax.scatter(X_trans[:, 0], X_trans[:, 1], Y_test, s=2)
                        self.save_fig(fig, settings.ARTIFACT_AS_FEATURES_FILENAME.format(model_idx=i))

            # Acquisition
            if context.acquisition_function is not None:
                fig = plot_function(normalized_f, context.acquisition_function, title="Acquisition functions", points=X)
                self.save_fig(fig, settings.ARTIFACT_GP_ACQ_FILENAME.format(model_idx=i))

            # Length scale
            if isinstance(true_model, LocalLengthScaleGPModel):
                fig = plot_function(normalized_f, lambda x: 1 / true_model.get_lengthscale(x)[:, None],
                                    title="1/Lengthscale", points=X)
                self.save_fig(fig, settings.ARTIFACT_LLS_GP_LENGTHSCALE_FILENAME.format(model_idx=i))

            # Plot feature space for DKLGP
            elif isinstance(true_model, FeatureModel):
                fig = true_model.plot_features(normalized_f)
                if fig is not None:
                    self.save_fig(fig, settings.ARTIFACT_DKLGP_FEATURES_FILENAME.format(model_idx=i))

                if context.verbosity['plot']:
                    plt.plot(true_model.training_loss)
                    plt.show()

    def log_info(self, msg):
        self._log.info(msg)

    def save_fig(self, fig, filename):
        fig.savefig(filename)
        if self.context.verbosity['plot']:
            plt.show()
        self._run.add_artifact(filename)
        plt.clf()

    def update_result(self, name, value):
        # Update the result dict with the latest value (notice `step` is ignored).
        result = self._run.result
        if type(result) is not dict:
            result = {}
        result[name] = value
        self._run.result = result

    def log_scalar(self, name, value, step):
        self._run.log_scalar(name, value, step)

        # Mongodb does not allow `.` in the key for a regular entry.
        name = name.replace(".", ":")
        self.update_result(name, value)

    def plot(self, algorithm: AcquisitionAlgorithm, i):
        # Log
        self._log.info("... starting BO round {} / {}".format(i, algorithm.n_iter))

        # Metrics
        rmse, max_err = calc_errors(algorithm.models[0], algorithm.f, rand=True)
        self.log_scalar('rmse', rmse, i)
        self.log_scalar('max_err', max_err, i)

        # TODO: save weights

        # Save files
        if i % 5 == 0 or i == algorithm.n_iter:
            filename = settings.ARTIFACT_BO_PLOT_FILENAME.format(i=i)
            fig = algorithm.plot()
            if fig is not None:
                self.save_fig(fig, filename)
            # and show

            # Save observations
            X_filename = settings.ARTIFACT_INPUT_FILENAME
            Y_filename = settings.ARTIFACT_OUTPUT_FILENAME
            np.save(X_filename, algorithm.X)
            np.save(Y_filename, algorithm.Y)
            self._run.add_artifact(X_filename)
            self._run.add_artifact(Y_filename)
