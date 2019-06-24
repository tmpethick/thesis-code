import time

import numpy as np
from matplotlib import pyplot as plt

from src.algorithms import AcquisitionAlgorithm
from src.environments import DataSet, BaseEnvironment, EnvironmentNormalizer
from src.experiment import settings
from src.models import NormalizerModel, DKLGPModel, GPModel, TransformerModel, LocalLengthScaleGPModel
from src.plot_utils import plot_model, plot_model_unknown_bounds, plot_function
from src.utils import random_hypercube_samples, calc_errors_model_compare_mean, calc_errors_model_compare_var, \
    calc_errors
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
            self.run_models(models, f.X_train, f.Y_train, None, f.X_val, f.Y_val)
        elif isinstance(f, BaseEnvironment):
            X_train, Y_train, Y_train_dir, X_val, Y_val = self.get_data_f(f)
            self.run_models(models, X_train, Y_train, Y_train_dir, X_val, Y_val)
            
            if not (hasattr(f, 'is_expensive') and f.is_expensive):
                self.plot_models(self.context)

    def get_data_f(self, f: BaseEnvironment):
        # Training
        assert isinstance(self.context.gp_samples, int), "n_samples need to be an int"
        bounds = f.bounds
        input_dim = f.input_dim

        # TODO: currently only based on the first model.
        if isinstance(self.context.model, ControlledLocationsModelMixin):
            X_train = None
            Y_train = None
        elif self.context.use_sample_grid:
            X_train = np.mgrid[[slice(axis[0], axis[1], self.context.gp_samples*1j) for axis in bounds]]
            X_train = np.moveaxis(X_train, 0, -1)
            X_train = np.reshape(X_train, (-1, X_train.shape[-1]))
            Y_train = f(X_train)
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
        X_val = random_hypercube_samples(self.context.gp_test_samples, bounds, rng=np.random.RandomState(1))
        Y_val = f(X_val)

        return X_train, Y_train, Y_train_dir, X_val, Y_val


    def run_models(self, models, X_train, Y_train, Y_train_dir, X_val, Y_val):
        f = self.context.obj_func
        for i, model in enumerate(models):
            start_time = time.clock()
            if isinstance(model, ControlledLocationsModelMixin):
                model.fit(f)
            else:
                model.init(X_train, Y_train, Y_dir=Y_train_dir)
            training_time = time.clock() - start_time

            # Test
            N = len(Y_val)
            start_time = time.clock()
            Y_hat, var = model.get_statistics(X_val, full_cov=False)
            pred_time = time.clock() - start_time

            rmse = np.sqrt(np.sum(np.square(Y_hat - Y_val)) / N)
            max_err = np.max(np.fabs(Y_hat - Y_val))

            self.log_info('Model{}: {} has RMSE={} max_err={}'.format(i, model, rmse, max_err))

            if i == 0:
                self.log_scalar('rmse', rmse, 0)
                self.log_scalar('max_err', max_err, 0)
                self.log_scalar('time.training', training_time, 0)
                self.log_scalar('time.pred', pred_time, 0)
            else:
                self.log_scalar('model{}.rmse'.format(i), rmse, 0)
                self.log_scalar('model{}.max_err'.format(i), max_err, 0)
                self.log_scalar('model{}.time.training'.format(i), training_time, 0)
                self.log_scalar('model{}.time.pred'.format(i), pred_time, 0)

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
            elif isinstance(true_model, DKLGPModel):
                fig = true_model.plot_features(normalized_f)
                if fig is not None:
                    self.save_fig(fig, settings.ARTIFACT_DKLGP_FEATURES_FILENAME.format(model_idx=i))

    def log_info(self, msg):
        self._log.info(msg)

    def save_fig(self, fig, filename):
        fig.savefig(filename)
        if self.context.verbosity['plot']:
            plt.show()
        self._run.add_artifact(filename)

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
