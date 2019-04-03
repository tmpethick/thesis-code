import sys

from src.environments import BaseEnvironment
from src.models import BaseModel
from src.plot_utils import plot1D, plot2D, plot_function


# For some reason it breaks without TkAgg when running from CLI.
# import matplotlib
# matplotlib.use("TkAgg")

import hashlib
import json

import numpy as np
from sacred import Experiment
from sacred.observers import MongoObserver

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")

from src import models as models_module
from src import acquisition_functions as acquisition_functions_module
from src import environments as environments_module
from src import kernels as kernels_module
from src.algorithms import AcquisitionAlgorithm
from src.utils import mean_square_error, random_hypercube_samples
from src import settings


def unpack(config_obj):
    return config_obj['name'], \
           config_obj.get('arg', ()), \
           config_obj.get('kwargs', {})


def create_kernel(name, input_dim, kwargs):
    Class = getattr(kernels_module, name)
    return Class(input_dim, **kwargs)

def hash_subdict(d, keys=None):
    if keys is None:
        keys = []

    d = {k: d.get(k, {}) for k in keys}
    return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()


def create_ex():
    ex = Experiment(settings.EXP_NAME, interactive=settings.EXP_INTERACTIVE)
    ex.observers.append(MongoObserver.create(db_name='lions'))

    ex.add_config({
        'verbosity': {
            'plot': True
        }
    })


    @ex.capture
    def dklgpmodel_training_callback(model, i, loss, _log, _run):
        # TODO: save model
        # Log
        _log.info('Iter %d/%d - Loss: %.3f' % (i + 1, model.n_iter, loss))

        # Metrics
        _run.log_scalar('DKLGPModel.training.loss', loss, i)


    def create_model(name, input_dim, kwargs):
        if 'kernel' in kwargs:
            kern_name, kern_args, kern_kwargs = unpack(kwargs.pop('kernel'))
            kwargs2 = dict(
                kernel=create_kernel(kern_name, input_dim, kern_kwargs)
            )
            kwargs2.update(kwargs)
        else:
            kwargs2 = kwargs

        if name == 'DKLGPModel':
            kwargs2['training_callback'] = dklgpmodel_training_callback

        Class = getattr(models_module, name)
        return Class(**kwargs2)


    @ex.config_hook
    def add_unique_hashes(config, command_name, logger):
        """Create the unique hash
        """

        # Model hash (for same models across different objective functions)
        # It is important that:
        # 1) `kwarg` are left out if none are present
        # 2) Notice if e.g. `model2` is empty then {} is set as default.
        model_hash_keys = ['mode', 'model', 'model2', 'acquisition_function', 'bo']
        model_hash = hash_subdict(config, keys=model_hash_keys)

        # Experiment hash (used for one model + obj function pair)
        exp_hash_key = ['mode', 'obj_func', 'model', 'model2', 'acquisition_function', 'bo']
        exp_hash = hash_subdict(config, keys=exp_hash_key)

        return {'model_hash': model_hash, 'exp_hash': exp_hash}


    @ex.capture
    def log_info(msg, _log):
        _log.info(msg)


    @ex.capture
    def update_ex_result(result, _run):
        _run.result = result


    @ex.capture
    def save_fig(fig, filename, _run, verbosity):
        fig.savefig(filename)
        if verbosity['plot']:
            plt.show()
        _run.add_artifact(filename)


    @ex.capture
    def plot(algorithm: AcquisitionAlgorithm, i, _run, _log):
        # Log
        _log.info("... starting BO round {} / {}".format(i, algorithm.n_iter))

        # Metrics
        mse = mean_square_error(algorithm.models[0], algorithm.f)
        _run.log_scalar('mse', mse, i)

        # TODO: save weights

        # Update real-time info
        _run.result = mse

        # Save files
        if i % 5 == 0 or i == algorithm.n_iter:
            filename = settings.ARTIFACT_BO_PLOT_FILENAME.format(i=i)
            fig = algorithm.plot()
            if fig is not None:
                save_fig(fig, filename)
            # and show

            # Save observations
            X_filename = settings.ARTIFACT_INPUT_FILENAME
            Y_filename = settings.ARTIFACT_OUTPUT_FILENAME
            np.save(X_filename, algorithm.X)
            np.save(Y_filename, algorithm.Y)
            _run.add_artifact(X_filename)
            _run.add_artifact(Y_filename)


    def test_gp_model(f: BaseEnvironment, models: [BaseModel], acquisition_function=None, n_samples=15):
        bounds = f.bounds
        input_dim = f.input_dim

        if input_dim == 1:
            X = np.random.uniform(bounds[0, 0], bounds[0, 1], (n_samples, 1))
        else:
            X = random_hypercube_samples(n_samples ** input_dim, bounds)

        Y = f(X)

        for model in models:
            model.init(X, Y)

        all_mse = np.zeros(len(models))

        for i, model in enumerate(models):
            if input_dim == 1:
                fig = plot1D(model, f)
            elif input_dim == 2:
                fig = plot2D(model, f)
            else:
                fig = None

            if fig is not None:
                save_fig(fig, settings.ARTIFACT_GP_FILENAME.format(model_idx=i))

            # Acquisition
            if acquisition_function is not None:
                fig = plot_function(f, acquisition_function, title="Acquisition functions", points=X)
                save_fig(fig, settings.ARTIFACT_GP_ACQ_FILENAME.format(model_idx=i))

            # Log
            mse = mean_square_error(model, f)
            log_info('MSE for {} with idx {}: {}'.format(model, i, mse))
            all_mse[i] = mse

        update_ex_result(all_mse)


    @ex.main
    def main(_config, _run):
        ## Model construction

        # Create environment
        name, args, kwargs = unpack(_config['obj_func'])
        f = getattr(environments_module, name)(**kwargs)

        # Create the model
        models = [_config['model']]
        if 'model2' in _config:
            models.append(_config['model2'])

        models = [create_model(model['name'], f.input_dim, model['kwargs'])
                  for model in models]

        # Create acq func
        if _config.get('acquisition_function'):
            name, args, kwargs = unpack(_config['acquisition_function'])
            Acq = getattr(acquisition_functions_module, name)
            acq = Acq(*models, **kwargs)
        else:
            acq = None

        if _config.get('bo'):
            assert acq is not None, "Acquisition function required for BO"
            # Create BO
            bo_kwargs = _config['bo']
            bo = AcquisitionAlgorithm(f, models, acq, **bo_kwargs)

            bo.run(callback=plot)
        else:
            bo = None
            test_gp_model(f, models, acquisition_function=acq, n_samples=_config['gp_samples'])

        # Hack to have model available after run in interactive mode.
        _run.interactive_stash = {
            'model': models[0],
            'model2': models[0] if len(models) >= 2 else None,
            'acq': acq,
            'bo': bo,
        }
    return ex


def notebook_run(*args, **kwargs):
    ex = create_ex()
    return ex.run(*args, **kwargs)


if __name__ == '__main__':
    ex = create_ex()
    ex.run_commandline(sys.argv + ["--force"])
