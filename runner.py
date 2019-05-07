import sys
import subprocess

from src.environments import BaseEnvironment
from src.models.models import BaseModel
from src.models.lls_gp import LocalLengthScaleGPModel
from src.models.dkl_gp import DKLGPModel
from src.plot_utils import plot1D, plot2D, plot_function, plot_model

# For some reason it breaks without TkAgg when running from CLI.
# import matplotlib
# matplotlib.use("TkAgg")

import hashlib
import json
import itertools

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
from src import algorithms as algorithm_module
from src.algorithms import AcquisitionAlgorithm
from src.utils import root_mean_square_error, random_hypercube_samples, construct_2D_grid, call_function_on_grid
from src import settings
from GPy.kern import Kern


def unpack(config_obj):
    return config_obj['name'], \
           config_obj.get('arg', ()), \
           config_obj.get('kwargs', {})


def create_kernel(name, kwargs, input_dim):
    print(name, kwargs, input_dim)
    Class = getattr(kernels_module, name)
    if issubclass(Class, Kern):
        return Class(input_dim, **kwargs)
    else:
        return Class(**kwargs)

def hash_subdict(d, keys=None):
    if keys is None:
        keys = []

    d = {k: d.get(k, {}) for k in keys}
    return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()


def create_ex(interactive=False):
    ex = Experiment(settings.EXP_NAME, interactive=interactive)

    if settings.SAVE:
        ex.observers.append(MongoObserver.create(url=settings.MONGO_DB_URL, db_name=settings.MONGO_DB_NAME))

    ex.add_config({
        'gp_use_derivatives': False,
        'verbosity': {
            'plot': True,
            'bo_show_iter': 30,
        }
    })


    @ex.capture
    def dklgpmodel_training_callback(model, i, loss, _log, _run):
        # TODO: save modelgp_use_derivatives
        # Log
        _log.info('Iter %d/%d - Loss: %.3f' % (i + 1, model.n_iter, loss))

        # Metrics
        _run.log_scalar('DKLGPModel.training.loss', loss, i)


    def create_model(name, kwargs, input_dim=None):
        # TODO: avoid this ad-hoc case...
        if name == 'TransformerModel':
            transformer = kwargs['transformer']
            prob_model = kwargs['prob_model']
            transformer = create_model(transformer['name'], transformer['kwargs'])
            # Currently only transformers with fixed output is supported.
            kwargs2 = {
                'transformer': transformer,
                'prob_model': create_model(prob_model['name'], 
                                           prob_model['kwargs'],
                                           input_dim=transformer.output_dim),
            }
        else:
            if 'kernel' in kwargs:
                assert input_dim is not None, "input_dim is required if a kernel is specified."
                kern_name, kern_args, kern_kwargs = unpack(kwargs.pop('kernel'))
                kwargs2 = dict(
                    kernel=create_kernel(kern_name, kern_kwargs, input_dim)
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
        rmse = root_mean_square_error(algorithm.models[0], algorithm.f, rand=True)
        _run.log_scalar('rmse', rmse, i)

        # TODO: save weights

        # Update real-time info
        _run.result = rmse

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


    @ex.capture
    def test_gp_model(f: BaseEnvironment, models: [BaseModel], _run, acquisition_function=None, n_samples=15, use_derivatives=False):
        bounds = f.bounds
        input_dim = f.input_dim

        if input_dim == 1:
            X = np.random.uniform(bounds[0, 0], bounds[0, 1], (n_samples, 1))
        else:
            X = random_hypercube_samples(n_samples, bounds)

        Y = f(X)

        if use_derivatives:
            Y_dir = f.derivative(X)
            for model in models:
                model.init(X, Y, Y_dir=Y_dir)
        else:
            for model in models:
                model.init(X, Y)


        all_rmse = np.zeros(len(models))

        for i, model in enumerate(models):
            fig = plot_model(model, f)

            if fig is not None:
                save_fig(fig, settings.ARTIFACT_GP_FILENAME.format(model_idx=i))

            # Transformed model
            # if isinstance(model, TransformerModel):
            #     model.transform
            #     fig = plot_function(f, acquisition_function, title="Acquisition functions", points=X)
            #     save_fig(fig, settings.ARTIFACT_GP_ACQ_FILENAME.format(model_idx=i))

            # Acquisition
            if acquisition_function is not None:
                fig = plot_function(f, acquisition_function, title="Acquisition functions", points=X)
                save_fig(fig, settings.ARTIFACT_GP_ACQ_FILENAME.format(model_idx=i))

            # Length scale
            if isinstance(model, LocalLengthScaleGPModel):
                fig = plot_function(f, lambda x: 1 / model.get_lengthscale(x)[:,None], title="1/Lengthscale", points=X)
                save_fig(fig, settings.ARTIFACT_LLS_GP_LENGTHSCALE_FILENAME.format(model_idx=i))
            
            # Plot feature space for DKLGP
            elif isinstance(model, DKLGPModel):
                fig = model.plot_features(f)
                if fig is not None:
                    save_fig(fig, settings.ARTIFACT_DKLGP_FEATURES_FILENAME.format(model_idx=i))

            # Log
            rmse = root_mean_square_error(model, f, rand=True)
            log_info('MSE for {} with idx {}: {}'.format(model, i, rmse))
            all_rmse[i] = rmse

        # Only store result for `model` (not `model2`) as it is rarely used on its own.
        _run.result = all_rmse[0]


    @ex.main
    def main(_config, _run, _log):
        ## Model construction

        # Create environment
        name, args, kwargs = unpack(_config['obj_func'])
        f = getattr(environments_module, name)(**kwargs)

        # Create the model
        models = [_config['model']]
        if 'model2' in _config:
            models.append(_config['model2'])

        models = [create_model(model['name'], model['kwargs'], input_dim=f.input_dim)
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
            Algorithm = getattr(algorithm_module, _config['bo']['name'])
            bo_kwargs = _config['bo']['kwargs']
            bo = Algorithm(f, models, acq, **bo_kwargs)

            # Updates _run.result
            bo.run(callback=plot)
        else:
            bo = None

            # TODO: Throw exceptions as warning so interactive mode can still play with the objects.
            # Updates _run.result
            # try:
            test_gp_model(f, models, 
                acquisition_function=acq, 
                n_samples=_config['gp_samples'],
                use_derivatives=_config['gp_use_derivatives'])
            # except:
            #     print("ups")

        #Hack to have model available after run in interactive mode.
        _run.interactive_stash = {
            'f': f,
            'model': models[0],
            'model2': models[0] if len(models) >= 2 else None,
            'acq': acq,
            'bo': bo,
        }
        mse = _run.result
        return mse

    @ex.command
    def test(_config):
        print(json.dumps(_config))

    return ex


from src.encoder import PythonDictSyntax


def config_dict_to_cli(conf, cmd_name=None):
    config_list =  ["{}={}".format(k, json.dumps(v, cls=PythonDictSyntax)) for k,v in conf.items()]
    cmd = ["python", "runner.py", "with"] + config_list
    if cmd_name is not None:
        cmd.insert(2, cmd_name)
    return cmd


def notebook_to_CLI(*args, config_updates=None, **kwargs):
    """Run as shell script.
    
    Keyword Arguments:
        config_updates {[type]} -- [description] (default: {None})
    """
    assert len(args) <= 1, "Currently only support `command name` as args."
    assert not kwargs, "Currently only supports `config_updates` changes."

    cmd_name = args[0] if args else None


    if config_updates is None:
        config_updates = {}

    cmd = config_dict_to_cli(config_updates, cmd_name=cmd_name)
    return cmd
    #print(subprocess.check_output(cmd))


def hpc_wrap(cmd):
    """Takes a python script and wraps it in `sbatch` over `ssh`.
    
    Arguments:
        cmd {[string]} -- The python script to be executed.
    
    Returns:
        [string] -- Return array that can be executed with `subprocess.call`.
    """
    python_cmd_args = " ".join(map(lambda x: "'{}'".format(x), cmd))
    server_cmd = "cd mthesis; sbatch hpc.sh {}".format(python_cmd_args)
    ssh_cmd = ["ssh", "simba", server_cmd]
    return ssh_cmd


def notebook_run_server(*args, **kwargs):
    # TODO: test if changes have been made to src
    cmd = notebook_to_CLI(*args, **kwargs)
    ssh_cmd = hpc_wrap(cmd) 
    print(ssh_cmd)
    subprocess.call(ssh_cmd)


def notebook_run_CLI(*args, **kwargs):
    cmd = notebook_to_CLI(*args, **kwargs)
    print(cmd)
    subprocess.call(cmd)


def notebook_run(*args, **kwargs):
    """Run experiment from a notebook/IPython env.
    
    Returns:
        Experiment -- Includes _run.interactive_stash to access constructured models.
    """
    assert not kwargs.get('options'), "Currently options are not supported since we override them."

    ex = create_ex(interactive=True)
    kwargs = dict(options = {'--force': True}, **kwargs)
    return ex.run(*args, **kwargs)


def execute(*args, **kwargs):
    if settings.MODE == settings.MODES.SERVER:
        func = notebook_run_server
    elif settings.MODE == settings.MODES.LOCAL_CLI:
        func = notebook_run_CLI
    else:
        func = notebook_run

    return func(*args, **kwargs)


if __name__ == '__main__':
    ex = create_ex(interactive=False)
    ex.run_commandline(sys.argv + ["--force"])
