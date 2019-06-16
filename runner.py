import sys
import subprocess
import time

from src.environments.core import BaseEnvironment
from src.environments.helpers import EnvironmentNormalizer
from src.models.core_models import BaseModel, GPModel
from src.models import NormalizerModel, TransformerModel, DKLGPModel
from src.models.lls_gp import LocalLengthScaleGPModel
from src.plot_utils import plot_function, plot_model, plot_model_unknown_bounds

# For some reason it breaks without TkAgg when running from CLI.
# from src import settings
# if settings.MODE in [settings.MODES.SERVER, settings.MODES.LOCAL_CLI]:
#     import matplotlib
#     matplotlib.use("TkAgg")

import hashlib
import json

import numpy as np
from sacred import Experiment
from sacred.observers import MongoObserver

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['figure.dpi'] = 300 # migh high-res friendlly

from src.algorithms import AcquisitionAlgorithm
from src.utils import calc_errors, calc_errors_model_compare_mean, calc_errors_model_compare_var, random_hypercube_samples
from src.experiment import settings


def hash_subdict(d, keys=None):
    """Create unique based on subdict of a dict.
    If keys is None full dict is used.
    """
    if keys is None:
        keys = d.keys()

    d = {k: d.get(k, {}) for k in keys}
    return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()


def create_ex(interactive=False):
    ex = Experiment(settings.EXP_NAME, interactive=interactive)

    if settings.SAVE:
        ex.observers.append(MongoObserver.create(url=settings.MONGO_DB_URL, db_name=settings.MONGO_DB_NAME))

    ex.add_config({
        'tag': 'default',
        'gp_use_derivatives': False,
        'model_compare': False, # Only works for GP model (not BO)
        'verbosity': {
            'plot': settings.MODE is not settings.MODES.SERVER, # do not plot on server by default.
            'bo_show_iter': 30,
        }
    })


    @ex.capture
    def dklgpmodel_training_callback(model, i, loss, _log, _run):
        # TODO: save model
        if i % 30 == 0:
            # Log
            _log.info('Iter %d/%d - Loss: %.3f' % (i + 1, model.n_iter, loss))
        
        if i % 5 == 0:  
            # Metrics
            _run.log_scalar('DKLGPModel.training.loss', loss, i)

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
        exp_hash_key = ['mode', 'obj_func', 'model', 'model2', 'acquisition_function', 'bo', 'gp_samples']
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
    def update_result(name, value, _run):
        # Update the result dict with the latest value (notice `step` is ignored).
        result = _run.result
        if type(result) is not dict:
            result = {}
        result[name] = value
        _run.result = result

    @ex.capture
    def log_scalar(name, value, step, _run):
        _run.log_scalar(name, value, step)

        # Mongodb does not allow `.` in the key for a regular entry.
        name = name.replace(".", ":")
        update_result(name, value)

    @ex.capture
    def plot(algorithm: AcquisitionAlgorithm, i, _run, _log):
        # Log
        _log.info("... starting BO round {} / {}".format(i, algorithm.n_iter))

        # Metrics
        rmse, max_err = calc_errors(algorithm.models[0], algorithm.f, rand=True)
        log_scalar('rmse', rmse, i)
        log_scalar('max_err', max_err, i)

        # TODO: save weights

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
    def test_gp_model(f: BaseEnvironment, models: [BaseModel], _run, acquisition_function=None, n_samples=15, use_derivatives=False, model_compare=False):
        bounds = f.bounds
        input_dim = f.input_dim

        if input_dim == 1:
            X = np.random.uniform(bounds[0, 0], bounds[0, 1], (n_samples, 1))
        else:
            X = random_hypercube_samples(n_samples, bounds)

        Y = f(X)

        training_time = np.empty(len(models))

        if use_derivatives:
            Y_dir = f.derivative(X)
            for model in models:
                model.init(X, Y, Y_dir=Y_dir)
        else:
            for i, model in enumerate(models):
                start_time = time.clock()
                model.init(X, Y)
                training_time[i] = time.clock() - start_time

        if model_compare:
            # TODO: For now only supports 2 models
            model1 = models[0]
            model2 = models[1]
            rmse, max_err = calc_errors_model_compare_mean(model1, model2, f)
            log_scalar('model_compare.mean.rmse', rmse, 0)
            log_scalar('model_compare.mean.max_err', max_err, 0)

            rmse, max_err = calc_errors_model_compare_var(model1, model2, f)
            log_scalar('model_compare.var.rmse', rmse, 0)
            log_scalar('model_compare.var.max_err', max_err, 0)


        for i, model in enumerate(models):
            fig = plot_model(model, f)

            if fig is not None:
                save_fig(fig, settings.ARTIFACT_GP_FILENAME.format(model_idx=i))

            # Transformed model
            # TODO: Plot sampled
            if isinstance(model, NormalizerModel):
                true_model = model.model
                normalized_f = EnvironmentNormalizer(f, model.X_normalizer, model.Y_normalizer)
                
                if f.input_dim == 1:
                    fig = plot_model_unknown_bounds(true_model)
                    save_fig(fig, settings.ARTIFACT_UNNORMALIZED_FILENAME.format(model_idx=i))
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
                        save_fig(fig, settings.ARTIFACT_AS_FEATURES_FILENAME.format(model_idx=i))

            # Acquisition
            if acquisition_function is not None:
                fig = plot_function(normalized_f, acquisition_function, title="Acquisition functions", points=X)
                save_fig(fig, settings.ARTIFACT_GP_ACQ_FILENAME.format(model_idx=i))

            # Length scale
            if isinstance(true_model, LocalLengthScaleGPModel):
                fig = plot_function(normalized_f, lambda x: 1 / true_model.get_lengthscale(x)[:,None], title="1/Lengthscale", points=X)
                save_fig(fig, settings.ARTIFACT_LLS_GP_LENGTHSCALE_FILENAME.format(model_idx=i))

            # Plot feature space for DKLGP
            elif isinstance(true_model, DKLGPModel):
                fig = true_model.plot_features(normalized_f)
                if fig is not None:
                    save_fig(fig, settings.ARTIFACT_DKLGP_FEATURES_FILENAME.format(model_idx=i))

            # Log
            start_time = time.clock()
            rmse, max_err = calc_errors(model, f, rand=True)
            pred_time = time.clock() - start_time
            
            log_info('Model{}: {} has RMSE={} max_err={}'.format(i, model, rmse, max_err))

            if isinstance(true_model, DKLGPModel) or isinstance(true_model, GPModel):
                log_info('Model{} has parameters: {}'.format(i, true_model.get_common_hyperparameters()))

            if hasattr(true_model, 'warnings') and len(true_model.warnings) > 0:
              update_result('WARNING', true_model.warnings)

            # Only store under `model{i}` for additional models to share interface with BO metrics.
            if i == 0:
                log_scalar('rmse', rmse, 0)
                log_scalar('max_err', max_err, 0)
                log_scalar('time.training', training_time[i], 0)
                log_scalar('time.pred', pred_time, 0)
            else:
                log_scalar('model{}.rmse'.format(i), rmse, 0)
                log_scalar('model{}.max_err'.format(i), max_err, 0)
                log_scalar('model{}.time.training'.format(i), training_time[i], 0)
                log_scalar('model{}.time.pred'.format(i), pred_time, 0)
        

    @ex.main
    def main(_config, _run, _log):
        def recursively_apply_to_dict(d, modifier=lambda k, v: None):
            def iterate(d_):
                for k, v in d_.items():
                    if isinstance(v, dict):
                        iterate(v)
                    modifier(k, v)

            iterate(d)

        # Add logging
        def modifer(k, v):
            if isinstance(v, dict) and v.get('name') == 'DKLGPModel':
                v.get('kwargs')['training_callback'] = dklgpmodel_training_callback
        recursively_apply_to_dict(_config, modifer)

        ## Model construction
        from src.experiment.context import ExperimentContext
        context = ExperimentContext.from_config(_config)

        f = context.obj_func
        bo = context.bo
        models = context.models
        
        if bo is not None:
            bo.run(callback=plot)
        else:
            # TODO: Throw exceptions as warning so interactive mode can still play with the objects.
            # Updates _run.result
            # try:
            test_gp_model(f, models, 
                acquisition_function=context.acquisition_function,
                n_samples=context.gp_samples,
                use_derivatives=context.gp_use_derivatives)
            # except:
            #     print("ups")

        #Hack to have model available after run in interactive mode.
        _run.interactive_stash = {
            'f': f,
            'model': models[0],
            'model2': models[1] if len(models) >= 2 else None,
            'acq': context.acquisition_function,
            'bo': bo,
        }
        mse = _run.result
        return mse

    @ex.command
    def test(_config):
        print(json.dumps(_config))

    return ex


from src.experiment.encoder import PythonDictSyntax


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
