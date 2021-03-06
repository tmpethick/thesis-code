import sys
import subprocess

import os
import  numpy as np
from src.experiment.runner import Runner
from src.experiment.encoder import PythonDictSyntax

# For some reason it breaks without TkAgg when running from CLI.
# from src import settings
# if settings.MODE in [settings.MODES.SERVER, settings.MODES.LOCAL_CLI]:
#     import matplotlib
#     matplotlib.use("TkAgg")

import hashlib
import json

from sacred import Experiment
from sacred.observers import MongoObserver

import matplotlib

from src.experiment.config_helpers import recursively_apply_to_dict

matplotlib.rcParams['figure.dpi'] = 300 # migh high-res friendlly

from src.experiment import settings

def unpack(config_obj):
    return config_obj['name'], \
           config_obj.get('arg', ()), \
           config_obj.get('kwargs', {})

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

    class DKLGPModelTrainingCallback(object):
        def __init__(self, _log, _run, _config):
            self._log = _log 
            self._run = _run
            self._config = _config
    
            self.threshold_factor = 7
            self.prev_loss = None
            self.moving_average_loss_diff_total = 0
            self.moving_average_loss_diff_N = 1

            self.save_next_flag = False

        def save_model(self, model, i):
            model.save(os.path.join(settings.MODEL_SNAPSHOTS_DIR, self._config['exp_hash'], str(i)))

        def __call__(self, model, i, loss):

            total_norm = 0
            for p in model.model.parameters():
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            self._run.log_scalar('DKLGPModel.training.grad_l2', total_norm, i)

            # if settings.MODE == settings.MODES.LOCAL:
            #     if loss > 10.0:
            #         self.save_model(model, i)

            # if settings.MODE == settings.MODES.LOCAL:
            #     if i % 10 == 0:
            #         self.save_model(model, i)

            # if self.prev_loss:
            #     loss_diff = np.abs(self.prev_loss - loss)
            # else:
            #     loss_diff = 0
            # self.prev_loss = loss

            # if self.save_next_flag:
            #     self.save_model(model, i)
            #     self.save_next_flag = False
            # elif i > 10 and loss_diff > self.threshold_factor * (self.moving_average_loss_diff_total / self.moving_average_loss_diff_N):
            #     self.save_model(model, i)
            #     self.save_next_flag = True

            # self.moving_average_loss_diff_total = self.moving_average_loss_diff_total + loss_diff
            # self.moving_average_loss_diff_N = self.moving_average_loss_diff_N + 1

            # print(self.prev_loss, loss, loss_diff, self.moving_average_loss_diff_total / self.moving_average_loss_diff_N)

            # TODO: save model
            if i % 10 == 0:
                # Log
                self._log.info('Iter %d/%d - Loss: %.3f' % (i + 1, model.n_iter, loss))

            if i % 5 == 0:
                # Metrics
                self._run.log_scalar('DKLGPModel.training.loss', loss, i)


    @ex.config_hook
    def add_unique_hashes(config, command_name, logger):
        """Create the unique hash and modify config (before main is called)
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

        return {
            'model_hash': model_hash, 
            'exp_hash': exp_hash,
            'gpu': settings.SERVER_DEST == 'dtu'
        }

    @ex.main
    def main(_config, _run, _log):
        # Add logging
        def modifer(k, v):
            if isinstance(v, dict) and v.get('name') in ['DKLGPModel', 'LinearFromFeatureExtractor', 'DNNBLR', 'SGPR', 'SSGP']:
                v.get('kwargs')['training_callback'] = DKLGPModelTrainingCallback(_log, _run, _config)
        recursively_apply_to_dict(_config, modifer)

        ## Context/model setup
        from src.experiment.context import ExperimentContext
        context = ExperimentContext.from_config(_config)

        # Run experiment
        runner = Runner(context, _log, _run)
        runner.run()

        #Hack to have model available after run in interactive mode.
        # TODO: just return context
        _run.interactive_stash = context
        mse = _run.result
        return mse

    @ex.command
    def test(_config):
        print(json.dumps(_config))

    return ex


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
    if settings.SERVER_DEST == 'dtu':
        # We have to be very careful with the use of single and double quotes here...
        python_cmd_args = " ".join(map(lambda x: "'{}'".format(x), cmd))
        python_cmd_args = python_cmd_args.replace("'", "\\x27")
        server_cmd = " ".join([
            "source /etc/profile;",
            "cd mthesis;",
            f"sed 's/CMD/{python_cmd_args}/g' < hpc-dtu.sh.template",
            f"| sed 's/QUEUE/{settings.QUEUE}/g'", 
            f"| sed 's/WALLTIME/{settings.WALLTIME}/g'", 
            f"| sed 's/EXTRA/{settings.EXTRA}/g'", 
            " | bsub",
        ])
        print(server_cmd)
        # Very helpful list of what is loaded in interactive-mode compared to non-interactive: https://schaazzz.github.io/linux-evironment-files-scripts/
        ssh_cmd = ['ssh' ,'s144448@login2.hpc.dtu.dk', server_cmd]
    else:
        python_cmd_args = " ".join(map(lambda x: "'{}'".format(x), cmd))
        server_cmd = "cd mthesis; sbatch hpc.sh {}".format(python_cmd_args)
        ssh_cmd = ["ssh", "simba", server_cmd]
    return ssh_cmd


def notebook_run_server(*args, **kwargs):
    # TODO: test if changes have been made to src
    cmd = notebook_to_CLI(*args, **kwargs)
    ssh_cmd = hpc_wrap(cmd)
    print(ssh_cmd)
    #subprocess.call(ssh_cmd)
    print(subprocess.check_output(ssh_cmd))


# DEPRECRATED since they don't automatically adjust settings.SAVE
# Only access through `execute`.
def notebook_run_CLI(*args, **kwargs):
    cmd = notebook_to_CLI(*args, **kwargs)
    print(cmd)
    subprocess.call(cmd)


def notebook_run(*args, **kwargs):
    """Run experiment from a notebook/IPython env.
    
    Returns:
        Experiment -- Includes _run.interactive_stash to access constructed models.
    """
    assert not kwargs.get('options'), "Currently options are not supported since we override them."

    ex = create_ex(interactive=True)
    kwargs = dict(options = {'--force': True}, **kwargs)
    return ex.run(*args, **kwargs)


def execute(*args, **kwargs):
    """It will run the local / local through CLI / server depending on settings.MODE.
    """
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
