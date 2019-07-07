import sys
import subprocess

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
from sacred import SETTINGS
SETTINGS.CONFIG.READ_ONLY_CONFIG = False

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

    @ex.capture
    def dklgpmodel_training_callback(model, i, loss, _log, _run):
        # TODO: save model
        if i % 10 == 0:
            # Log
            _log.info('Iter %d/%d - Loss: %.3f' % (i + 1, model.n_iter, loss))

        if i % 5 == 0:
            # Metrics
            _run.log_scalar('DKLGPModel.training.loss', loss, i)

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

        return {'model_hash': model_hash, 'exp_hash': exp_hash}

    @ex.main
    def main(_config, _run, _log):
        # Add logging
        def modifer(k, v):
            if isinstance(v, dict) and v.get('name') == 'DKLGPModel':
                v.get('kwargs')['training_callback'] = dklgpmodel_training_callback
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
        server_cmd = "sed 's/\"$@\"/{}/g' < hpc-dtu.sh | bsub".format(python_cmd_args)
        print(server_cmd)
        ssh_cmd = ["ssh", "s144448@login2.hpc.dtu.dk", server_cmd]
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
    r = subprocess.call(ssh_cmd)
    print(r)


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
