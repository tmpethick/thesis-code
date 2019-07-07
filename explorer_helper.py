import sys
sys.path.append('.')

import math
import pandas as pd
import numpy as np
import incense
from incense import ExperimentLoader
import matplotlib.pyplot as plt
from src.experiment import settings

# Print helpers
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import Terminal256Formatter
from pprint import pformat

from notebook_header import *

def save_table(str_, label):
    print(str_)
    with open('thesis_tables/{}.tex'.format(label),'w') as tf:
        tf.write(str_)

def pprint_color(obj):
    print(highlight(pformat(obj), PythonLexer(), Terminal256Formatter()))
    
import addict
from src.utils import _calc_errors, random_hypercube_samples

def get_name(exp_row):
    # Create short hand name for convinience
    name = exp_row['model']
    if exp_row['bo']: 
        name = name + " BO"
    if exp_row['acq'] is not None: 
        name = name + " " + exp_row['acq']
    name = name + " " + exp_row['f']
    return name

def get_model_name(model_config):
    if model_config.name == 'TransformerModel':
        kwargs = model_config.kwargs
        name = "T<{},{}>".format(
            get_model_name(model_config.kwargs['transformer']), 
            get_model_name(model_config.kwargs['prob_model']))
    elif model_config.name == 'NormalizerModel':
        name = "N<{}>".format(get_model_name(model_config.kwargs.model))
    else:
        name = model_config.get('name', None)
    return name


def prefix_dict(dict_, prefix):
    return {"{}{}".format(prefix, k): v for k,v in dict_.items()}

def get_exp_key_col(exp):
    """Convert Experiment to pandas row columns.
    """
    config = exp.to_dict()['config']
    config = addict.Dict(config)

    if isinstance(config.obj_func, str):
        print(exp)
    
    # TODO: Remove custom hack to make f unique when taking parameter D.
    fD = str(config.obj_func.kwargs.get('D', ''))
    
    name = get_model_name(config.model)
    name2 = get_model_name(config.model2)
    
    # TODO: Hack and does not support when DataSet's default size is used.
    # DataSet
    N = config.obj_func.kwargs.get('subset_size', None)
    if N is not None:
        N = config.obj_func.kwargs.subset_size
    # Samples
    elif config.gp_samples:
        N = config.gp_samples
    # BO
    else:
        # TODO: BO
        pass

    exp_row = {
        settings.MODEL_HASH: config[settings.MODEL_HASH],
        settings.EXP_HASH: config[settings.EXP_HASH],
        'model': name,
        'model2': name2,
        'acq': config.get('acquisition_function', {}).get('name'),
        'bo': bool(config.get('bo', None)),
        'f': str(config['obj_func']['name']) + fD,
        'N': N,
        'config': config,
        'tag': config['tag'],
        'exp': exp,
        'id': exp.id,
        'status': exp.status,
    }
    # Create short hand name for convinience
    exp_row['name'] = get_name(exp_row)
    
    # Unpack the results as columns
    if hasattr(exp, 'result') and exp.result is not None:
        exp_row.update(prefix_dict(exp.result, 'result.'))
    
    return exp_row


def get_bo_plots(exp):
    return {k: v for k,v in exp.artifacts.items() if k.startswith('bo-plot')}

# ------------- Add entries -----------------

    
def create_baseline(df):
    from src import environments as environments_module
    from runner import unpack, hash_subdict

    functions = df.drop_duplicates(subset='f').apply(lambda r: [r.f, r.config.obj_func, r.config], axis=1)

    baseline_df = pd.DataFrame()
    
    for f_name, func, config in functions:
        name, args, kwargs = unpack(func)
        f = getattr(environments_module, name)(**kwargs)

        if isinstance(f, DataSet):
            X_train = f.X_train
            Y_train = f.Y_train
            X_test = f.X_test
            Y_test = f.Y_test
        elif isinstance(f, BaseEnvironment):
            # Training samples 
            n_samples = config.gp_samples
            X_train = random_hypercube_samples(n_samples, f.bounds)
            Y_train = f(X_train)
            X_test = random_hypercube_samples(2500, f.bounds)
            Y_test = f(X_test)
        else: 
            return None

        Y_est = np.mean(Y_train, axis=0)
        mean_estimator = lambda X: np.repeat(Y_est[None,:], X.shape[0], axis=0)
            
        mae, rmse, max_err = errors(mean_estimator(X_test), Y_test)

        mean_name = 'mean'
        mean_exp_hash = hash_subdict({'model': mean_name, 'f': func}, keys=['model', 'f'])
        
        baseline_df = baseline_df.append([{
            'exp_hash': mean_exp_hash,
            'model_hash': mean_name,
            'model': mean_name, 
            'config': config,
            'f': f_name,
            'result.rmse': rmse,
            'result.max_err': max_err,
        }])
    
    baseline_df = baseline_df.set_index('exp_hash').sort_index()
    return baseline_df

def create_SG_df(df, depth=3, refinement_level=10, f_tol=1e-3):
    """Runs SG and A-SG for every unique function in `df`.
    Assumes df indexed by exp_hash
    """
    # Performance of SG and A-SG
    functions = df.drop_duplicates(subset='f').apply(lambda r: [r.f, r.config.obj_func, r.config], axis=1)

    # Add SG and A-SG to all f
    from src import environments as environments_module
    from runner import unpack, hash_subdict
    from src.models.ASG import AdaptiveSparseGrid

    # Remove multiindex for easy appending
    SG_df = pd.DataFrame()

    for f_name, func, config in functions:
        name, args, kwargs = unpack(func)
        f = getattr(environments_module, name)(**kwargs)

        print("Fitting SG")
        sg = AdaptiveSparseGrid(depth=depth, refinement_level=0)
        sg.fit(f)

        print("Fitting A-SG")
        asg = AdaptiveSparseGrid(depth=1, refinement_level=refinement_level, f_tol=f_tol, point_tol=1000)
        asg.fit(f)

        sg_rmse, sg_max_err = _calc_errors(sg.evaluate, f, f, rand=True)
        asg_rmse, asg_max_err = _calc_errors(asg.evaluate, f, f, rand=True)

        # Hack to create unique exp_hash (unique pr. model,f pair)
        sg_exp_hash = hash_subdict({'model': 'SG', 'f': func}, keys=['model', 'f'])
        asg_exp_hash = hash_subdict({'model': 'A-SG', 'f': func}, keys=['model', 'f'])
        
        SG_df = SG_df.append([{
            'exp_hash': sg_exp_hash,
            'model_hash': 'SG',  # just have to be unique for the model.
            'model': 'SG', 
            'config': config,
            'f': f_name, 
            'result.rmse': sg_rmse,
            'result.max_err': sg_max_err,
            'N': sg.grid.getNumPoints(), 
            'depth': sg.total_depth
        }])
        SG_df = SG_df.append([{
            'exp_hash': asg_exp_hash,
            'model_hash': 'A-SG',
            'model': 'A-SG',
            'config': config,
            'f': f_name, 
            'result.rmse': asg_rmse, 
            'result.max_err': asg_max_err,
            'N': asg.grid.getNumPoints(), 
            'depth': asg.total_depth
        }])

    #SG_df = SG_df.set_index(['model', 'f']).sort_index()
    SG_df = SG_df.set_index('exp_hash').sort_index()
    return SG_df

# -------------- Aggregate ------------------

def aggregate_results(df, describe=False):
    """Aggregate all results (i.e. final value of metrics)."""
    agg = dict.fromkeys(df, 'first')
    result_keys = [k for k in agg.keys() if k.startswith('result.') and k not in ['result.WARNING', 'result.hyperparameters']]
    if describe:
        for k in result_keys:
            agg[k] = 'describe'
    else:
        for k in result_keys:
            agg[k] = lambda x: np.nanmean(x)

    df = df.drop('Ntemp', axis=1, errors='ignore')    
    df['Ntemp'] = df['N'].fillna(-1).astype(int)
    return df.reset_index().groupby(['exp_hash', 'Ntemp']).agg(agg)

# ------------------ View -------------------

def view_df(df, indexes=['model_hash'], cols=['result.rmse'], f_as_col=False):
    df = df.reset_index().set_index(indexes + ['f']).sort_index()
    df = df[cols]
    if f_as_col:
        return df.unstack('f')
    else:
        return df

def select_experiment_with_rmse(df, rmse, atol=1e-6):
    _ = df[np.isclose(df["result.rmse"], rmse, atol=atol)]
    exp = _.iloc[0].exp

    pprint_color(exp.config)
    for name, artifact in exp.artifacts.items():
        artifact.show()

    loss = exp.metrics.get('DKLGPModel.training.loss')
    if loss is not None:
        loss.plot()

    return exp

import datetime as dt

# Load
loader = ExperimentLoader(
    mongo_uri=settings.MONGO_DB_URL,
    db_name=settings.MONGO_DB_NAME
)

def get_df(status='COMPLETED'):
    query = {
        'start_time': {
            '$gte': dt.datetime.strptime('2019-05-14T15:24:39.914Z', "%Y-%m-%dT%H:%M:%S.%fZ")}}
            #'$lt': dt.datetime.strptime('2019-05-14T15:24:39.914Z', "%Y-%m-%dT%H:%M:%S.%fZ")}}
    if status is not None:
        query['status'] = status
    exps = loader.find(query)

    #exps = loader.find({'status': 'COMPLETED'})
    df = pd.DataFrame([get_exp_key_col(exp) for exp in exps])
    df = df.set_index('exp_hash').sort_index()
    return df


