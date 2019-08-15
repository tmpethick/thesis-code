import matplotlib
matplotlib.use("TKAgg")

from notebook_header import *
import pyipopt
import os

# Ensure all nodes gets killed if one fails (otherwise the job will continue).
import sys
# def kill_mpi(exctype, value, tb):
#     from mpi4py import MPI
#     print(exctype, value, tb)
#     MPI.COMM_WORLD.Abort()
# sys.excepthook = kill_mpi

# Prevent logging
pyipopt.set_loglevel(0)

if 'LSB_JOBID' in os.environ:
    path = os.path.join(settings.GROWTH_MODEL_SNAPSHOTS_DIR, os.environ['LSB_JOBID'])
    GrowthModel = GrowthModelDistributed
else:
    path = os.path.join(settings.GROWTH_MODEL_SNAPSHOTS_DIR, 'test')
    GrowthModel = GrowthModel

# Save this script so we know what model configuration are associated with the output.
from shutil import copyfile
pathlib.Path(path).mkdir(parents=True, exist_ok=True)
copyfile(os.path.realpath(__file__), os.path.join(path, 'script.py'))

# model = TransformerModel.from_config({
#     'transformer': {
#         'name': 'ActiveSubspace',
#         'kwargs': {
#             'output_dim': 1
#         }
#     },
#     'prob_model': {
#         'name': 'DKLGPModel',
#         'kwargs': dict(
#             verbose=False,
#             n_iter=2000,
#             learning_rate=0.01,
#             nn_kwargs=dict(layers=None),
#             use_cg=True,
#             max_cg_iter=30000,
#             precond_size=20,
#             use_double_precision=True,
#             noise_lower_bound=1e-10,
#             train_eval_cg_tolerance=1e-4,
#         )
#     }
# })


model = SGPR.from_config({
    'do_pretrain': True,
    'pretrain_n_iter': 3000,
    'grad_clip': 2,
    'learning_rate': 0.01,
    'n_iter': 3000,
    'nn_kwargs': {'layers': [1]},
    'gp_kwargs': {'inducing_points': 100},
    'max_cg_iter': 30000,
    'precond_size': 20,
    'use_cg': True,
    'noise': None,
    'use_double_precision': True,
    'noise_lower_bound': 1e-8,
    'train_eval_cg_tolerance': 1e-4,
    'eval_cg_tolerance': 1e-6,
    'verbose': False,
})

# model = DKLGPModel(
#     verbose=False,
#     n_iter=2000,
#     learning_rate=0.01,
#     nn_kwargs=dict(layers=[1]),
#     use_cg=True,
#     max_cg_iter=30000,
#     precond_size=20,
#     use_double_precision=True,
#     noise_lower_bound=1e-10,
#     train_eval_cg_tolerance=1e-4,
# )
model = NormalizerModel(model=model)

gm = GrowthModel(
    output_dir=path,
    n_agents=50,
    beta=0.96,#0.8,
    zeta=0.5,
    psi=0.36,
    gamma=2.0,
    delta=0.06,#0.025
    eta=1,
    k_bar=0.2,
    k_up=3.0,
    c_bar=1e-2,
    c_up=10.0,
    l_bar=1e-2,
    l_up=10.0,
    inv_bar=1e-2,
    inv_up=10.0,
    numstart=1,
    numits=100,
    No_samples=1000,
    No_samples_postprocess=20
)

#callback = GrowthModelCallback(gm, verbose=True)
def plot(i, growth_model, model):
    if isinstance(model, NormalizerModel):
        true_model = model.model
        X = random_hypercube_samples(100, growth_model.bounds)
        X = model.X_normalizer.normalize(X)
        X_train = model.X_normalizer.normalize(model.X)
        Y_train = model.Y_normalizer.normalize(model.Y)
    else:
        X = random_hypercube_samples(100, growth_model.bounds)
        true_model = model
        X_train = model.X
        Y_train = model.Y

    if hasattr(true_model, 'feature_extractor') and true_model.feature_extractor is not None and true_model.feature_extractor.output_dim == 1:
        Z = true_model.get_features(X)
        Z_train = true_model.get_features(X_train)
        Y, _ = true_model.get_statistics(X, full_cov=False)
        path = os.path.join(growth_model.params.model_dir + str(i), 'plot.png')
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(Z, Y)
        ax.scatter(Z_train, Y_train)
        ax.set_title(f"Value iteration {i}")

        fig.savefig(path, transparent=False)


gm.loop(model, callback=plot)
