from notebook_header import *
import pyipopt
import os


# Prevent logging
pyipopt.set_loglevel(0)

path = os.path.join(settings.GROWTH_MODEL_SNAPSHOTS_DIR, os.environ['LSB_JOBID'])

# Save this script so we know what model configuration are associated with the output.
from shutil import copyfile
pathlib.Path(path).mkdir(parents=True, exist_ok=True)
copyfile(os.path.realpath(__file__), os.path.join(path, 'script.py'))

# Get GP to converge
# A-SG
# ASGP

model = DKLGPModel(
    verbose=False,
    n_iter=2000,
    nn_kwargs=dict(layers=None),
    use_cg=True,
    max_cg_iter=30000,
    precond_size=20,
    use_double_precision=True,
    noise_lower_bound=1e-10,
    train_eval_cg_tolerance=1e-4,
)

gm = GrowthModelDistributed(
#gm = GrowthModel(
    output_dir=path,
    n_agents=10,
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
    numits=100,
    No_samples_postprocess=20
)
#callback = GrowthModelCallback(gm, verbose=True)
def plot(i, growth_model, model):
    if model.feature_extractor is not None and model.feature_extractor.output_dim == 1:
        X = random_hypercube_samples(1000, growth_model.bounds)
        Z = model.get_features(X)
        Y, _ = model.get_statistics(X, full_cov=False)
        path = os.path.join(growth_model.params.model_dir + str(i), 'plot.png')
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(Z, Y)
        ax.set_title(f"Value iteration {i}")

        fig.savefig(path, transparent=False)


gm.loop(model, callback=plot)
