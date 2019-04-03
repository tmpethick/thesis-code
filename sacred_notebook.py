#%%
%load_ext autoreload
%autoreload 2

from runner import notebook_run

#%%

notebook_run(config_updates={
    'obj_func': {
        'name': 'Kink2D',
    },
    'model': {
        'name': 'DKLGPModel',
        'kwargs': {
            'n_iter': 50,
            'nn_kwargs': {
                'layers': (50, 25, 10, 2),
            }
        }
    },
    'gp_samples': 5,
}, options={'--force': True})


#%%
# ------------------ Templates --------------------

# %% Fully specified GP without acq
notebook_run(config_updates={
    'obj_func': {
        'name': 'Kink1D',
    },
    'model': {
        'name': 'GPModel',
        'kwargs': {
            'kernel': {
                'name': 'GPyRBF',
                'kwargs': {
                    'lengthscale': 1
                }
            },
            'noise_prior': None,
            'do_optimize': False,
            'num_mcmc': 10,
            'subsample_interval': 10,
            'step_size': 1e-1,
            'leapfrog_steps': 20,
            'n_burnin': 100,
        },
    },
    'gp_samples': 5,
}, options={'--force': True})


# %% Fully specified GP with acq
notebook_run(config_updates={
    'obj_func': {
        'name': 'Kink1D',
    },
    'model': {
        'name': 'GPModel',
        'kwargs': {
            'kernel': {
                'name': 'GPyRBF',
                'kwargs': {
                    'lengthscale': 1
                }
            },
            'noise_prior': None,
            'do_optimize': False,
            'num_mcmc': 10,
            'subsample_interval': 10,
            'step_size': 1e-1,
            'leapfrog_steps': 20,
            'n_burnin': 100,
        }
    },
    'acquisition_function': {
        'name': 'QuadratureAcquisition',
    },
    'gp_samples': 5,
}, options={'--force': True})

# %% Fully specified BO

notebook_run(config_updates={
    'obj_func': {
        'name': 'Kink1D',
    },
    'model': {
        'name': 'GPModel',
        'kwargs': {
            'kernel': {
                'name': 'GPyRBF',
                'kwargs': {
                    'lengthscale': 1
                }
            },
            'noise_prior': None,
            'do_optimize': False,
            'num_mcmc': 10,
            'subsample_interval': 10,
            'step_size': 1e-1,
            'leapfrog_steps': 20,
            'n_burnin': 100,
        }
    },
    'acquisition_function': {
        'name': 'QuadratureAcquisition',
    },
    'bo': {
        'n_init': 5,
        'n_iter': 10,
        'n_acq_max_starts': 10,
    },
}, options={'--force': True})

#%%

notebook_run(config_updates={
    'obj_func': {
        'name': 'Kink1D',
    },
    'model': {
        'name': 'LocalLengthScaleGPModel',
        'kwargs': {
            'l_samples': 5,
        }
    },
    'acquisition_function': {
        'name': 'QuadratureAcquisition',
    },
    'bo': {
        'n_init': 5,
        'n_iter': 10,
        'n_acq_max_starts': 10,
    },
}, options={'--force': True})


# %% Small modifications

notebook_run(config_updates={},
       named_configs=['configs/default.yaml', 'configs/gpy.yaml'],
       options={'--force': True})


# %%
# --------------- Utils ----------------

# %% Save configs
notebook_run('save_config', named_configs=['gpy'], config_updates={'config_filename': 'configs/gpy.yaml'},
       options={'--force': True})

# %% Print configs

notebook_run('print_config', options={'--force': True})
