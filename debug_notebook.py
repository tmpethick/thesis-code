from runner import notebook_run

notebook_run(config_updates={
    'obj_func': {
        'name': 'Kink2D',
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
            'num_mcmc': 0,
            'subsample_interval': 10,
            'step_size': 1e-1,
            'leapfrog_steps': 20,
            'n_burnin': 100,
        }
    },
    'acquisition_function': {
        'name': 'CurvatureAcquisition',
    },
    'bo': {
        'name': 'SampleAlgorithm',
        'kwargs': {
            'n_init': 5,
            'n_iter': 20,
            'n_acq_max_starts': 10,
        }
    },
}, options={'--force': True})
