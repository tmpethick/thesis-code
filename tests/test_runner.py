


def test_grid_sampling():
    from notebook_header import *
    kissmodel = {
        'name': 'NormalizerModel',
        'kwargs': {
            'model': {
                'name': 'DKLGPModel',
                'kwargs': {
                    'learning_rate': 0.1,
                    'n_iter': 30,
                    'nn_kwargs': {'layers': None},
    #                'gp_kwargs': {'n_grid': M},
                    'use_cg': True,
                    'noise': None
                }
            }
        }
    }

    #models = [gp, kissmodel]
    run = execute(config_updates={
        'tag': 'heston',
        'obj_func': {
            'name': 'HestonOptionPricer',
        },
        'model': kissmodel,
        'gp_samples': 5,
        'use_sample_grid': True,
        'gp_test_samples': 2,
    })
    plt.scatter(run.interactive_stash.model.X[...,0], run.interactive_stash.model.X[...,1])
