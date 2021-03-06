


def test_pickadability():
    """Tests that nested model with PyTorch (feature extractor) is pickadable.
    """
    from src.models import NormalizerModel

    import numpy as np
    import tempfile
    import pickle as pickle
    gp = NormalizerModel.from_config({
        'model': {
            'name': 'DKLGPModel',
            'kwargs': {
                'learning_rate': 0.1,
                'n_iter': 1,
                'nn_kwargs': {'layers': [10,1]},
                #'gp_kwargs': {'n_grid': 1000},
                'use_cg': True,
                'noise': None
            }
        }
    })
    with tempfile.NamedTemporaryFile() as fd:
        pickle.dump(gp, fd, protocol=pickle.HIGHEST_PROTOCOL)
        fd.seek(0)
        gp.init(np.random.uniform(0,1, (10,1)), np.zeros((10,1)))
        gp = pickle.load(fd)

def test_DKL_pickadability():
    from notebook_header import *
    import numpy as np
    from src.models import DKLGPModel
    import pickle
    import tempfile

    gp = DKLGPModel()

    f = Sinc()
    X = random_hypercube_samples(20, f.bounds)
    Y = f(X)

    gp.init(X, Y)
    plot_model(gp, f)
    gp.save('model_test')

    gp = DKLGPModel.load('model_test')
    plot_model(gp, f)
    # => Plots should be similar

def test_normalization_pickadability():
    from notebook_header import *

    path = os.path.join(settings.GROWTH_MODEL_SNAPSHOTS_DIR, 'test')

    model = NormalizerModel.from_config({
        'model': {
            'name': 'DKLGPModel',
            'kwargs': dict(
                verbose=False,
                n_iter=1,
                nn_kwargs=dict(layers=None),
                use_cg=True,
                max_cg_iter=30000,
                precond_size=20,
                use_double_precision=True,
                noise_lower_bound=1e-10,
                train_eval_cg_tolerance=1e-4,
            )
        }
    })
    model.init(np.random.uniform(size=(100,4)), np.random.uniform(size=(100,1)))
    model.save(path)
    model = SaveMixin.load(path)
    # TODO: cleanup files

def test_transformer_pickadability():
    from notebook_header import *

    path = os.path.join(settings.GROWTH_MODEL_SNAPSHOTS_DIR, 'test')

    model = TransformerModel.from_config({
        'transformer': {
            'name': 'ActiveSubspace',
            'kwargs': {
                'output_dim': 1
            }
        },
        'prob_model': {
            'name': 'DKLGPModel',
            'kwargs': dict(
                verbose=False,
                n_iter=100,
                nn_kwargs=dict(layers=None),
                use_cg=True,
                max_cg_iter=30000,
                precond_size=20,
                use_double_precision=True,
                noise_lower_bound=1e-10,
                train_eval_cg_tolerance=1e-4,
            )
        }
    })
    model.init(np.zeros((100,4)), np.zeros((100,1)), Y_dir=np.zeros((100,4)))
    model.save(path)
    model = SaveMixin.load(path)

    # TODO: cleanup files
