##################################################
# This notebook is used for debugging in PyCharm
##################################################
from runner import notebook_run

training_size_to_total_size = lambda x: int(x * 1/(0.8*0.8))

# Without feature extractor
# run = notebook_run(config_updates={
#     'obj_func': {
#         'name': 'SPXOptions',
#         'kwargs': {'D': 1, 'subset_size': training_size_to_total_size(10000)},
#     },
#     'model': {
#         'name': 'NormalizerModel',
#         'kwargs': {
#             'model': {
#                 'name': 'DKLGPModel',
#                 'kwargs': {
#                     'learning_rate': 0.1,
#                     'n_iter': 1,
#                     'nn_kwargs': {'layers': None},
#                     'gp_kwargs': {'n_grid': 1000},
#                     'noise': None
#                 }
#             }
#         }
#     },
# })

# Without normalizer
# run = notebook_run(config_updates={
#     'obj_func': {
#         'name': 'SPXOptions',
#         'kwargs': {'D': 1, 'subset_size': training_size_to_total_size(10000)},
#     },
#     'model': {
#         'name': 'DKLGPModel',
#         'kwargs': {
#             'learning_rate': 0.1,
#             'n_iter': 1,
#             'nn_kwargs': {'layers': None},
#             'gp_kwargs': {'n_grid': 1000},
#             'noise': None
#         }
#     },
# })

# Without sacred
import matplotlib.pyplot as plt
import numpy as np
from src.environments.financial import SPXOptions
from src.models import DKLGPModel

model = DKLGPModel.from_config({
    'learning_rate': 0.1,
    'n_iter': 100,
    'nn_kwargs': {'layers': None},
    'gp_kwargs': {'n_grid': 100000},
    'use_cg': True,
    'noise': None
})
data = SPXOptions(D=1, subset_size=100000)
model.init(data.X_train, data.Y_train)

# N = len(data.Y_test)
# Y_hat, var = model.get_statistics(data.X_test, full_cov=False)
# rmse = np.sqrt(np.sum(np.square(Y_hat - data.Y_test)) / N)
# err_max = np.max(np.fabs(Y_hat - data.Y_test))
