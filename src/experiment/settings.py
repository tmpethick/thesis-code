import os
from enum import Enum

from src.env import MONGO_DB_PASSWORD

# ---------------- Setting options
# ----------------

class MODES(Enum):
    SERVER = 1
    LOCAL_CLI = 2
    LOCAL = 3

# ----------------- Settings
# -----------------

MODE = MODES.SERVER
SAVE = MODE is MODES.SERVER

SERVER_DEST = 'epfl' # 'dtu' or 'epfl'
WALLTIME = '00:40'
QUEUE = 'gpuv100' #gputitanxpascal
EXTRA = '' # e.g. #BSUB -R "select[gpu32gb]"

if MONGO_DB_PASSWORD is not None:
    MONGO_DB_URL = 'mongodb+srv://admin:{}@lions-rbvzc.mongodb.net/test?retryWrites=true'.format(MONGO_DB_PASSWORD)
else:
    MONGO_DB_URL = 'mongodb+srv://lions-public:zkKHsvxud683gEQv@lions-rbvzc.mongodb.net/test?retryWrites=true'
MONGO_DB_NAME = 'test'

EXP_NAME = 'lions'

EXP_HASH = 'exp_hash'     # Unique for (model, obj_func) pair.
MODEL_HASH = 'model_hash' # Unique for each model.

ARTIFACT_BO_PLOT_FILENAME = 'artifacts/bo-plot-{i}.png'
ARTIFACT_INPUT_FILENAME = 'artifacts/X.npy'
ARTIFACT_OUTPUT_FILENAME = 'artifacts/Y.npy'

ARTIFACT_GP_FILENAME = 'artifacts/gp-{model_idx}.png'
ARTIFACT_GP_ACQ_FILENAME = 'artifacts/gp-acq-{model_idx}.png'
ARTIFACT_LLS_GP_LENGTHSCALE_FILENAME = 'artifacts/llsgp-lengthscale-{model_idx}.png'
ARTIFACT_DKLGP_FEATURES_FILENAME = 'artifacts/dklgp-features-{model_idx}.png'
ARTIFACT_AS_FEATURES_FILENAME = 'artifacts/AS-features-{model_idx}.png'

ARTIFACT_UNNORMALIZED_FILENAME = 'artifacts/unnormalized-model-{model_idx}.png'

SETTINGS_DIR = os.path.dirname(os.path.abspath(__file__))
THESIS_FIGS_DIR = os.path.join(SETTINGS_DIR, '../../thesis_figs/')
MODEL_SNAPSHOTS_DIR = os.path.join(SETTINGS_DIR, '../../output/models/')
GROWTH_MODEL_SNAPSHOTS_DIR = os.path.join(SETTINGS_DIR, '../../output/growth-model/')
