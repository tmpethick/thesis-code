# #%%
#=========================================================
# We are here to gain confidence in our models. Welcome
#=========================================================

%load_ext autoreload
%autoreload 2

from runner import notebook_run, notebook_run_CLI, notebook_run_server, execute

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")

from src.utils import *
from src.plot_utils import *
from src.kernels import *
from src.models.models import *
from src.models.dkl_gp import *
from src.models.lls_gp import *
from src.environments import *
from src.acquisition_functions import *
from src.algorithms import *

#%% Testing RFF and SKI behaviour

class SinExp(BaseEnvironment):
    bounds = np.array([-10, 10])

    def __call__(self, X):
        return np.sin(X) * np.exp(- X ** 2 / (2*5**2))

# TODO: implmenent RFF log likelihood (see )

#%% Stress test on 1D (running time test)
# https://arxiv.org/pdf/1511.01870.pdf

# inducing points on [-12,13]

# Test training runtime
# ExactGP (n=10^5 -> 10^2s)
# MSGP 
# for small n=10^2: m influence run time alot (always slower than ExactGP)
# for big n=10^7: m almost no influence. Runtime similar to ExactGP with n=10^4.

# Prediction time plot as function of n:
# Constant for all

# Prediction time plot as function of m:
# Should be constant for KISS
# Linear for SSGP?

#%% SSGP vs SKI (2D) (accuracy test)
# Compare n=[20, 100, 1000]. m=[10, 100, 1000] (plot m/SMAE)
# Compute SMAE


#%% DKL for discontinouity (p. 15)
# https://arxiv.org/pdf/1511.02222.pdf

#%%
