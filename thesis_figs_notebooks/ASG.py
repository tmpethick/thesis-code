#%%
%load_ext autoreload
%autoreload 2

from runner import notebook_run, notebook_run_CLI, notebook_run_server, execute

import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import *
from src.plot_utils import *
from src.kernels import *
from src.models.models import *
from src.models.dkl_gp import *
from src.models.lls_gp import *
from src.environments import *
from src.acquisition_functions import *
from src.algorithms import *

from src.plot_utils import latexify, savefig
latexify(columns=1)

def calc_error(i, model):
    Y_hat = model.evaluate(X_test)
    max_error = np.max(np.fabs(Y_hat - Y_test))
    print("{0:9d} {1:9d}  {2:1.2e}".format(i+1, model.grid.getNumPoints(), max_error))


#%%

from src.models.asg import AdaptiveSparseGrid
from src.utils import random_hypercube_samples

# f = Sinc2D() # Behaves weirdly with nothing plotted for non-adaptive, and adaptive cannot capture the rotated embedding.

f = Kink2D()
X_test = random_hypercube_samples(1000, f.bounds)
Y_test = f(X_test)

# Without Adaptive
asg = AdaptiveSparseGrid(f, depth=5, refinement_level=0)
asg.fit(callback=calc_error)
fig = asg.plot()

#%%

# Adaptive
asg = AdaptiveSparseGrid(f, depth=1, refinement_level=5, f_tol=1e-2)
asg.fit(callback=calc_error)
fig = asg.plot()
savefig(fig, 'ASG/Kink2D.pgf')

#%%
