#%%
%load_ext autoreload
%autoreload 2

#%%
import numpy as np
from src.plot_utils import plot2D, plot1D
from src.algorithms import random_hypercube_samples
from src.models.dkl_gp import DKLGPModel
from src.environments import Kink1D, Kink2D, BaseEnvironment
from src.models.models import BaseModel

import matplotlib.pyplot as plt
plt.interactive(True)

import seaborn as sns
sns.set_style("darkgrid")

def test_gp_model(f: BaseEnvironment, model: BaseModel, n_samples=15):
   bounds = f.bounds
   input_dim = f.input_dim

   if input_dim == 1:
      X = np.random.uniform(bounds[0, 0], bounds[0, 1], (n_samples, 1))
   else:
      X = random_hypercube_samples(n_samples ** input_dim, bounds)

   y = f(X)[:, 0]

   model.fit(X, y)

   if input_dim == 1:
      plot1D(model, f)
   elif input_dim == 2:
      plot2D(model, f)


#%% -------------------- 1D ---------------------

f = Kink1D()
model = DKLGPModel(gp_dim=2)
test_gp_model(f, model)

#%%

f = Kink2D()
model = DKLGPModel(gp_dim=2)
test_gp_model(f, model)


# Test as GP
# Test as BO

# Plot lengthscale change
# Compare with base-line (standard GP)
