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
from src.models.asg import *
from src.environments import *
from src.acquisition_functions import *
from src.algorithms import *

latexify(columns=1)


f = Kink2D()
X_test = random_hypercube_samples(1000, f.bounds)
# Uses uniform since bounds are [0,1] to ensure implementation is not broken...
X_test = np.random.uniform(size=(1000,2))
N_test = X_test.shape[-1]
Y_test = f(X_test)

def calc_error(i, model):
    max_error, L2_err = model.calc_error(X_test, Y_test)
    print("{0:9d} {1:9d}  Loo={2:1.2e}  L2={3:1.2e}".format(i+1, model.grid.getNumPoints(), max_error, L2_err))


#%%

# f = Sinc2D() # Behaves weirdly with nothing plotted for non-adaptive, and adaptive cannot capture the rotated embedding.

# Without Adaptive
asg = AdaptiveSparseGrid(f, depth=15, refinement_level=0)
asg.fit(callback=calc_error)
fig = asg.plot()

# I should be able to get good performance with non-adaptive SG with 3e5 points.


#%%

# Extremely sensitive to f_tol (0.0099 works, >=0.01 breaks)

# Adaptive
asg = AdaptiveSparseGrid(f, depth=1, refinement_level=30, f_tol=0.0099)
asg.fit(callback=calc_error)
fig = asg.plot()
#savefig(fig, 'ASG/Kink2D.pgf')

#%%

# L2 and Loo as function of #points

refinement_levels = 30
f_tol = 0.0099

def test_depth_to_error(ASG_creator, max_points=4e5):
    N = np.empty(refinement_levels)
    Loo_err = np.empty(refinement_levels)
    L2_err = np.empty(refinement_levels)

    for i in range(refinement_levels):
        ASG = ASG_creator(i)
        ASG.fit()

        N[i] = ASG.grid.getNumPoints()
        Loo_err[i], L2_err[i] = ASG.calc_error(X_test, Y_test)
        
        if N[i] > max_points:
            break

    return N[:i], Loo_err[:i], L2_err[:i]

N, Loo_err, L2_err = test_depth_to_error(lambda i: AdaptiveSparseGrid(f, depth=i, refinement_level=0, f_tol=f_tol))
plt.plot(N, Loo_err, label="$L_\infty$ error - SG", marker='*', c='black')
plt.plot(N, L2_err, label="$L_2$ error - SG", marker=11, c='black')

N, Loo_err, L2_err = test_depth_to_error(lambda i: AdaptiveSparseGrid(f, depth=1, refinement_level=i, f_tol=f_tol))
plt.plot(N, Loo_err, label="$L_\infty$ error - ASG", marker='*', dashes=[2,2], c='black')
plt.plot(N, L2_err, label="$L_2$ error - ASG", marker=11,  dashes=[2,2], c='black')

plt.xlabel('\#Points N')
plt.ylabel('Error')
plt.yscale('log')
plt.xscale('log')
plt.legend()
savefig(fig, 'ASG/depth_to_error.pgf')


#%%
# Error as function of Threshold 

thresholds = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]

def test_depth_to_error(ASG_creator, max_points=4e5):
    Loo_err = np.empty(len(thresholds))
    L2_err = np.empty(len(thresholds))

    for i, threshold in enumerate(thresholds):
        ASG = ASG_creator(threshold)
        print("Threshold={}   N={}".format(threshold, ASG.grid.getNumPoints()))
        ASG.fit()

        Loo_err[i], L2_err[i] = ASG.calc_error(X_test, Y_test)

    return Loo_err[:i], L2_err[:i]


Loo_err, L2_err = test_depth_to_error(lambda threshold: AdaptiveSparseGrid(f, depth=15, refinement_level=0, f_tol=threshold))
plt.plot(thresholds, Loo_err, label="$L_\infty$ error - SG", marker='*')
plt.plot(thresholds, L2_err, label="$L_2$ error - SG", marker=11)

Loo_err, L2_err = test_depth_to_error(lambda threshold: AdaptiveSparseGrid(f, depth=30, refinement_level=1, f_tol=threshold))
plt.plot(thresholds, Loo_err, label="$L_\infty$ error - ASG", marker='*')
plt.plot(thresholds, L2_err, label="$L_2$ error - ASG", marker=11)

plt.xlabel('Thresholds')
plt.ylabel('Error')
plt.yscale('log')
plt.legend()
savefig(fig, 'ASG/threshold_to_error.pgf')

#%% Adaptivity breaks down completely for "circular" embeddings

f2 = Kink2D()

for i in [0, -0.5, -0.8]:
    f2.bounds = np.array([[-1,1], [i,1]])
    asg = AdaptiveSparseGrid(f2, depth=1, refinement_level=10, f_tol=1e-2)
    asg.fit(callback=calc_error)
    fig = asg.plot()
    plt.show()
