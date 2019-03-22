#%%
%load_ext autoreload
%autoreload 2

# %%

import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import GPy

from src.algorithms import AcquisitionAlgorithm, random_hypercube_samples
from src.models import GPModel, RandomFourierFeaturesModel 
from src.acquisition_functions import QuadratureAcquisition

# Plotting
import seaborn as sns
sns.set_style("darkgrid")

#%% Linear kernel

k = GPy.kern.Linear(1) + GPy.kern.White(1, variance=10)
k.plot()
X = np.linspace(0.,1.,500)
X = X[:,None]
C = k.K(X,X)
plt.imshow(C, interpolation='nearest')
plt.show()
mu = np.zeros((500))
Z = np.random.multivariate_normal(mu,C,20)
for i in range(20):
      plt.plot(X[:],Z[i,:])

#%% Plotting the quadratic behavior of the "flipped" diagonal
A = np.zeros(C.shape[0])
for i in range(C.shape[0]):
   j = (C.shape[0] - 1) - i
   A [i] = C[i,j]
plt.plot(A)

#%% Plot any kernel
import numpy as np
import pylab as plt
import GPy
import re

def get_equation(kern):
    match = re.search(r'(math::)(\r\n|\r|\n)*(?P<equation>.*)(\r\n|\r|\n)*', kern.__doc__)
    return '' if match is None else match.group('equation').strip()


# Try plotting sample paths here
k = GPy.kern.LinearFull(input_dim=1, rank=1, kappa=np.array([1000]), W=1000*np.ones((1, 1)))

X = np.linspace(0.,1.,500) # define X to be 500 points evenly spaced over [0,1]
X = X[:,None] # reshape X to make it n*p --- we try to use 'design matrices' in GPy 

mu = np.zeros((500))# vector of the means --- we could use a mean function here, but here it is just zero.
C = k.K(X,X) # compute the covariance matrix associated with inputs X

# Generate 20 separate samples paths from a Gaussian with mean mu and covariance C
Z = np.random.multivariate_normal(mu,C,20)

            
kernel_equation = get_equation(k)
#print kernel_equation
from IPython.display import Math, display
display(Math(kernel_equation))

fig = plt.figure()     # open a new plotting window
plt.subplot(121)
for i in range(3):
      plt.plot(X[:],Z[i,:])

plt.title('{} samples'.format(kernel_name))

plt.subplot(122)
plt.imshow(C, interpolation='nearest')
plt.title('{} covariance'.format(kernel_name))

#%%
