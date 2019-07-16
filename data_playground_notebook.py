#%%
import scipy.io
import pandas as pd

# mat = scipy.io.loadmat('data/RF_FF3_Char_Inter.mat')
# data = mat.get('returns')
# df = pd.DataFrame(data)

#%%

gm = GrowthModel(
    n_agents=50,
    beta=0.8,
    zeta=0.5,
    psi=0.36,
    gamma=2.0,
    delta=0.025,
    eta=1,
    k_bar=0.2,
    k_up=3.0,
    c_bar=1e-2,
    c_up=10.0,
    l_bar=1e-2,
    l_up=10.0,
    inv_bar=1e-2,
    inv_up=10.0,
    numits=66, 
    No_samples_postprocess=20
)
gm.post.ls_error()

#%%
# Lets find why this model does not save well 

model = DKLGPModel(
                verbose=False,
                n_iter=500,
                use_cg=False,
                use_double_precision=True,
                noise_lower_bound=1e-10,
            )
X = np.array([[1.20662212, 1.42368947],
 [2.15336735, 0.36863132],
 [2.0669468,  2.07778603],
 [0.78907117, 0.56099363],
 [1.08319938, 1.21839016],
 [1.79655096, 1.42808424],
 [2.96744675, 0.48572547],
 [0.78485492, 0.65166665],
 [2.02870331, 0.90921649],
 [1.50567016, 0.88439166],
 [0.64511483, 0.5090504 ],
 [2.03772285, 0.58691226],
 [0.75043061, 1.23243048],
 [2.49878104, 0.47188357],
 [2.54624574, 0.46907554],
 [2.9340865,  1.51222336],
 [2.93493105, 1.89356746],
 [2.26993802, 0.30972582],
 [0.9918595,  0.53655037],
 [1.02919255, 0.53243761]])
Y = np.array([[-873.88452805],
 [-874.3631474 ],
 [-877.71616787],
 [-844.93347347],
 [-869.57054253],
 [-889.64624869],
 [-868.65538094],
 [-847.37696186],
 [-889.8643134 ],
 [-880.62077453],
 [-835.73839718],
 [-881.95936714],
 [-853.95022259],
 [-877.04924812],
 [-876.40538399],
 [-882.7807875 ],
 [-875.09347409],
 [-871.81949209],
 [-853.65530298],
 [-855.12340945]])

model.init(X, Y)
model.save("model_save")
loaded_model = DKLGPModel.load("model_save")

#%%



#%%
from notebook_header import *

model = DKLGPModel(
    verbose=False,
    n_iter=500,
    use_cg=False,
    use_double_precision=True,
    noise_lower_bound=1e-10,
)

# kernel = GPyRBF
# model = GPModel(kernel=kernel, do_optimize=True)

# from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern
# kernel = RBF()
# model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=1)

gm = GrowthModelDistributed(
#gm = GrowthModel(
    n_agents=2,
    beta=0.8,
    zeta=0.5,
    psi=0.36,
    gamma=2.0,
    delta=0.025,
    eta=1,
    k_bar=0.2,
    k_up=3.0,
    c_bar=1e-2,
    c_up=10.0,
    l_bar=1e-2,
    l_up=10.0,
    inv_bar=1e-2,
    inv_up=10.0,
    numits=20, 
    No_samples_postprocess=20
)
#callback = GrowthModelCallback(gm, verbose=True)
gm.loop(model)


#%%
from src.environments.financial import AAPL

f = AAPL(D=2)
X = f.X_test
Y = f.Y_test[:,0]


#%%

import numpy as np

#d = np.stack((f.X_test, f.Y_test))
X = (np.array([1.3, 0.0, 1.3])
Y = np.array([4, 10, 20])
d = np.stack(indexes, Y)).T
df = pd.DataFrame(d)
df.groupby([0]).agg({1: 'mean'})



#%%

# economic_policies
df = pd.read_csv('data/economic_policies/Output.plt', header=None, delim_whitespace=True)
X = df.loc[:, 0:1].values
policies = df.loc[:, 3::2].values

#%%

D = 20
df = pd.read_csv(f'data/economic_policies/GPR_training-{D}d.txt', header=None, delim_whitespace=True)
X = df.loc[:,3:3+D-1]
policies = df.loc[:,3+D+1:]
policies.shape

#%%

mat_hyp = scipy.io.loadmat('data/scalable/precipitation/precipitation3240-hyp.mat')
mat = scipy.io.loadmat('data/scalable/precipitation/precipitation3240.mat')
X_hyp = mat_hyp['Xhyp']
Y_hyp = mat_hyp['yhyp']
X_train = mat['X']
Y_train = mat['y']
X_test = mat['Xtest']
Y_test = mat['ytest']

#%%

DATASETS = [
    '3droad',
    'airfoil',
    'autompg',
    'autos',
    'bike',
    'breastcancer',
    'buzz',
    'challenger',
    'concrete',
    'concreteslump',
    'elevators',
    'energy',
    'fertility',
    'forest',
    'gas',
    'houseelectric',
    'housing',
    'keggdirected',
    'keggundirected',
    'kin40k',
    'machine',
    'parkinsons',
    'pendulum',
    'pol',
    'protein',
    'pumadyn32nm',
    'servo',
    'skillcraft',
    'slice',
    'sml',
    'solar',
    'song',
    'stock',
    'tamielectric',
    'wine',
    'yacht',
]


#%% UCI

import scipy.io
mat = scipy.io.loadmat('data/uci/{0}/{0}.mat'.format('3droad'))
data = mat
data['data'].shape
# data['cvo']
data

#%%

import scipy.io
mat = scipy.io.loadmat('data/OptionData_0619/optionsSPXweekly_96_17.mat')
data = mat['optionsSPX']

#%%

COLS = [
    "strike",
    "midquote",
    "tau",
    "r",
    "q",
    "bid",
    "ask",
    "IV",
    "volume",
    "OpenInterest",
    "Delta",
    "Gamma",
    "Vega",
    "Theta",
    "LastTradeDate",
    "Callput",
    "Date",
    "S"]

df = pd.DataFrame(data=dict(zip(COLS, data.T)))

order_X = [
'strike',
'tau',
'S',
'r',
'q',
'volume',
'OpenInterest',
'LastTradeDate',
'ask',
'bid',
]
df
#%%

df[order_X[:3]]


#%%
df.shape
