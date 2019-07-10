#%%
import scipy.io
import pandas as pd

# mat = scipy.io.loadmat('data/RF_FF3_Char_Inter.mat')
# data = mat.get('returns')
# df = pd.DataFrame(data)

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
