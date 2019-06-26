#%%
import scipy.io
import pandas as pd

# mat = scipy.io.loadmat('data/RF_FF3_Char_Inter.mat')
# data = mat.get('returns')
# df = pd.DataFrame(data)

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
