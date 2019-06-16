#%%
import scipy.io
import pandas as pd

# mat = scipy.io.loadmat('data/RF_FF3_Char_Inter.mat')
# data = mat.get('returns')
# df = pd.DataFrame(data)

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
