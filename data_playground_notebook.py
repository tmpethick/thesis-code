#%%
import scipy.io
import pandas as pd

mat = scipy.io.loadmat('data/RF_FF3_Char_Inter.mat')
data = mat.get('returns')

# (Time, RF + 3F + 69Char + 220PR)

#%% 

df = pd.DataFrame(data)

#%%
