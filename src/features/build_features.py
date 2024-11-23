import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction


# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------
df = pd.read_pickle('../../data/interim/02_outliers_removed_chauvenets.pkl')
predictor_columns = list(df.columns[:6])

plt.style.use("fivethirtyeight")
plt.rcParams['figure.figsize'] = (20,5)
plt.rcParams['figure.dpi']=100
plt.rcParams["lines.linewidth"] =2

# --------------------------------------------------------------
# Dealing with missing values (imputation)
# --------------------------------------------------------------
df.info()
subset = df[df['set']==35]['gyr_y'].plot()


for col in predictor_columns:
    df[col] = df[col].interpolate()

df.info()

# --------------------------------------------------------------
# Calculating set duration
# --------------------------------------------------------------
df[df['set'] == 25]["acc_y"].plot()
df[df['set'] == 50 ]["acc_y"].plot()
duration = df[df['set']==1].index[-1] - df[df['set']==1].index[0]

duration.seconds

for s in df['set'].unique():
    start = df[df['set']==s].index[0]
    stop = df[df['set']==s].index[-1]
    duration = stop - start
    df.loc[(df['set'] == s),"duration"] = duration.seconds
    
df.groupby(['category'])["duration"].mean()
    
# --------------------------------------------------------------
# Butterworth lowpass filter
# --------------------------------------------------------------
df_lowpass = df.copy()
LowPass = LowPassFilter()

fs = 1000/200
cutoff = 1

df_lowpass = LowPass.low_pass_filter(df_lowpass,"acc_y",fs,cutoff,order = 5)

for col in predictor_columns:
    df_lowpass = LowPass.low_pass_filter(df_lowpass,col,fs,cutoff,order = 5)
    df_lowpass = LowPass.low_pass_filter(df_lowpass,col,fs,cutoff,order = 5)
    del df_lowpass[col + "_lowpass"]

# --------------------------------------------------------------
# Principal component analysis PCA
# --------------------------------------------------------------
df_pca = df_lowpass.copy()
PCA = PrincipalComponentAnalysis()

pc_values = PCA.determine_pc_explained_variance(df_pca,predictor_columns)
df_pca = PCA.apply_pca(df_pca,predictor_columns,3)
# --------------------------------------------------------------
# Sum of squares attributes
# --------------------------------------------------------------
df_squared = df_pca.copy()
acc_r = df_squared['acc_x']**2 + df_squared['acc_y']**2 + df_squared['acc_z']**2
gyr_r = df_squared['gyr_x']**2 + df_squared['gyr_y']**2 + df_squared['gyr_z']**2

df_squared['acc_r'] = np.sqrt(acc_r)
df_squared['acc_r'] = np.sqrt(gyr_r)

# --------------------------------------------------------------
# Temporal abstraction
# --------------------------------------------------------------


# --------------------------------------------------------------
# Frequency features
# --------------------------------------------------------------


# --------------------------------------------------------------
# Dealing with overlapping windows
# --------------------------------------------------------------


# --------------------------------------------------------------
# Clustering
# --------------------------------------------------------------


# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------