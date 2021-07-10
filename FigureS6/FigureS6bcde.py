import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common.plotting_functions import *
from common.distance_metrics import *

from scipy.stats import pearsonr

    
# %%

fname = 'data/noise_df_no6s.csv'

df = pd.read_csv(fname)
df = df.set_index(df.cell_specimen_ids.values.astype('int'))
df.index.name = 'unit_id'

df = df.groupby('unit_id').mean()

# %%

df_ophys = pd.read_csv('data/df_ophys_200620.csv', index_col=0, low_memory=False)

df = df.join(df_ophys, on='unit_id', how='inner', rsuffix='_orig')

# %%
metrics = ['sig_fraction_spont_', 'lifetime_sparseness_']
stim = 'dg'

plt.figure(num=1010, figsize=(10,5))
plt.clf()

A_max = 0.25
sigma_max = 0.125

matplotlib.rcParams.update({'font.size': 14})

plot_axes = False

for metric_idx, metric in enumerate(metrics):
    
    for quantile in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        
        selection = df['firing_rate_' + stim] > 0.0
        
        sub_df = df[selection]
        
        selection = sub_df[metric + stim] > np.quantile(sub_df[metric + stim], quantile)
        
        plt.subplot(4,3,1 + 6*metric_idx)
        h,b = np.histogram(sub_df[selection].sigmaest, bins=100)
        plt.semilogy(b[:-1],h, color='purple', alpha=(1-quantile)*0.9+0.1)
        plt.xlim([0.01,sigma_max])
        
        plt.subplot(4,3,2+ 6*metric_idx)
        h,b = np.histogram(sub_df[selection].tauest, bins=100)
        plt.semilogy(b[:-1],h, color='orange', alpha=(1-quantile)*0.9+0.1)

        selection &= sub_df.aest > 0
        plt.subplot(4,3,3+ 6*metric_idx)
        h,b = np.histogram(np.log(sub_df[selection].aest), bins=100)
        plt.semilogy(b[:-1],h, color='red', alpha=(1-quantile)*0.9+0.1)
    
    selection = df['firing_rate_' + stim] > 0.0
    
    M_orig = df[selection][metric + stim]
    M = np.copy(M_orig)
    
    if metric_idx == 0:
        M += np.random.random(len(M))*0.065
        M[M > 1] = np.nan
    
    plt.subplot(4,3,4+ 6*metric_idx)
    plt.scatter(df[selection].sigmaest, M, s=1 ,c='purple',alpha=0.1)
    plt.ylim([0,1])
    plt.ylabel(metric + stim)
    r, p = pearsonr(df[selection].sigmaest, M_orig)
    plt.title(np.around(r,3))
    plt.xlim([0.01,sigma_max])
    
    plt.subplot(4,3,5+ 6*metric_idx)
    plt.scatter(df[selection].tauest, M, s=1 ,c='orange',alpha=0.1)
    plt.ylim([0,1.01])
    r, p = pearsonr(df[selection].tauest, M_orig)
    plt.title(np.around(r,3))

    
    plt.subplot(4,3,6+ 6*metric_idx)
    plt.scatter(np.log(df[selection].aest), M, s=1 ,c='red',alpha=0.1)
    plt.ylim([0,1.01])
    r, p = pearsonr(np.log(df[selection].aest), M_orig)
    plt.title(np.around(r,3))

plt.subplots_adjust(wspace=0.5, hspace=0.4, left=0.1, right=0.9, bottom=0.1, top=0.9)
plt.tight_layout()

