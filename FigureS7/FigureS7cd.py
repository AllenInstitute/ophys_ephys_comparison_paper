import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from common.distance_metrics import *
from common.plotting_functions import *

# %%

df_ophys = pd.read_csv('data/df_ophys_with_cv.csv', index_col=0, low_memory=False)

df_ephys = pd.read_csv('data/df_ephys_with_forward_model.csv', index_col=0, low_memory=False)

df_ophys['cv_dg'] = df_ophys['std_pref_dg'] / df_ophys['mean_pref_dg']

# %%

from scipy.ndimage.filters import gaussian_filter1d

plt.figure(991)
plt.clf()

stims = ['dg']
metrics = ['lifetime_sparseness_', 'sig_fraction_spont_']

qc_metric = 'cv_dg' # vs. firing_rate_dg

areas = ['VISp', 'VISl', 'VISal', 'VISpm', 'VISam']

quantiles = np.arange(0.95,0.0,-0.01)
bins = np.arange(0,1.1,0.1)

min_loc = np.zeros((2,5))
    
for metric_idx, metric in enumerate(metrics):
    
    for stim in stims:
        
        for area_idx, area in enumerate(areas):
        
            data1 = []
            data2 = []
            thresh = []
            
            selection = (df_ephys['ecephys_structure_acronym'] == area)
                         
            if metric != 'sig_fraction_spont_':
                selection &= (df_ephys['sig_fraction_spont_' + stim + '_ophys_ml'] > 0.25)
            else:
                selection &= (df_ephys['firing_rate_' + stim + '_ophys_ml'] > 0.0)
                         
            ephys_values = df_ephys[selection][metric + stim + '_ophys_ml'] #, bins, 'k', plot=False)
            
            for q in quantiles:
                
                selection = (np.invert(np.isnan(df_ophys['fano_' + stim])))
                
                threshold = np.nanquantile(df_ophys[selection][qc_metric], q)
                thresh.append(threshold)
                selection = (df_ophys[qc_metric] <= threshold) & \
                                  (df_ophys['ecephys_structure_acronym'] == area)
                                  
                if metric != 'sig_fraction_spont_':
                    selection &= (df_ophys['sig_fraction_spont_' + stim] > 0.25)
                
                ophys_values = df_ophys[selection][metric+stim]
                S, W, J = compare_distributions(ophys_values, ephys_values, bins)
                
                data1.append(J)
                
                if metric == 'sig_fraction_spont_':
                    data2.append(np.mean(ophys_values > 0.25))
                else:
                    data2.append(np.median(ophys_values))
                    
            min_loc[metric_idx, area_idx] = quantiles[np.argmin(data1)]
                
            plt.subplot(3,2,metric_idx+1)
            plt.plot(data2, color=get_color_palette(area, name='seaborn')) #, alpha=(area_idx+2)/7)
            
            plt.subplot(3,2,metric_idx+3)
            plt.plot(data1, color=get_color_palette(area, name='seaborn'))
    
    plt.subplot(3,2,metric_idx+1)
    ax = plt.gca()
    [ax.spines[loc].set_visible(False) for loc in ['right', 'top']]  
    
    plt.xlabel('Percent of cells included')

    plt.xticks(ticks=np.arange(0,101,10),
               labels=np.arange(100,-1,-10))
    
    if metric == 'sig_fraction_spont_':
        plt.ylabel('Fraction responsive')
    else:
        plt.ylabel('Median lifetime sparseness')
      
    plt.subplot(3,2,metric_idx+3)
    plt.ylim([0,0.5])
    ax = plt.gca()
    [ax.spines[loc].set_visible(False) for loc in ['right', 'top']]  
    
    plt.xlabel('Percent of cells included')

    plt.xticks(ticks=np.arange(0,101,10),
               labels=np.arange(100,-1,-10))
    plt.ylabel('J-S Distance')

plt.subplot(3,2,5)

M = df_ophys[df_ophys.ecephys_structure_acronym.isin(areas)][qc_metric]
h,b = np.histogram(M+0.00001, np.linspace(0,4,200))
h_smooth = gaussian_filter1d(h,5)
logB = b
plt.plot(logB[:-1], h_smooth, color='green')
plt.bar(logB[:123],h_smooth[:123], width=np.mean(np.diff(logB)), color='green', alpha=0.6)

for q in np.arange(0.1,1.0,0.1):
    t = np.nanquantile(df_ophys[qc_metric], q)
    i = np.searchsorted(b,t)
    plt.plot([logB[i-1],logB[i-1]],[0,h_smooth[i-1]],'-k', alpha=0.2)
    
ax = plt.gca()
[ax.spines[loc].set_visible(False) for loc in ['right', 'top']]  

plt.xlabel('Coefficient of variation')
plt.ylabel('Cell count')

plt.subplot(3,2,6)

M = df_ophys[df_ophys.ecephys_structure_acronym.isin(areas)][qc_metric]
h,b = np.histogram(M+0.00001, np.linspace(0,4,200))
h_smooth = gaussian_filter1d(h,5)
logB = b
plt.plot(logB[:-1], h_smooth, color='green')
plt.bar(logB[:49],h_smooth[:49], width=np.mean(np.diff(logB)), color='green', alpha=0.6)

for q in np.arange(0.1,1.0,0.1):
    t = np.nanquantile(df_ophys[qc_metric], q)
    i = np.searchsorted(b,t)
    plt.plot([logB[i-1],logB[i-1]],[0,h_smooth[i-1]],'-k', alpha=0.2)
    
ax = plt.gca()
[ax.spines[loc].set_visible(False) for loc in ['right', 'top']]  

plt.xlabel('Coefficient of variation')
plt.ylabel('Cell count')
