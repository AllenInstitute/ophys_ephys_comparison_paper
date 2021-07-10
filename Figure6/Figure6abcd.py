import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common.plotting_functions import *
from common.distance_metrics import *

# %%

df_ophys = pd.read_csv('data/df_ophys_200620.csv', index_col=0, low_memory=False)

df_ephys = pd.read_csv('data/df_ephys_200620.csv', index_col=0, low_memory=False)

df_ephys_synth_ml = pd.read_csv('data/df_synth_ophys.csv', index_col=0)

df_ephys = df_ephys.join(df_ephys_synth_ml, on='unit_id', rsuffix='_ophys_ml', how='inner')

# %%

from scipy.ndimage.filters import gaussian_filter1d

plt.figure(99)
plt.clf()

stims = ['dg']
metrics = ['lifetime_sparseness_', 'sig_fraction_spont_']

qc_metric = 'firing_rate_dg'

areas = ['VISp', 'VISl', 'VISal', 'VISpm', 'VISam']

quantiles = np.arange(0.0, 0.95, 0.01)
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
                
                selection = (df_ophys['firing_rate_' + stim] > 0.0)
                
                threshold = np.nanquantile(df_ophys[selection][qc_metric], q)
                thresh.append(threshold)

                selection = (df_ophys[qc_metric] >= threshold) & \
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
            plt.plot(data2, color=get_color_palette(area, name='seaborn'))
            
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

M = df_ophys[df_ophys.ecephys_structure_acronym.isin(areas)].firing_rate_dg
h,b = np.histogram(M+0.00001, np.logspace(-1.5,0.8,200))
h_smooth = gaussian_filter1d(h,2)
logB = np.log10(b)
plt.plot(logB[:-1],h_smooth, color='green')
plt.bar(logB[48:-1],h_smooth[48:], width=np.mean(np.diff(logB)), color='green', alpha=0.6)

for t in thresh[::10]:
    i = np.searchsorted(b,t)
    plt.plot([logB[i-1],logB[i-1]],[0,h_smooth[i-1]],'-k', alpha=0.2)
    
ax = plt.gca()
[ax.spines[loc].set_visible(False) for loc in ['right', 'top']]  

plt.xticks(ticks=[-1,0],
           labels=[0.1, 1])
plt.xlabel('Event rate (Hz)')
plt.ylabel('Cell count')

plt.subplot(3,2,6)

M = df_ophys[df_ophys.ecephys_structure_acronym.isin(areas)].firing_rate_dg
h,b = np.histogram(M+0.00001, np.logspace(-1.5,0.8,200))
h_smooth = gaussian_filter1d(h,2)
logB = np.log10(b)
plt.plot(logB[:-1],h_smooth, color='green')
plt.bar(logB[70:-1],h_smooth[70:], width=np.mean(np.diff(logB)), color='green', alpha=0.6)

for t in thresh[::10]:
    i = np.searchsorted(b,t)
    plt.plot([logB[i-1],logB[i-1]],[0,h_smooth[i-1]],'-k', alpha=0.2)

ax = plt.gca()
[ax.spines[loc].set_visible(False) for loc in ['right', 'top']]  

plt.xticks(ticks=[-1,0],
           labels=[0.1, 1])
plt.xlabel('Event rate (Hz)')
plt.ylabel('Cell count')

                        # %%
            
plt.figure(99)
plt.clf()

stims = ['dg']
metrics = ['lifetime_sparseness_', 'sig_fraction_spont_']

qc_metric = 'isi_violations'

areas = ['VISp', 'VISl', 'VISal', 'VISpm', 'VISam']

quantiles = np.arange(0.95,0.0,-0.01)
bins = np.arange(0,1.1,0.1)
    
for metric_idx, metric in enumerate(metrics):
    
    for stim in stims:
        
        for area_idx, area in enumerate(areas):
        
            data1 = []
            data2 = []
            thresh = []
            
            selection = (df_ophys['ecephys_structure_acronym'] == area)
                         
            if metric != 'sig_fraction_spont_':
                selection &= (df_ophys['sig_fraction_spont_' + stim] > 0.25)
            else:
                selection &= (df_ophys['firing_rate_' + stim] > 0.0)
                         
            ophys_values = df_ophys[selection][metric + stim] #, bins, 'k', plot=False)
            
            for q in quantiles:
                
                threshold = np.quantile(df_ephys[qc_metric], q)
                thresh.append(threshold)
                
                #print(threshold)
                selection = (df_ephys[qc_metric] <= threshold) & \
                                  (df_ephys['ecephys_structure_acronym'] == area)
                                  
                if metric != 'sig_fraction_spont_':
                    selection &= (df_ephys['sig_fraction_spont_' + stim] > 0.25)
                else:
                    selection &= (df_ephys['firing_rate_' + stim] > 0.0)
                
                ephys_values = df_ephys[selection][metric+stim + '_ophys_ml'] #, bins,'k', plot=False)
                
                S, W, J = compare_distributions(ophys_values, ephys_values, bins)
                data1.append(J)
                
                if metric == 'sig_fraction_spont_':
                    data2.append(np.mean(ephys_values > 0.25))
                else:
                    data2.append(np.median(ephys_values))
                
            plt.subplot(3,2,metric_idx+1)
            plt.plot(data2, color=get_color_palette(area, name='seaborn')) #, alpha=(area_idx+2)/7)
            
            plt.subplot(3,2,metric_idx+3)
            plt.plot(data1, color=get_color_palette(area, name='seaborn'))
            
    plt.subplot(3,2,metric_idx+1)
    plt.ylim([0.4,0.8])
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
    plt.ylim([0,0.3])
    ax = plt.gca()
    [ax.spines[loc].set_visible(False) for loc in ['right', 'top']]  
    
    plt.xlabel('Percent of cells included')
    plt.xticks(ticks=np.arange(0,101,10),
               labels=np.arange(100,-1,-10))
    plt.ylabel('J-S Distance')



plt.subplot(3,2,5)


M = df_ephys[df_ephys.ecephys_structure_acronym.isin(areas)].isi_violations
h,b = np.histogram(M+0.00001, np.logspace(-5,-0.3012,100))
h_smooth = gaussian_filter1d(h[1:],2)
logB = np.log10(b)
plt.plot(logB[1:-1],h_smooth, color='gray')
plt.bar(logB[0],h[0],width=0.1, color='orange', alpha=0.8)
plt.bar(logB[1:-9],h_smooth[:-8],width=np.mean(np.diff(logB)), color='orange', alpha=0.8)

for t in thresh[0:90:10]:
    i = np.searchsorted(b,t)
    plt.plot([logB[i-2],logB[i-2]],[0,h_smooth[i-3]],'-k', alpha=0.2)
    
ax = plt.gca()
[ax.spines[loc].set_visible(False) for loc in ['right', 'top']]  

plt.xticks(ticks=[-5,-4,-3,-2,-1],
           labels=[0, 0.0001, 0.01, 0.01, 0.1])
plt.xlabel('ISI violations')
plt.ylabel('Unit count')

plt.subplot(3,2,6)

plt.plot(logB[1:-1],h_smooth, color='gray')
plt.bar(logB[0],h[0],width=0.1, color='orange', alpha=0.8)
plt.bar(logB[1:-34],h_smooth[:-33],width=np.mean(np.diff(logB)), color='orange', alpha=0.8)

for t in thresh[0:90:10]:
    i = np.searchsorted(b,t)
    plt.plot([logB[i-2],logB[i-2]],[0,h_smooth[i-3]],'-k', alpha=0.2)
    
ax = plt.gca()
[ax.spines[loc].set_visible(False) for loc in ['right', 'top']]  

plt.xticks(ticks=[-5,-4,-3,-2,-1],
           labels=[0, 0.0001, 0.01, 0.01, 0.1])
plt.xlabel('ISI violations')
plt.ylabel('Unit count')
