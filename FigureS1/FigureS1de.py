import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from common.plotting_functions import *
from common.distance_metrics import *

# %%

df_ophys = pd.read_csv('data/df_ophys_200620.csv', index_col=0, low_memory=False)
df_ephys = pd.read_csv('data/df_ephys_200620.csv', index_col=0, low_memory=False)

# %%

ophys_color = 'green'
ephys_color = 'gray'

metric = 'g_dsi_'
stimuli = ['dg', 'sg', 'ns', 'nm1']

bins = np.arange(0,1.1,0.1)

JS = []

for stim_idx, stim in enumerate(stimuli[0:1]):
    plt.figure(2101+stim_idx, figsize=(19,4))
    plt.clf()

    axis, ophys_values = plot_selectivity_comparison(metric + stim, 
                                df_ophys[df_ophys['sig_fraction_spont_' + stim] > 0.25], 
                                ophys_color)
    axis, ephys_values = plot_selectivity_comparison(metric + stim, 
                                df_ephys[(df_ephys['sig_fraction_spont_' + stim] > 0.25)], 
                                ephys_color
                                )
    

    for area_idx, area in enumerate(areas):
    
        S, W, J = compare_distributions(ophys_values[area_idx],
                                      ephys_values[area_idx],
                                      bins)
        
        JS.append(J)

        plt.subplot(1,5,area_idx+1)
        plt.text(0.75, 1.2, 'D = ' + str(np.around(J,2)), fontsize=12)
        
plt.tight_layout()

print(np.mean(JS))

metric = 'g_osi_'
stimuli = ['dg', 'sg', 'ns', 'nm1']

JS = []

for stim_idx, stim in enumerate(stimuli[0:1]):
    plt.figure(2102+stim_idx, figsize=(19,4))
    plt.clf()

    axis, ophys_values = plot_selectivity_comparison(metric + stim, 
                                df_ophys[df_ophys['sig_fraction_spont_' + stim] > 0.25], 
                                ophys_color)
    axis, ephys_values = plot_selectivity_comparison(metric + stim, 
                                df_ephys[(df_ephys['sig_fraction_spont_' + stim] > 0.25)], 
                                ephys_color
                                )
    

    for area_idx, area in enumerate(areas):
    
        S, W, J = compare_distributions(ophys_values[area_idx],
                                      ephys_values[area_idx],
                                      bins)
        
        JS.append(J)

        plt.subplot(1,5,area_idx+1)
        plt.text(0.75, 1.2, 'D = ' + str(np.around(J,2)), fontsize=12)
        
plt.tight_layout()

print(np.mean(JS))


# %%