import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common.plotting_functions import *
from common.distance_metrics import *

# %%

df_ophys = pd.read_csv('data/df_ophys_200620.csv', index_col=0, low_memory=False)
df_ephys = pd.read_csv('data/df_ephys_200620.csv', index_col=0, low_memory=False)
                           
# %%
    
import matplotlib

matplotlib.rcParams.update({'font.size': 14})

ophys_color = 'green'
ephys_color = 'gray'

metric = 'sig_fraction_spont_'
stimuli = ['dg', 'sg', 'ns', 'nm3']
stim_names = ['Drifting Gratings',
              'Static Gratings',
              'Natural Scenes',
              'Natural Movies']

plt.figure(2000, figsize=(18,4))
plt.clf()

for stim_idx, stim in enumerate(stimuli):
    plt.subplot(1,4,stim_idx+1)

    axis, ophys_values = plot_responsiveness_comparison(metric + stim, 
                                                        df_ophys[df_ophys['firing_rate_' + stim] > 0], ophys_color, offset=-0.18)
    axis, ephys_values = plot_responsiveness_comparison(metric + stim, 
                        df_ephys[(df_ephys['firing_rate_' + stim] > 0)], ephys_color, offset=+0.18)
    
    plt.title(stim_names[stim_idx])
    
    for area_idx, area in enumerate(areas):
        
        bins = np.arange(0,1.1,0.1)
        S, W, J = compare_distributions(ophys_values[area_idx],
                                      ephys_values[area_idx],
                                      bins)

        plt.text(area_idx-0.2, np.mean(ephys_values[area_idx] > 0.25) + 0.05, str(np.around(J,2)), fontsize=12)
    
plt.tight_layout()
    
# %%

ophys_color = 'green'
ephys_color = 'gray'

metric = 'pref_tf_'
stim = 'dg'

plt.figure(1999, figsize=(15,4))
plt.clf()

axis, ophys_values = plot_preference_comparison(metric + stim, 
                           df_ophys[(df_ophys['sig_fraction_spont_' + stim] > 0.25) *
                                    (df_ophys[metric + 'multi_' + stim] == False)],
                           ophys_color, 
                           offset=-0.18)
axis, ephys_values = plot_preference_comparison(metric + stim, 
                           df_ephys[(df_ephys['sig_fraction_spont_' + stim] > 0.25) *
                                    (df_ephys[metric + 'multi_' + stim] == False)],
                           ephys_color, 
                           offset=+0.18)
    

bins = np.arange(0,5.1,1.0)

for area_idx, area in enumerate(areas):
    

    S, W, J = compare_distributions(ophys_values[area_idx],
                                      ephys_values[area_idx],
                                      bins)

    plt.subplot(1,5,area_idx+1)
    plt.text(2, 0.5, 'D = ' + str(np.around(J,4)))
    
plt.tight_layout()

# %%
        
ophys_color = 'green'
ephys_color = 'gray'

metric = 'lifetime_sparseness_'
stimuli = ['dg', 'sg', 'ns', 'nm1']

bins = np.arange(0,1.1,0.1)

JS = []

for stim_idx, stim in enumerate(stimuli[0:1]):
    plt.figure(2001+stim_idx, figsize=(19,4))
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
        plt.text(0.1, 0.9, 'D = ' + str(np.around(J,2)))
        
plt.tight_layout()

print(np.mean(JS))
