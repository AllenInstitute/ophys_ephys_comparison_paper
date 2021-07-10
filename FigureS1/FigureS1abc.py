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

metric = 'pref_ori_'
stim = 'dg'

areas = ['VISp', 'VISl', 'VISal', 'VISpm', 'VISam']

plt.figure(2010, figsize=(15,4))
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
    

for area_idx, area in enumerate(areas):
    
    _, ophys_inds = np.unique(np.sort(ophys_values[area_idx]), return_inverse=True)
    _, ephys_inds = np.unique(np.sort(ephys_values[area_idx]), return_inverse=True)

    S, W, J = compare_distributions(ophys_inds,
                                      ephys_inds,
                                      np.arange(0,len(_) - 0.9,1.0))

    plt.subplot(1,5,area_idx+1)
    plt.text(2, 0.01, 'D = ' + str(np.around(J,4)))
    
plt.tight_layout()

# %%

ophys_color = 'green'
ephys_color = 'gray'

metric = 'pref_sf_'
stim = 'sg'

areas = ['VISp', 'VISl', 'VISal', 'VISpm', 'VISam']

plt.figure(2010, figsize=(15,4))
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
    

for area_idx, area in enumerate(areas):
    
    _, ophys_inds = np.unique(np.sort(ophys_values[area_idx]), return_inverse=True)
    _, ephys_inds = np.unique(np.sort(ephys_values[area_idx]), return_inverse=True)

    S, W, J = compare_distributions(ophys_inds,
                                      ephys_inds,
                                      np.arange(0,len(_) - 0.9,1.0))

    plt.subplot(1,5,area_idx+1)
    plt.text(2, 0.01, 'D = ' + str(np.around(J,4)))
    
plt.tight_layout()

# %%

ophys_color = 'green'
ephys_color = 'gray'

metric = 'pref_ori_'
stim = 'sg'

areas = ['VISp', 'VISl', 'VISal', 'VISpm', 'VISam']

plt.figure(2010, figsize=(15,4))
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
    

for area_idx, area in enumerate(areas):
    
    _, ophys_inds = np.unique(np.sort(ophys_values[area_idx]), return_inverse=True)
    _, ephys_inds = np.unique(np.sort(ephys_values[area_idx]), return_inverse=True)

    S, W, J = compare_distributions(ophys_inds,
                                      ephys_inds,
                                      np.arange(0,len(_) - 0.9,1.0))

    plt.subplot(1,5,area_idx+1)
    plt.text(2, 0.01, 'D = ' + str(np.around(J,4)))
    
plt.tight_layout()