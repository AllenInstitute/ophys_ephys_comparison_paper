import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from common.filtering_functions import *
from common.plotting_functions import *
from common.distance_metrics import *

# %%

df_ophys = pd.read_csv('data/df_ophys_200620.csv', index_col=0, low_memory=False)
df_ephys = pd.read_csv('data/df_ephys_200620.csv', index_col=0, low_memory=False)

# %%

ophys_color = 'green'
ophys_resampled_color = 'limegreen'

ephys_color = 'gray'
ephys_resampled_color = 'lightgray'

areas = ['VISp', 'VISl', 'VISal', 'VISpm', 'VISam']
common_names = ['V1','LM', 'AL', 'PM', 'AM']

plt.figure(num=19191, figsize=(15,4))
plt.clf()

matplotlib.rcParams.update({'font.size': 14})

layers = [2,4,5,6]

for area_idx, area in enumerate(areas):
    
    plt.subplot(1,5,area_idx+1)

    sub_df = df_ophys[(df_ophys.ecephys_structure_acronym == area)]
    
    h,b = np.histogram(sub_df['cortical_layer'], bins=np.arange(8), density=True)
    
    x = np.arange(4)-0.15
    y_ophys = h[np.array([2,4,5,6])]

    plt.barh(x, y_ophys, height=0.2, color=ophys_color)
    
    sub_df = df_ephys[(df_ephys.ecephys_structure_acronym == area)]
        
    h,b = np.histogram(sub_df['cortical_layer'], bins=np.arange(8), density=True)

    x = np.arange(4)+0.15
    y_ephys = h[np.array([2,4,5,6])]

    plt.barh(x, y_ephys, height=0.2, color=ephys_color)
    plt.yticks(ticks=np.arange(4), labels=['L2/3','L4','L5','L6'])
    plt.title(common_names[area_idx])
    plt.xlim([0,0.65])
    if area_idx == 0:
        plt.xlabel('Fraction')
        
    plt.gca().invert_yaxis()
        
    ax = plt.gca()
    [ax.spines[loc].set_visible(False) for loc in ['top', 'right']] 
    
plt.tight_layout()
        
# %%

new_df_ophys, new_df_ephys = apply_layer_matching(df_ophys, df_ephys)

plt.figure(num=19192, figsize=(15,4))
plt.clf()

layers = [2,4,5,6]

for area_idx, area in enumerate(areas):
    
    plt.subplot(1,5,area_idx+1)

    sub_df = new_df_ophys[(new_df_ophys.ecephys_structure_acronym == area)]
    
    h,b = np.histogram(sub_df['cortical_layer'], bins=np.arange(8), density=True)
    
    x = np.arange(4)-0.15
    y_ophys = h[np.array([2,4,5,6])]

    plt.barh(x, y_ophys, height=0.2, color=ophys_resampled_color)

    
    sub_df = new_df_ephys[(new_df_ephys.ecephys_structure_acronym == area)]
        
    h,b = np.histogram(sub_df['cortical_layer'], bins=np.arange(8), density=True)

    x = np.arange(4)+0.15
    y_ephys = h[np.array([2,4,5,6])]

    plt.barh(x, y_ephys, height=0.2, color=ephys_resampled_color)
    plt.yticks(ticks=np.arange(4), labels=['L2/3','L4','L5','L6'])
    plt.title(common_names[area_idx])
    plt.xlim([0,0.65])
    if area_idx == 0:
        plt.xlabel('Fraction')
        
    plt.gca().invert_yaxis()
        
    ax = plt.gca()
    [ax.spines[loc].set_visible(False) for loc in ['top', 'right']] 
    
plt.tight_layout()

# %%

bins = np.arange(0,1.1,0.1)

plt.figure(191)
plt.clf()

metric = 'sig_fraction_spont_'
stim = 'dg'

for area_idx, area in enumerate(areas):
    
    ophys_values = df_ophys[(df_ophys['firing_rate_' + stim] > 0) &
                            (df_ophys.ecephys_structure_acronym == area)][metric + stim]
    
    new_ophys_values = new_df_ophys[(new_df_ophys['firing_rate_' + stim] > 0) &
                            (new_df_ophys.ecephys_structure_acronym == area)][metric + stim]
    
    plt.subplot(2,5,area_idx+1)
    h1 = cumulative_histogram(ophys_values, bins, ophys_color)
    h2 = cumulative_histogram(new_ophys_values, bins, ophys_resampled_color, style='--')
    plt.ylim([0,1.05])
    plt.title(common_names[area_idx])
    plt.xticks(ticks = np.arange(0,1.1,0.25),
              labels=['']*5)
    plt.yticks(ticks = [0.0, 0.5, 1.0],
              labels=['']*3)
    
    S, W, J = compare_distributions(ophys_values, new_ophys_values, bins)
    plt.text(0.5, 0.15,'D = ' + str(np.around(J,5)) + '\n')
    
    ephys_values = df_ephys[(df_ephys['firing_rate_' + stim] > 0) &
                            (df_ephys.ecephys_structure_acronym == area)][metric + stim]
    
    new_ephys_values = new_df_ephys[(new_df_ephys['firing_rate_' + stim] > 0) &
                            (new_df_ephys.ecephys_structure_acronym == area)][metric + stim]
    
    plt.subplot(2,5,area_idx+6)
    h1 = cumulative_histogram(ephys_values, bins, ephys_color)
    h2 = cumulative_histogram(new_ephys_values, bins, ephys_resampled_color, style='--')
    plt.ylim([0,1.05])
    
    S, W, J = compare_distributions(ephys_values, new_ephys_values, bins)
    plt.text(0.5, 0.15,'D = ' + str(np.around(J,5)) + '\n')
    
    if area_idx == 0:
        plt.xticks(ticks = np.arange(0,1.1,0.25), labels=['0.0', '','0.5', '', '1.0'])
        plt.xlabel('Significant trials')
        plt.yticks(ticks = [0.0, 0.5, 1.0])
        plt.ylabel('Cumulative fraction')
    else:
        plt.xticks(ticks = np.arange(0,1.1,0.25),
              labels=['']*5)
    

# %%
        
bins = np.arange(0,1.1,0.1)

plt.figure(192)
plt.clf()

metric = 'lifetime_sparseness_'
stim = 'dg'

for area_idx, area in enumerate(areas):
    
    ophys_values = df_ophys[(df_ophys['sig_fraction_spont_' + stim] > 0.25) &
                            (df_ophys.ecephys_structure_acronym == area)][metric + stim]
    
    new_ophys_values = new_df_ophys[(new_df_ophys['sig_fraction_spont_' + stim] > 0.25) &
                            (new_df_ophys.ecephys_structure_acronym == area)][metric + stim]
    
    plt.subplot(2,5,area_idx+1)
    h1 = cumulative_histogram(ophys_values, bins, ophys_color)
    h2 = cumulative_histogram(new_ophys_values, bins, ophys_resampled_color, style='--')
    plt.ylim([0,1.05])
    plt.title(common_names[area_idx])
    plt.xticks(ticks = np.arange(0,1.1,0.25),
              labels=['']*5)
    plt.yticks(ticks = [0.0, 0.5, 1.0],
              labels=['']*3)
    
    S, W, J = compare_distributions(ophys_values, new_ophys_values, bins)
    plt.text(0.5, 0.15,'D = ' + str(np.around(J,5)) + '\n')
    
    ephys_values = df_ephys[(df_ephys['sig_fraction_spont_' + stim] > 0.25) &
                            (df_ephys.ecephys_structure_acronym == area)][metric + stim]
    
    new_ephys_values = new_df_ephys[(new_df_ephys['sig_fraction_spont_' + stim] > 0.25) &
                            (new_df_ephys.ecephys_structure_acronym == area)][metric + stim]
    
    plt.subplot(2,5,area_idx+6)
    h1 = cumulative_histogram(ephys_values, bins, ephys_color)
    h2 = cumulative_histogram(new_ephys_values, bins, ephys_resampled_color, style='--')
    plt.ylim([0,1.05])
    
    S, W, J = compare_distributions(ephys_values, new_ephys_values, bins)
    plt.text(0.5, 0.15,'D = ' + str(np.around(J,4)) + '\n')
    
    if area_idx == 0:
        plt.xticks(ticks = np.arange(0,1.1,0.25), labels=['0.0', '','0.5', '', '1.0'])
        plt.xlabel('Lifetime sparseness')
        plt.yticks(ticks = [0.0, 0.5, 1.0])
        plt.ylabel('Cumulative fraction')
    else:
        plt.xticks(ticks = np.arange(0,1.1,0.25),
              labels=['']*5)
    
# %%
    
bins = np.arange(0,5.1,1.0)

plt.figure(193)
plt.clf()

metric = 'pref_tf_'
stim = 'dg'

for area_idx, area in enumerate(areas):
    
    ophys_values = np.ceil(np.log2(df_ophys[(df_ophys['sig_fraction_spont_' + stim] > 0.25) &
                            (df_ophys.ecephys_structure_acronym == area)][metric + stim]))
    
    new_ophys_values = np.ceil(np.log2(new_df_ophys[(new_df_ophys['sig_fraction_spont_' + stim] > 0.25) &
                            (new_df_ophys.ecephys_structure_acronym == area)][metric + stim]))
    
    plt.subplot(2,5,area_idx+1)
    h1 = cumulative_histogram(ophys_values, bins, ophys_color)
    h2 = cumulative_histogram(new_ophys_values, bins, ophys_resampled_color, style='--')
    plt.ylim([0,1.05])
    plt.title(common_names[area_idx])
    
    plt.xticks(ticks = np.arange(5),
              labels=['']*5)
    plt.yticks(ticks = [0.0, 0.5, 1.0],
              labels=['']*3)
    
    S, W, J = compare_distributions(ophys_values, new_ophys_values, bins)
    plt.text(2, 0.15,'D = ' + str(np.around(J,5)) + '\n')
    
    ephys_values = np.ceil(np.log2(df_ephys[(df_ephys['sig_fraction_spont_' + stim] > 0.25) &
                            (df_ephys.ecephys_structure_acronym == area)][metric + stim]))
    
    new_ephys_values = np.ceil(np.log2(new_df_ephys[(new_df_ephys['sig_fraction_spont_' + stim] > 0.25) &
                            (new_df_ephys.ecephys_structure_acronym == area)][metric + stim]))
    
    plt.subplot(2,5,area_idx+6)
    h1 = cumulative_histogram(ephys_values, bins, ephys_color)
    h2 = cumulative_histogram(new_ephys_values, bins, ephys_resampled_color, style='--')
    plt.ylim([0,1.05])
    
    S, W, J = compare_distributions(ephys_values, new_ephys_values, bins)
    plt.text(2, 0.15,'D = ' + str(np.around(J,4)) + '\n')
    
    if area_idx == 0:
        plt.xticks(ticks = np.arange(5),
              labels=['1','2','4','8','15 Hz'])
        plt.xlabel('Preferred temporal frequency')
        plt.yticks(ticks = [0.0, 0.5, 1.0])
        plt.ylabel('Cumulative fraction')
    else:
        plt.xticks(ticks = np.arange(5),
              labels=['']*5)
    

    
    