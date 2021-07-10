import pandas as pd
import numpy as np

from common.filtering_functions import *

# %%

df_ophys = pd.read_csv('data/df_ophys_200620.csv', index_col=0, low_memory=False)
df_ephys = pd.read_csv('data/df_ephys_200620.csv', index_col=0, low_memory=False)

# %%

subselection = get_session_subselection(plot_on=True)
 
# %%    

from scipy.ndimage.filters import gaussian_filter1d
filter_window = 4
num_points = 150

metric = 'lifetime_sparseness_'
stim = 'dg'

areas = ('VISp', 'VISl', 'VISal', 'VISpm', 'VISam')
common_names = ('V1', 'LM', 'AL', 'PM', 'AM')

ephys_values = []
ophys_values = []

threshold = 0.0

plt.figure(2998)
plt.clf()

total_ephys = 0
total_synth = 0
total_ophys = 0
total_ophys_2 = 0

cells_per_area = 50

for area_idx, area in enumerate(areas):
    
    plt.subplot(1,5,area_idx+1)

    sub_df = df_ophys[(df_ophys.ecephys_structure_acronym == area) &\
                          (df_ophys['firing_rate_' + stim] > 0) &\
                            (df_ophys['sig_fraction_spont_' + stim] > 0.25)]
    
    total_ophys += len(sub_df)
    
    h, b, = np.histogram(sub_df[metric + stim], 
                         bins=np.linspace(0,1,num_points), 
                         density=True)
    
    plt.bar(b[:-1], gaussian_filter1d(h,filter_window), 
            width =np.mean(np.diff(b)),
            color='green', 
            alpha=0.2)
    
    plt.plot(b[:-1], gaussian_filter1d(h,filter_window), 
             color='green',
             linewidth=2.0)
    
    if stim == 'dg':
        ok_exps = subselection['drifting_gratings'][area]
    elif stim == 'sg':
        ok_exps = subselection['static_gratings'][area]
    elif stim == 'ns':
        ok_exps = subselection['natural_scenes'][area]
    
    container_ids, counts = np.unique(ok_exps, return_counts=True)
    
    new_df = []
    
    for cid, count in zip(container_ids, counts):
        sub_sub_df = sub_df[sub_df.container_id == cid]
        selection = np.random.permutation(len(sub_sub_df))[:count*cells_per_area]
        new_df.append(sub_sub_df.iloc[selection])
        
    sub_df = pd.concat(new_df)
    
    total_ophys_2 += len(sub_df)
    
    h, b, = np.histogram(sub_df[metric + stim], 
                         bins=np.linspace(0,1,num_points), 
                         density=True)
    
    plt.bar(b[:-1], gaussian_filter1d(h,filter_window), 
            width =np.mean(np.diff(b)),
            color='blue', 
            alpha=0.2)
    
    plt.plot(b[:-1], gaussian_filter1d(h,filter_window), 
             color='blue',
             linewidth=2.0)
    
    sub_df = df_ephys[(df_ephys['firing_rate_' + stim] > 0.0) &\
                  (df_ephys['sig_fraction_spont_' + stim] > 0.25)]
        
    total_ephys += len(sub_df)

    plt.title(common_names[area_idx])
    
    ax = plt.gca()
    plt.gca().get_yaxis().set_visible(False)
    [ax.spines[loc].set_visible(False) for loc in ['right', 'top', 'left']]  
    
    if area_idx == 0:
        plt.xlabel(metric + stim)
        
    plt.ylim([0,3.5])
    
plt.tight_layout()
print(total_ephys)
print(total_ophys)
