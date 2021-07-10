import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%

from scipy.ndimage.filters import gaussian_filter1d
import matplotlib

df_ophys = pd.read_csv('data/df_ophys_200620.csv', index_col=0, low_memory=False)
df_ephys = pd.read_csv('data/df_ephys_200620.csv', index_col=0, low_memory=False)

directory = 'data/response_metrics_nnd'

# %%
baseline_selectivity_ephys = df_ephys[(df_ephys.ecephys_structure_acronym == 'VISp') &
                                      (df_ephys.sig_fraction_spont_dg > 0.25)].lifetime_sparseness_dg

baseline_selectivity_ophys = df_ophys[(df_ophys.ecephys_structure_acronym == 'VISp') &
                                      (df_ophys.sig_fraction_spont_dg > 0.25)].lifetime_sparseness_dg

# %%

levels = ['0.0','1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0']

plt.figure(1)
plt.clf()
    
num_points = 120

cmap = matplotlib.cm.get_cmap('cividis')

for level_idx, level in enumerate(levels):
    
    files = glob.glob(directory + '/*_' + level + '*.csv')
    
    df = []
    
    for file_idx, file in enumerate(files):
        
        df.append(pd.read_csv(file, index_col=0, low_memory=False))
        
    df = pd.concat(df)
        
    plt.subplot(3, len(levels) // 3, level_idx+1)
    
    sub_df = df[df.sig_fraction_spont_dg > 0.25]
    
    smoothed_hist(sub_df.lifetime_sparseness_dg,
                  np.linspace(0,1,num_points),
                  linecolor='k',
                  fillcolor= cmap(level_idx / 11))
   
    if level_idx == 3:
        smoothed_hist(baseline_selectivity_ephys,
                      np.linspace(0,1,num_points),
                      linecolor='grey',
                      fillcolor= None,
                      linestyle='--')
        
        
        S,W,J = compare_distributions(sub_df.lifetime_sparseness_dg, baseline_selectivity_ephys,
                                      np.arange(0,1.1,0.1))
        
        plt.text(0.8, 1, str(np.around(J,2)), color='grey')
    
    if level_idx == 4:
        smoothed_hist(baseline_selectivity_ophys,
                      np.linspace(0,1,num_points),
                      linecolor='green',
                      fillcolor= None,
                      linestyle='--')
        
        S,W,J = compare_distributions(sub_df.lifetime_sparseness_dg, baseline_selectivity_ophys,
                                      np.arange(0,1.1,0.1))
        
        plt.text(0.8, 2, str(np.around(J,2)), color='green')
    
    plt.title(len(sub_df))
    
    plt.text(0.5, 1, level)
    
    axis = plt.gca()
    axis.get_yaxis().set_visible(False)
    [axis.spines[loc].set_visible(False) for loc in ['right', 'top', 'left']]  
    
plt.tight_layout()

# %%

plt.figure(2)
plt.clf()

stimuli = ['dg', 'sg', 'ns']
levels = ['0.0','1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0', '10.0']


for level_idx, level in enumerate(levels):
    
    files = glob.glob(directory + '/*_' + level + '*.csv')
    
    df = []
    
    for file_idx, file in enumerate(files):
        
        df.append(pd.read_csv(file, index_col=0, low_memory=False))
        
    df = pd.concat(df)
    
    for stim_idx, stim in enumerate(stimuli):
    
        plt.subplot(1, 3, stim_idx+1)
        
        sub_df = df[df['firing_rate_' + stim] > 0.0]
            
        M = np.mean(sub_df['sig_fraction_spont_' + stim] > 0.25)
        
        plt.bar(level_idx, M, 
                width = 0.5,
                color=cmap(level_idx / 11), 
                alpha= 0.6)
        
        if level_idx == 0:
            plt.title(str(len(sub_df)))
            
            baseline_resp_ephys = df_ephys[(df_ephys.ecephys_structure_acronym == 'VISp') &
                                      (df_ephys['firing_rate_' + stim] > 0.0)]['sig_fraction_spont_' + stim]

            baseline_resp_ophys = df_ophys[(df_ophys.ecephys_structure_acronym == 'VISp') &
                                      (df_ophys['firing_rate_' + stim] > 0.0)]['sig_fraction_spont_' + stim]

            
            M_ephys = np.mean(baseline_resp_ephys > 0.25)
            plt.plot([-0.5,10.5],[M_ephys, M_ephys],'--',color='grey')
            
            M_ophys = np.mean(baseline_resp_ophys > 0.25)
            plt.plot([-0.5,10.5],[M_ophys, M_ophys],'--',color='green')
        
        axis = plt.gca()
        [axis.spines[loc].set_visible(False) for loc in ['top', 'right']]  
    
        plt.ylim([0,1])
    