import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%

df_ophys = pd.read_csv('data/df_ophys_with_cv.csv', index_col=0, low_memory=False)

# %%

plt.figure(100)
plt.clf()

for stim_idx, stim in enumerate(['dg', 'sg', 'ns']):
    
    sub_df = df_ophys[df_ophys['firing_rate_' + stim] > 0]
    
    CV = sub_df['std_pref_' + stim] / sub_df['mean_pref_' + stim]

    plt.subplot(2,3,1 + stim_idx)
    
    if stim == 'dg':
        rand_scatter = 0.1
    else:
        rand_scatter = 0.05
        
    plt.scatter(sub_df['sig_fraction_spont_' + stim] + 
               (np.random.rand(len(sub_df))-0.5)*rand_scatter, 
            CV, c='k',s=1, alpha=rand_scatter)
    
    plt.xlabel('Response reliability')
    plt.ylabel('Coefficient of variation (' + stim + ')')
    
    if stim == 'dg':
        plt.ylim([0,4])
    else:
        plt.ylim([0,8])
        
    plt.subplot(2,3,4 + stim_idx)
    
    plt.scatter(sub_df['mean_pref_' + stim], sub_df['std_pref_' + stim], 
                c = sub_df['sig_fraction_spont_' + stim],
                s = 0.5, cmap='cividis')
    plt.colorbar()
    plt.xlabel('Mean pref')
    plt.ylabel('Std pref')
    
    if stim == 'dg':
        xmax = 10
    else:
        xmax = 3
    
    for cv in [0.5, 1.0, 2.0]:
        x = np.arange(0, xmax, 0.1)
        y = x * cv
        plt.plot(x, y, '--k', linewidth=0.5)
    
    if stim == 'dg':
        plt.xlim([0,xmax])
        plt.ylim([0,8])
    else:
        plt.xlim([0,xmax])
        plt.ylim([0,2.0])
        
   # plt.axis('off')
    
    #sub_df = df_ephys[df_ephys['firing_rate_' + stim] > 2]
    
    
   # plt.plot(sub_df['sig_fraction_spont_' + stim] + np.random.rand(len(sub_df))*0.01, 
   ##plt.ylim([0,10])