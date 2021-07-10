import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%

df_ophys = pd.read_csv('/mnt/nvme0/ophys-ephys/df_ophys_200620.csv', index_col=0, low_memory=False)
df_ephys = pd.read_csv('/mnt/nvme0/ophys-ephys/df_ephys_200620.csv', index_col=0, low_memory=False)

df = pd.read_csv('data/unit_table_running_speed_metrics.csv',
                 index_col=0).rename_axis(index='unit_id')

# %%
  
df_ephys = df_ephys.merge(df, on='unit_id')
df_ophys = df_ophys.merge(df, on='unit_id')

# %%

from scipy.stats import wilcoxon

plt.figure(1933)
plt.clf()

for stim_idx, stim in enumerate(['dg','ns', 'sg']):
        
    plt.subplot(3,3,stim_idx*3+1)
    
    A = df_ophys['sig_fraction_spont_' + stim + '_stationary']
    B = df_ophys['sig_fraction_spont_' + stim + '_running']
    
    mask = np.invert(np.isnan(A)) * np.invert(np.isnan(B))

    
    if stim == 'dg':
        mult = 0.15
    else:
        mult = 0.05
        
    shiftA = np.random.rand(len(A)) * mult
    shiftB = np.random.rand(len(B)) * mult

    plt.scatter((A+shiftA)/(1+mult),
                (B+shiftB)/(1+mult),s =1,c='green',alpha=0.5)
    plt.plot([0,0.25],[0.25,0.25],'k',alpha=1)
    plt.plot([0.25,0.25],[0,0.25],'k',alpha=1)
    plt.xlabel('Stationary')
    plt.ylabel('Running')
    plt.title('Sig fraction ' + stim + ': ophys')
    
    stat, p = wilcoxon(A,B)
    
    plt.subplot(3,3,stim_idx*3+3)
    plt.bar(0,np.mean(A[mask] > 0.25),color='green')
    plt.bar(1,np.mean(B[mask] > 0.25), color='lightgreen')

    plt.subplot(3,3,stim_idx*3+2)
    
    A = df_ephys['sig_fraction_spont_' + stim + '_stationary']
    B = df_ephys['sig_fraction_spont_' + stim + '_running']
    
    mask = np.invert(np.isnan(A)) * np.invert(np.isnan(B))
    
    shiftA = np.random.rand(len(A)) * mult
    shiftB = np.random.rand(len(B)) * mult
    
    plt.scatter((A+shiftA)/(1+mult),
                (B+shiftB)/(1+mult),s =1,c='gray',alpha=0.5)
    plt.plot([0,0.25],[0.25,0.25],'k',alpha=1)
    plt.plot([0.25,0.25],[0,0.25],'k',alpha=1)
    plt.xlabel('Stationary')
    plt.ylabel('Running')
    plt.title('Sig fraction ' + stim + ': ephys')
    
    stat, p = wilcoxon(A,B)
    
    plt.subplot(3,3,stim_idx*3 +3)
    plt.bar(3,np.mean(A[mask] > 0.25),color='gray')
    plt.bar(4,np.mean(B[mask] > 0.25), color='lightgray')
    plt.ylabel('Fraction responding')
    
    [plt.gca().spines[loc].set_visible(False) for loc in ['top', 'right']] 
    plt.xlim([-1.5,5.5])
    
    plt.xticks(ticks=[0,1,3,4],labels=['stationary','running','stationary','running'],rotation=45)
    
plt.tight_layout()

# %%

from scipy.stats import wilcoxon

plt.figure(1934)
plt.clf()

for stim_idx, stim in enumerate(['dg','ns', 'sg']):
        
    plt.subplot(3,3,stim_idx*3+1)
    
    A = df_ophys['lifetime_sparseness_' + stim + '_stationary']
    B = df_ophys['lifetime_sparseness_' + stim + '_running']
    
    plt.scatter(A,
                B,s =1,c='green', alpha=0.5)
    plt.plot([0,1],[0,1],'--k')
    plt.xlim([0,1])
    plt.ylim([0,1])  

    plt.xlabel('Stationary')
    plt.ylabel('Running')
    plt.title('Lifetime sparseness ' + stim + ': ophys')
    
    stat, p = wilcoxon(A,B)
    
    plt.subplot(3,3,stim_idx*3+3)
    x = np.random.rand(len(A))
    plt.plot(x, B-A, '.', markersize=1, color='green', alpha=0.5)
    M = np.nanmean(B-A)
    plt.plot([0,1],[M,M],'-k',linewidth=3.0)
      

    plt.subplot(3,3,stim_idx*3+2)
    
    A = df_ephys['lifetime_sparseness_' + stim + '_stationary']
    B = df_ephys['lifetime_sparseness_' + stim + '_running']
    
    plt.scatter(A, B,s =1, c='gray', alpha=0.5)
    plt.plot([0,1],[0,1],'--k')
    plt.xlim([0,1])
    plt.ylim([0,1])  
    plt.xlabel('Stationary')
    plt.ylabel('Running')
    plt.title('Lifetime sparseness ' + stim + ': ephys')
    
    stat, p = wilcoxon(A,B)
    
    plt.subplot(3,3,stim_idx*3 +3)
    x = np.random.rand(len(A))
    plt.plot(x+2, B-A, '.', markersize=1, color='gray', alpha=0.5)
    M = np.nanmean(B-A)
    plt.plot([2,3],[M,M],'-k',linewidth=3.0)
    plt.xlim([-0.5,3.5])
    plt.ylim([-0.4,0.4])
    plt.plot([-0.5,3.5],[0,0],'--k', alpha=0.5)
    plt.ylabel('Running - stationary')
    
    [plt.gca().spines[loc].set_visible(False) for loc in ['top', 'right']] 
    
plt.tight_layout()