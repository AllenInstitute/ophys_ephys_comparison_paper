import numpy as np
import pandas as pd
import glob

# %%

available_files = glob.glob('data/parameter_grid/*.csv')

df_ophys = pd.read_csv('data/df_ophys_200620.csv', index_col=0, low_memory=False)
df_ephys = pd.read_csv('data/df_ephys_200620.csv', index_col=0, low_memory=False)


# %%

def extract_params(fname):
    
    basename = fname.split('/')[-1].split('.csv')[0]
    
    params = basename.split('_')[1:]
    
    return tuple([float(p) for p in params])

from scipy.ndimage.filters import gaussian_filter1d

estimates = np.empty((10,10,10,3))
estimates[:] = np.nan

metric = 'lifetime_sparseness_'
stim = 'dg'

areas = ['VISpm', 'VISl', 'VISp', 'VISam', 'VISal']

df3 = pd.read_csv(available_files[0])
df3 = df3.join(df_ephys['ecephys_structure_acronym'], on='unit_id')
area_counts = df3.groupby('ecephys_structure_acronym').count()['nwb']
area_ratios = area_counts / np.sum(area_counts)

selection = (df_ophys['firing_rate_' + stim] > 0.00) 
            
ophys_counts = df_ophys.groupby('ecephys_structure_acronym').count()['nwb']
ophys_ratios = ophys_counts / np.sum(ophys_counts)

new_counts = area_ratios/ area_ratios.loc['VISam'] * ophys_counts.loc['VISam']

new_df = []

for area_idx, area in enumerate(new_counts.index.values):
    sub_df = df_ophys[df_ophys.ecephys_structure_acronym == area]
    random_inds = np.random.permutation(len(sub_df))
    new_df.append(sub_df.iloc[random_inds[:int(new_counts.loc[area])]])
    
sub_df_ophys = pd.concat(new_df)

selection = (sub_df_ophys['sig_fraction_spont_' + stim] > 0.25) 

bins = np.arange(0,1.1,0.1)

ophys_values = sub_df_ophys[selection][metric + stim]#, bins, 'k', plot=False) + 0.001

sigma_est = []
a_est = []
tau_est = []

for index in range(len(available_files)):
    
    fname = available_files[index]
    
    params = extract_params(fname)
    
    sigma_est.append(params[0])
    a_est.append(params[1])
    tau_est.append(params[2])
    
sigma_est = np.sort(np.unique(np.array(sigma_est)))
a_est = np.sort(np.unique(np.array(a_est)))
tau_est = np.sort(np.unique(np.array(tau_est)))
    
for index in range(len(available_files)):
    
    fname = available_files[index]
    
    params = extract_params(fname)
    
    df3 = pd.read_csv(fname)
    df3 = df3.join(df_ephys['ecephys_structure_acronym'], on='unit_id')
    df3 = df3[df3.ecephys_structure_acronym.isin(areas)]

    i0 = np.where(sigma_est == params[0])[0][0]
    i1 = np.where(a_est == params[1])[0][0]
    i2 = np.where(tau_est == params[2])[0][0]
    
    sub_df = df3[df3['sig_fraction_spont_' + stim] > 0.25]
    ephys_values = sub_df[metric + stim].values
    
    S, W, J = compare_distributions(ophys_values, ephys_values, bins)
    
    estimates[i0, i1, i2, 0] = np.nanmedian(ephys_values)
    estimates[i0, i1, i2, 1] = len(sub_df) / len(df3)
    estimates[i0, i1, i2, 2] = J

# %%
        
plt.figure(199)
plt.clf()

matplotlib.rcParams.update({'font.size': 10})

M = np.median(sub_df_ophys[selection][metric + stim])

hilite = [0.047, 0.37]
hilite_idx = 1
 
for metric_idx in range(3):
    
    if metric_idx == 0:
        max_val = M
    elif metric_idx == 1:
        max_val =1.0
    else:
        max_val = 0.5
    
    for i in range(10):
    
        
        plt.subplot(6,5,i+1+10*metric_idx)
        plt.imshow(estimates[:,i,:,metric_idx], aspect='auto', origin='lower', cmap='cividis',
                       extent=(np.min(sigma_est), np.max(sigma_est), 
                                               np.min(tau_est), np.max(tau_est)), vmin=0, vmax=max_val)
        plt.title('A = ' + str(a_est[i]), fontsize=10)
        if i == hilite_idx:
            plt.plot(hilite[0], hilite[1], 's', color='white',markersize=10, linewidth=1,markerfacecolor=None)

        plt.xlabel('$\sigma$')
        plt.ylabel('$tau$')
        plt.yticks(ticks=[0.5,1.0,1.5])
        plt.xticks(ticks=[0.05,0.10])
    
plt.tight_layout()

