import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%

import pickle as pkl

rs = pkl.load(open('data/running_speeds.pkl','rb'))
ts = pkl.load(open('data/running_times.pkl','rb'))
fn = pkl.load(open('data/running_file_names.pkl','rb'))

from allensdk.core.brain_observatory_cache import BrainObservatoryCache
from allensdk.brain_observatory.brain_observatory_exceptions import EpochSeparationException

df_ephys = pd.read_csv('/mnt/nvme0/ecephys_nwb_files_20200109/ephys_units_table.csv')

boc = BrainObservatoryCache(manifest_file='/mnt/hdd0/brain_observatory_cache/manifest.json')

containers = boc.get_experiment_containers()

container_ids = [c['id'] for c in containers]

# %%

def get_layer_name_ophys(depth):
    
    if depth < 200:
        return 2
    elif depth > 200 and depth < 325:
        return 4
    elif depth >= 325 and depth < 500:
        return 5
    elif depth > 500:
        return 6

# %%

threshold = 5.0

areas = ['VISp', 'VISl', 'VISal', 'VISam', 'VISpm']
stimuli = list(rs.keys())

all_dfs = []

for stim_idx, stim in enumerate(stimuli):
    
    for exp_idx in range(len(rs[stim])):
        
        fname = fn[stim][exp_idx]
        speeds = rs[stim][exp_idx]
        times = ts[stim][exp_idx]
        
        if fname.find('2p') > 0:
            
            container_id = int(os.path.basename(fname).split('.')[0])
            idx = container_ids.index(container_id)
            cre_line = containers[idx]['cre_line']
            area = containers[idx]['targeted_structure']
            layer = get_layer_name_ophys(containers[idx]['imaging_depth'])
            is_ophys = True
            
            ID = container_id
            
        else:
            
            if fname.find('spikes') > 0:
                
                specimen_id = int(os.path.basename(fname).split('.')[0].split('e')[1])
                sub_df = df_ephys[df_ephys.specimen_id == specimen_id]
                ID = specimen_id
            else:
                session_id = int(os.path.basename(fname).split('_')[1].split('.')[0])
                sub_df = df_ephys[df_ephys.ecephys_session_id == session_id]
                ID = session_id
                
            cre_line = sub_df.iloc[0].genotype
            area = [a for a in sub_df.ecephys_structure_acronym.unique() if np.isin(a, areas)]
            layer = np.nan
            is_ophys = False
            
        average_speed = np.mean(speeds)
        fraction = np.mean(speeds > threshold)
        
        if cre_line != 'Slc17a7-IRES2-Cre;Camk2a-tTA;Ai94':
        
            all_dfs.append(pd.DataFrame(index=[ID], data = {'area' : [area],
                                                      'layer' : [layer],
                                                      'is_ophys' : [is_ophys],
                                                      'cre_line' : [cre_line],
                                                      'average_speed' : [average_speed],
                                                      'fraction' : [fraction],
                                                      'stimulus' : [stim]}))
            
# %%            

df = pd.concat(all_dfs) 

# %%

from scipy.stats import ranksums

plt.figure(1999)
plt.clf()

def plot_points(sub_df, metric, x_offset=0):
    
    values = sub_df[metric]
    
    mean_values = sub_df.groupby('cre_line').mean()['average_speed'].sort_values()
    
    for idx, cre_line in enumerate(mean_values.index.values):
        cre_df = sub_df[sub_df.cre_line == cre_line]
        new_offset = x_offset + idx / len(mean_values)
        x = np.random.rand(len(cre_df)) / len(mean_values)
        cre_values = cre_df[metric]
        plt.scatter(x + new_offset, cre_values, 
                    c=color_mapping[cre_line], s=2, alpha=0.55)
        m = np.mean(cre_values)
        s = np.std(cre_values)
        plt.plot([new_offset, new_offset], [m-s, m+s], linewidth=5, alpha=0.4,
                 color = color_mapping[cre_line])

    
    M = np.median(values)
    plt.plot([x_offset,x_offset+1],[M,M],'k', linewidth=2.0, alpha=0.75)
    
    return values
    

for stim_idx, stim in enumerate(stimuli):
    
    plt.subplot(2,4,1+stim_idx)

    A = plot_points(df[(df.is_ophys) &
                (df.stimulus == stim)],
                'average_speed')

    B = plot_points(df[(df.is_ophys == False) &
                (df.stimulus == stim)],
                'average_speed', 2)
    
    plt.xlim([-0.5,3.5])
    
    s,p = ranksums(A,B)
    plt.text(1.5, 40, '$P$ = ' + str(np.around(p, 5)))
    plt.title(stim)
    plt.ylim([0,50])
    plt.ylabel('Average speed')
    
    plt.xticks(ticks=[0.5, 2.5], labels=['Ophys', 'Ephys'])
    plt.plot([1,2],[np.median(A), np.median(B)], '--k')
    
    [plt.gca().spines[loc].set_visible(False) for loc in ['top', 'right']]
    
    plt.subplot(2,4,1+stim_idx + 4)
    
    A = plot_points(df[(df.is_ophys) &
                (df.stimulus == stim)],
                'fraction')
    
    B = plot_points(df[(df.is_ophys == False) &
                (df.stimulus == stim)],
                'fraction',
                2)
    
    s,p = ranksums(A,B)
    plt.text(1.5, 0.85, '$P$ = ' + str(np.around(p, 5)))
    plt.plot([-0.5,3.5], [1., 1.], ':k', linewidth=1)
    plt.ylim([0,1.01])
    plt.ylabel('Running fraction')
    plt.xlim([-0.5,3.5])
    plt.xticks(ticks=[0.5, 2.5], labels=['Ophys', 'Ephys'])
    plt.plot([1,2],[np.median(A), np.median(B)], '--k', linewidth=2)
    
    [plt.gca().spines[loc].set_visible(False) for loc in ['top', 'right']] 
    
plt.tight_layout()       