import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from common.distance_metrics import *
from common.plotting_functions import *

# %%

df_ophys = pd.read_csv('/mnt/nvme0/ophys-ephys/df_ophys_200620.csv', index_col=0, low_memory=False)
df_ephys = pd.read_csv('/mnt/nvme0/ophys-ephys/df_ephys_200620.csv', index_col=0, low_memory=False)

# %%

def get_session_subselection(plot_on=False):
    
    df = pd.read_csv('data/running_speed_info.csv',
                 index_col=0)

    subselection  = {}
    
    areas = ['VISp', 'VISl', 'VISal', 'VISpm', 'VISam']
    common_names = ['V1','LM', 'AL', 'PM', 'AM']
    stimuli = ['drifting_gratings', 'static_gratings', 'natural_scenes', 'natural_movie_three']
    
    for stim_idx, stim in enumerate(stimuli):
        
        subselection[stim] = {}
        
        for area_idx, area in enumerate(areas):
            
            sub_df = df[(df.is_ophys) &
                    (df.stimulus == stim) &
                    (df.area == area)]
            
            fraction_ophys = np.sort(sub_df.fraction.values)
            
            if plot_on:
                plt.subplot(len(stimuli),len(areas),1+stim_idx*len(areas) + area_idx)
                h,b = np.histogram(fraction_ophys, bins =np.arange(0,1,0.01), density=True)
                plt.plot(b[:-1], np.cumsum(h)/100, 'g')
            
            sub_df = df[(df.is_ophys == False) &
                    (df.stimulus == stim)]
            
            mask = sub_df.area.apply(lambda x: area in x)
            
            sub_df = sub_df[mask]
            
            fraction_ephys = np.sort(sub_df.fraction.values)
            
            if plot_on:
                h,b = np.histogram(fraction_ephys, bins =np.arange(0,1,0.01), density=True)
                plt.plot(b[:-1], np.cumsum(h)/100, color='gray')
            
            inds = np.searchsorted(fraction_ophys, fraction_ephys)
            inds = np.delete(inds, np.where(inds == len(fraction_ophys)))
        
            fraction_ophys_2 = fraction_ophys[inds]
            
            sub_df = df[(df.is_ophys) &
                    (df.stimulus == stim) &
                    (df.area == area)]
            
            ophys_exps = np.sort(sub_df.index.values[inds])
            
            subselection[stim][area] = ophys_exps
            
            if plot_on:
                h,b = np.histogram(fraction_ophys_2, bins =np.arange(0,1,0.01), density=True)
                plt.plot(b[:-1], np.cumsum(h)/100, color='tab:blue')
                plt.ylim([0,1])
                plt.yticks(ticks=[0,0.5,1.0])
            
                if stim_idx == 0:
                    plt.title(common_names[area_idx])
                if area_idx == 0:
                    plt.ylabel(stim)
                    
                [plt.gca().spines[loc].set_visible(False) for loc in ['top', 'right']] 
        
    return subselection

def apply_behavior_matching(df_ophys, df_ephys, stim, cells_per_area = 50):
    
    subselection = get_session_subselection()
    
    areas = ['VISp', 'VISl', 'VISal', 'VISpm', 'VISam']
    
    new_df = []
    
    for area_idx, area in enumerate(areas):
        
        sub_df = df_ophys[(df_ophys.ecephys_structure_acronym == area) &\
                          (df_ophys['firing_rate_' + stim] > 0)]
        
        if stim == 'dg':
            ok_exps = subselection['drifting_gratings'][area]
        elif stim == 'sg':
            ok_exps = subselection['static_gratings'][area]
        elif stim == 'ns':
            ok_exps = subselection['natural_scenes'][area]
        
        container_ids, counts = np.unique(ok_exps, return_counts=True)
 
        for cid, count in zip(container_ids, counts):
            sub_sub_df = sub_df[sub_df.container_id == cid]
            selection = np.random.permutation(len(sub_sub_df))[:count*cells_per_area]
            new_df.append(sub_sub_df.iloc[selection])
            
    return pd.concat(new_df), df_ephys

def apply_layer_matching(df_ophys, df_ephys):
    
    areas = ['VISp', 'VISl', 'VISal', 'VISpm', 'VISam']
    
    layers = np.array([2,4,5,6])
    
    new_df_ophys = []
    new_df_ephys = []
    
    np.random.seed(192)
    
    for area_idx, area in enumerate(areas):
    
        sub_df_ophys = df_ophys[(df_ophys.ecephys_structure_acronym == area)]
        h,b = np.histogram(sub_df_ophys['cortical_layer'], bins=np.arange(8), density=True)
        ophys_fraction = h[layers]
        h,b = np.histogram(sub_df_ophys['cortical_layer'], bins=np.arange(8), density=False)
        ophys_count = h[layers]
        
        sub_df_ephys = df_ephys[(df_ephys.ecephys_structure_acronym == area)]
        h,b = np.histogram(sub_df_ephys['cortical_layer'], bins=np.arange(8), density=True)
        ephys_fraction = h[layers]
        h,b = np.histogram(sub_df_ephys['cortical_layer'], bins=np.arange(8), density=False)
        ephys_count = h[layers]
        
        ephys_fraction[-1] = 0 # remove layer 6
        ephys_fraction = ephys_fraction / np.sum(ephys_fraction) # new fraction
        
        new_ophys_count = (ephys_fraction / np.max(ephys_fraction) * ophys_count[2]).astype('int')
        new_ephys_count = np.copy(ephys_count)
        new_ephys_count[-1] = 0
        
        for layer_idx, layer in enumerate(layers):
            
            layer_df_ophys = sub_df_ophys[sub_df_ophys.cortical_layer == layer]
            layer_df_ephys = sub_df_ephys[sub_df_ephys.cortical_layer == layer]
                
            order = np.random.permutation(len(layer_df_ophys))
            new_df_ophys.append(layer_df_ophys.iloc[order[:new_ophys_count[layer_idx]]])
            new_df_ephys.append(layer_df_ephys.iloc[np.arange(new_ephys_count[layer_idx])]) 
            
    new_df_ophys = pd.concat(new_df_ophys)
    new_df_ephys = pd.concat(new_df_ephys)
    
    return new_df_ophys, new_df_ephys

def apply_forward_model(df_ophys, df_ephys):
    
    metrics = ['sig_fraction_spont_', 'lifetime_sparseness_']
    stimuli = ['dg', 'sg', 'ns']
    
    df_ephys_synth_ml = pd.read_csv('data/unit_table_synth_ophys_200325.csv', index_col=0)

    df_ephys = df_ephys.join(df_ephys_synth_ml, on='unit_id', rsuffix='_ophys_ml', how='inner')
    
    for metric in metrics:
        for stimulus in stimuli:
            df_ephys[metric + stimulus] = df_ephys[metric + stimulus + '_ophys_ml']
    
    df_ephys['pref_tf_dg'] = df_ephys['pref_tf_dg_ophys_ml']
    df_ephys['pref_sf_sg'] = df_ephys['pref_sf_sg_ophys_ml']
    
    return df_ophys, df_ephys

def apply_ISI_selection(df_ophys, df_ephys):
    
    return df_ophys, df_ephys[df_ephys.isi_violations < 0.001]


def apply_event_rate_selection(df_ophys, df_ephys, stim):
    
    areas = ['VISp', 'VISl', 'VISal', 'VISpm', 'VISam']

    new_df_ophys = []
    
    for area_idx, area in enumerate(areas):
        
        sub_df = df_ophys[(df_ophys.ecephys_structure_acronym == area) &
                          (df_ophys['firing_rate_' + stim] > 0.0)]
        
        threshold = np.quantile(sub_df['firing_rate_' + stim], 0.7)
        
        new_df_ophys.append(sub_df[sub_df['firing_rate_' + stim] > threshold])
    
    return pd.concat(new_df_ophys), df_ephys

def distance_summary(df_ophys, df_ephys, stimulus_filter = None):
    
    metric = 'sig_fraction_spont_'
    stimuli = ['dg', 'sg', 'ns', 'nm3']
    areas = ['VISp', 'VISl', 'VISal', 'VISpm', 'VISam']
    
    if stimulus_filter == None:
        check_stim = lambda x : True
    else:
        check_stim = lambda x : x == stimulus_filter
    
    support = np.arange(0,1.1,0.1)
    
    distances = np.zeros((4, 3, 5, 3)) # stimuli, metrics, areas, distance types
    unit_count = np.zeros((4, 3, 5, 2)) # stimuli, metrics, areas, modality
    
    for stim_idx, stim in enumerate(stimuli):
        
        if check_stim(stim):
        
            for area_idx, area in enumerate(areas):
                
               ophys_selection = df_ophys['firing_rate_' + stim] > 0
               ephys_selection = df_ephys['firing_rate_' + stim] > 0
        
               ophys_selection &= df_ophys.ecephys_structure_acronym == area
               ephys_selection &= df_ephys.ecephys_structure_acronym == area
               
               ophys_values = df_ophys[ophys_selection][metric + stim]
               ephys_values = df_ephys[ephys_selection][metric + stim]
        
               S, W, J = compare_distributions(ophys_values, ephys_values, support)
                
               distances[stim_idx, 0, area_idx, 0] = S
               distances[stim_idx, 0, area_idx, 1] = W
               distances[stim_idx, 0, area_idx, 2] = J
               
               unit_count[stim_idx, 0, area_idx, 0] = len(ophys_values)
               unit_count[stim_idx, 0, area_idx, 1] = len(ephys_values)
    
    metric = 'lifetime_sparseness_'
    
    support = np.arange(0,1.1,0.1)
    
    for stim_idx, stim in enumerate(stimuli):
        
        if check_stim(stim):
        
            for area_idx, area in enumerate(areas):
                
               ophys_selection = df_ophys['sig_fraction_spont_' + stim] > 0.25
               ephys_selection = df_ephys['sig_fraction_spont_' + stim] > 0.25
        
               ophys_selection &= df_ophys.ecephys_structure_acronym == area
               ephys_selection &= df_ephys.ecephys_structure_acronym == area
               
               ophys_values = df_ophys[ophys_selection][metric + stim]
               ephys_values = df_ephys[ephys_selection][metric + stim]
        
               S, W, J = compare_distributions(ophys_values, ephys_values, support)
                
               distances[stim_idx, 1, area_idx, 0] = S
               distances[stim_idx, 1, area_idx, 1] = W
               distances[stim_idx, 1, area_idx, 2] = J
               
               unit_count[stim_idx, 1, area_idx, 0] = len(ophys_values)
               unit_count[stim_idx, 1, area_idx, 1] = len(ephys_values)
    
    metric = 'pref_tf_'
    stim = 'dg'
    
    if check_stim(stim):
    
        support = np.arange(0,5.1,1.0)
        
        for area_idx, area in enumerate(areas):
            
            ophys_selection = df_ophys['sig_fraction_spont_' + stim] > 0.25
            ephys_selection = df_ephys['sig_fraction_spont_' + stim] > 0.25
        
            ophys_selection &= df_ophys[metric + 'multi_' + stim] == False
            ephys_selection &= df_ephys[metric + 'multi_' + stim] == False
        
            ophys_selection &= df_ophys.ecephys_structure_acronym == area
            ephys_selection &= df_ephys.ecephys_structure_acronym == area
           
            ophys_values = np.ceil(np.log2(df_ophys[ophys_selection][metric + stim]))
            ephys_values = np.ceil(np.log2(df_ephys[ephys_selection][metric + stim]))
        
            S, W, J = compare_distributions(ophys_values, ephys_values, support)
            
            distances[0, 2, area_idx, 0] = S
            distances[0, 2, area_idx, 1] = W
            distances[0, 2, area_idx, 2] = J
            
            unit_count[stim_idx, 2, area_idx, 0] = len(ophys_values)
            unit_count[stim_idx, 2, area_idx, 1] = len(ephys_values)
    
    metric = 'pref_sf_'
    stim = 'sg'
    
    if check_stim(stim):
    
        support = np.arange(1.0,6.1,1.0)
    
        for area_idx, area in enumerate(areas):
            
            ophys_selection = df_ophys['sig_fraction_spont_' + stim] > 0.25
            ephys_selection = df_ephys['sig_fraction_spont_' + stim] > 0.25
        
            ophys_selection &= df_ophys[metric + 'multi_' + stim] == False
            ephys_selection &= df_ephys[metric + 'multi_' + stim] == False
        
            ophys_selection &= df_ophys.ecephys_structure_acronym == area
            ephys_selection &= df_ephys.ecephys_structure_acronym == area
        
            ophys_values = np.log2(df_ophys[ophys_selection][metric + stim] * 100)
            ephys_values = np.log2(df_ephys[ephys_selection][metric + stim] * 100)
        
            S, W, J = compare_distributions(ophys_values, ephys_values, support)
            
            distances[1, 2, area_idx, 0] = S
            distances[1, 2, area_idx, 1] = W
            distances[1, 2, area_idx, 2] = J
            
            unit_count[stim_idx, 2, area_idx, 0] = len(ophys_values)
            unit_count[stim_idx, 2, area_idx, 1] = len(ephys_values)
        
    return distances, unit_count

# %% apply forward model
    
new_df_ophys, df_ephys_fm = apply_forward_model(df_ophys, df_ephys)

# %% Basedline distances

all_distances = np.zeros((5, 4, 3, 5, 3)) # filtering, stimuli, metrics, areas, distance types
all_counts = np.zeros((5, 4, 3, 5, 2)) # filtering, stimuli, metrics, areas, modality

all_distances_fm = np.zeros((5, 4, 3, 5, 3)) # filtering, stimuli, metrics, areas, distance types
all_counts_fm = np.zeros((5, 4, 3, 5, 2)) # filtering, stimuli, metrics, areas, modality

# %%
all_distances[0,:,:,:,:], all_counts[0, :,:,:,:] = distance_summary(df_ophys, df_ephys)

all_distances_fm[0,:,:,:,:], all_counts_fm[0,:,:,:,:] = distance_summary(df_ophys, df_ephys_fm)

# %% Effect of layer matching

new_df_ophys, new_df_ephys = apply_layer_matching(df_ophys, df_ephys)
all_distances[1,:,:,:,:], all_counts[1,:,:,:,:] = distance_summary(new_df_ophys, new_df_ephys)

new_df_ophys, new_df_ephys = apply_layer_matching(df_ophys, df_ephys_fm)
all_distances_fm[1,:,:,:,:], all_counts_fm[1,:,:,:,:] = distance_summary(new_df_ophys, new_df_ephys)

# %% Effect of behavior matching

stimuli = ['dg', 'sg', 'ns']

for stim_idx, stim in enumerate(stimuli):
    
    new_df_ophys, new_df_ephys = apply_behavior_matching(df_ophys, df_ephys, stim)

    result, counts = distance_summary(new_df_ophys, new_df_ephys)

    all_distances[2,stim_idx,:,:,:] = result[stim_idx, :,:,:] 
    all_counts[2,stim_idx,:,:,:] = counts[stim_idx, :,:,:]
    
    
for stim_idx, stim in enumerate(stimuli):
    
    new_df_ophys, new_df_ephys = apply_behavior_matching(df_ophys, df_ephys_fm, stim)

    result, counts = distance_summary(new_df_ophys, new_df_ephys)

    all_distances_fm[2,stim_idx,:,:,:] = result[stim_idx, :,:,:] 
    all_counts_fm[2,stim_idx,:,:,:] = counts[stim_idx, :,:,:]

# %% Effect of event rate sub-selection

stimuli = ['dg', 'sg', 'ns', 'nm3']

for stim_idx, stim in enumerate(stimuli):

    new_df_ophys, new_df_ephys = apply_event_rate_selection(df_ophys, df_ephys, stim)
    
    distances, counts = distance_summary(new_df_ophys, new_df_ephys, stimulus_filter = stim)
     
    all_distances[3,stim_idx,:,:,:] = distances[stim_idx, :, :, :]
    all_counts[3,stim_idx,:,:,:] = counts[stim_idx, :, :, :]
    
    
for stim_idx, stim in enumerate(stimuli):

    new_df_ophys, new_df_ephys = apply_event_rate_selection(df_ophys, df_ephys_fm, stim)
    
    distances, counts = distance_summary(new_df_ophys, new_df_ephys, stimulus_filter = stim)
     
    all_distances_fm[3,stim_idx,:,:,:] = distances[stim_idx, :, :, :]
    all_counts_fm[3,stim_idx,:,:,:] = counts[stim_idx, :, :, :]

# %% Effect of ISI sub-selection

new_df_ophys, new_df_ephys = apply_ISI_selection(df_ophys, df_ephys)

all_distances[4,:,:,:,:], all_counts[4,:,:,:,:] = distance_summary(new_df_ophys, new_df_ephys)

new_df_ophys, new_df_ephys = apply_ISI_selection(df_ophys, df_ephys_fm)

all_distances_fm[4,:,:,:,:], all_counts_fm[4,:,:,:,:] = distance_summary(new_df_ophys, new_df_ephys)

# %%

from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

def add_boxes(axis, height, offset=0, color='gray'):
    
    collection = []
    
    for i in range(3):
        rect = Rectangle((0+i*2-0.5+offset, -1), 1, height)
        collection.append(rect)
    
    # Create patch collection with specified colour/alpha
    pc = PatchCollection(collection, facecolor=color, alpha=0.1,
                         edgecolor=None)
    
    # Add collection to axes
    axis.add_collection(pc)


plt.figure(1)
plt.clf()

stimuli = ['dg', 'sg', 'ns', 'nm3']
areas = ['VISp', 'VISl', 'VISal', 'VISpm', 'VISam']

common_names = ['V1', 'LM', 'AL', 'PM', 'AM']

for stim_idx, stim in enumerate(stimuli):
    
    if stim_idx < 3:
        metric_mask = np.array([0, 1, 2, 3, 4]) # baseline, layer, behavior, event, ISI
    else:
        metric_mask =np.array([0, 1, 3, 4])
        
    if stim_idx < 2:
        metrics = ['responsiveness', 'selectivity', 'preference']
    else:
        metrics = ['responsiveness', 'selectivity']
    
    for metric_idx, metric in enumerate(metrics):
        
        plt.subplot(4, 3, stim_idx*3 + metric_idx + 1)
        
        M = np.zeros((2, len(metric_mask), len(areas)))
        for area_idx, area in enumerate(areas):
            
            D = all_distances[metric_mask,stim_idx, metric_idx, area_idx, 2]
            x = metric_mask + area_idx*0.04 - 0.12
            
            for i in range(len(metric_mask)):
                M[0, i, area_idx] = D[i]
            
            plt.plot(x, D, '.', color=get_color_palette(area, name='seaborn'), alpha=0.7) # - D[0], '.')
            
            D = all_distances_fm[metric_mask,stim_idx, metric_idx, area_idx, 2]
            plt.plot(x + 6, D, '.', color=get_color_palette(area, name='seaborn'), alpha=0.7) # - D[0], '.')

            for i in range(len(metric_mask)):
                M[1, i, area_idx] = D[i]
             
        
        for i in range(len(metric_mask)):
            for j in range(2):
                X = metric_mask[i] - 0.4
                if j == 1:
                    X += 6
                Y = np.mean(M[j, i,:])
                plt.plot([X,X+0.8],[Y,Y],'-k',linewidth=2.0,
                         alpha=0.3)
        
        baseline = np.mean(M[0,0,:])
        plt.plot([-0.5,10.5],[baseline,baseline],'-k',
                 alpha=0.1)
                
            
        if metric_idx == 0:
            plt.ylabel(stim)
        
        if stim_idx == 0:
            plt.title(metric)
            
        plt.xlim([-0.5,10.5])
        
        axis = plt.gca()
        [axis.spines[loc].set_visible(False) for loc in ['top', 'right']] 
        
        plt.plot([-0.5,10.5],[0,0],'--k', alpha=0.3)
        add_boxes(axis, 2)
        add_boxes(axis, 2, 6, 'orange')
        
        if metric_idx == 0:
        
            plt.ylim([-0.05,0.4])
        elif metric_idx == 1:
            plt.ylim([-0.1,0.8])
        else:
            plt.ylim([-0.05,0.4])
        
        if stim_idx == 3:
            plt.xticks(ticks = np.arange(11),
                   labels=['Baseline', 'Layers','Behavior', 'Events', 'ISI', ' ',
                           'Baseline', 'Layers','Behavior', 'Events', 'ISI'],
                   rotation=45)
        else:
            axis.get_xaxis().set_visible(False)
            axis.spines['bottom'].set_visible(False)
            
    
plt.subplot(4, 3, 9)

for area_idx, area in enumerate(areas):
    
    plt.plot(area_idx, 0, '.', color=get_color_palette(area,
                                                       name ='seaborn'),
             alpha=0.8)
    
plt.ylim([1,2])
plt.legend(common_names, loc='center left')
plt.axis('off')

# %%
