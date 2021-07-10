import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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



def get_session_subselection(plot_on=False):
    
    df = pd.read_csv('/mnt/nvme0/ophys-ephys/running_speed_info.csv',
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