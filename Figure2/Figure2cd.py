import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%


from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
from allensdk.brain_observatory.ecephys.ecephys_session import EcephysSession 

from allensdk.brain_observatory.ecephys.stimulus_analysis import DriftingGratings

cache_directory = "/mnt/nvme0/ecephys_cache_dir"

manifest_path = os.path.join(cache_directory, "manifest.json")

cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)

# %%

df_ophys = pd.read_csv('data/df_ophys_200620.csv', index_col=0, low_memory=False)
df_ephys = pd.read_csv('data/df_ephys_200620.csv', index_col=0, low_memory=False)

# %%

pref_ori = 90
pref_tf = 1

candidate_ephys = (df_ephys.ecephys_structure_acronym == 'VISp') & \
                (df_ephys.lifetime_sparseness_dg < 0.6) & \
                (df_ephys.lifetime_sparseness_dg > 0.1) & \
                (df_ephys.pref_ori_dg == pref_ori) & \
                (df_ephys.pref_tf_dg == pref_tf) & \
                (df_ephys.sig_fraction_spont_dg < 0.8) & \
                (df_ephys.sig_fraction_spont_dg > 0.6) & \
                (df_ephys.presence_ratio == 0.99) & \
                (df_ephys.firing_rate_dg > 3.0)
            
                    
print(np.sum(candidate_ephys))

candidate_ophys = (df_ophys.ecephys_structure_acronym == 'VISp') & \
                 (df_ophys.lifetime_sparseness_dg < 1.0) & \
                 (df_ophys.lifetime_sparseness_dg > 0.6) & \
                 (df_ophys.pref_ori_dg == pref_ori) & \
                 (df_ophys.pref_tf_dg == pref_tf) & \
                 (df_ophys.sig_fraction_spont_dg < 0.6) & \
                 (df_ophys.sig_fraction_spont_dg > 0.4)
                
print(np.sum(candidate_ophys))

# %%

def get_sig_fraction(dg, unit_id, n_samples=1000):

    n_baseline = 1000
    n_samples = 1000

    def baseline_shift_range(arr, amount):
        return np.apply_along_axis(lambda x : x + (np.random.rand(1))*amount,1,arr) 

    bin_edges = np.arange(0 + dg._bin_offset, 
                          dg._trial_duration + dg._bin_offset,
                          dg._default_bin_resolution)

    spont_start = dg.stim_table_spontaneous.iloc[0].start_time
    spont_end = dg.stim_table_spontaneous.iloc[0].stop_time
    spont_trial_inds = np.ones((n_samples,))*dg.stim_table_spontaneous.index.values[0]

    baseline_shift_callback = lambda arr: baseline_shift_range(arr, spont_end-spont_start)
    
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        responses_spont = dg.ecephys_session.presentationwise_spike_counts(bin_edges, 
                                                      stimulus_presentation_ids = spont_trial_inds,
                                                      unit_ids = [unit_id],
                                                      use_amplitudes = dg._use_amplitudes,
                                                      time_domain_callback=baseline_shift_callback)
    
        responses = dg.ecephys_session.presentationwise_spike_counts(bin_edges, 
                                                      stimulus_presentation_ids = dg.stim_table.index.values,
                                                      unit_ids = [unit_id],
                                                      use_amplitudes = dg._use_amplitudes
                                                      )

        resp_mean = responses.sum(dim='time_relative_to_stimulus_onset')
        resp_spont_mean = responses_spont.sum(dim='time_relative_to_stimulus_onset')
    
        resp_for_unit = resp_mean.sel(unit_id=unit_id).data
        spont_for_unit = resp_spont_mean.sel(unit_id=unit_id).data
        
        mask = dg.stim_table.stimulus_condition_id == dg._get_preferred_condition(unit_id)
        
        sig_level_spont = np.quantile(spont_for_unit, 0.95)
        sig_fraction_spont = np.sum(resp_for_unit[mask] > sig_level_spont) / np.sum(mask) 

    return dg, sig_level_spont, sig_fraction_spont
        

# %%

ephys_id = df_ephys[candidate_ephys].index.values[2]
ophys_id = df_ophys[candidate_ophys].index.values[3]

ecephys_session_id = int(df_ephys.loc[ephys_id].ecephys_session_id)
ophys_nwb = os.path.join('data', os.path.basename(df_ophys.loc[ophys_id].nwb))

file_types = [False, True]

print(df_ophys.loc[ophys_id].lifetime_sparseness_dg)
print(df_ophys.loc[ophys_id].g_osi_dg)

print(df_ephys.loc[ephys_id].lifetime_sparseness_dg)
print(df_ephys.loc[ephys_id].g_osi_dg)

for IDX, file_type in enumerate(file_types):
       
    plt.figure(192+IDX)
    plt.clf()
    
    if file_type:
        bins = np.arange(0,2,0.033)
        unit_id = ophys_id
        session = EcephysSession.from_nwb_path(ophys_nwb, api_kwargs={
            "amplitude_cutoff_maximum": np.inf,
            "presence_ratio_minimum": -np.inf,
            "isi_violations_maximum": np.inf
        })
    else:
        bins = np.arange(0,2.0,0.001)
        unit_id = ephys_id
        session = cache.get_session_data(ecephys_session_id, 
            amplitude_cutoff_maximum = np.inf,
            presence_ratio_minimum = -np.inf,
            isi_violations_maximum = np.inf)
    
    dg = DriftingGratings(session, is_ophys_session=file_type)

    counts = session.presentationwise_spike_counts(
        bins,
        dg.stim_table.index.values,
        [unit_id],
        use_amplitudes=file_type)

    plt.figure(192+IDX)
    plt.clf()
    
    plt.figure(292+IDX)
    plt.clf()
    
    pref_cond = dg._get_preferred_condition(unit_id)
    
    ori_magnitudes = np.zeros((8,))
    tf_magnitudes = np.zeros((5,))
    
    for ori_idx, orival in enumerate(dg.orivals):
        
        for tf_idx, tfval in enumerate(dg.tfvals):
            
            plt.figure(192+IDX)
            plt.subplot2grid((6,9),(tf_idx,ori_idx), rowspan=1, colspan=1)
        
            presentation_ids = dg.stim_table[(dg.stim_table.temporal_frequency == tfval) &
                                             (dg.stim_table.orientation == orival)].index.values
        
            ok_trials = np.where(counts.stimulus_presentation_id.isin(presentation_ids))[0]
            
            sub_counts = counts.isel(stimulus_presentation_id=ok_trials)
        
            magnitudes = sub_counts.sum(dim='time_relative_to_stimulus_onset')
            
            ori_magnitudes[ori_idx] += np.sum(magnitudes)
            tf_magnitudes[tf_idx] += np.sum(magnitudes)
            
            for trial_idx in np.arange(len(ok_trials)):
                
                D = sub_counts.sel(stimulus_presentation_id=sub_counts.stimulus_presentation_id[trial_idx])
                
                spike_inds = np.where(np.squeeze(D.data))[0]
                spike_amplitudes = np.squeeze(D.data)[spike_inds]
                spike_times = D.time_relative_to_stimulus_onset[spike_inds]
                
                x = spike_times
                y = np.ones(x.shape) + trial_idx
                
                s = spike_amplitudes
                
                if file_type:
                    s = s*10
                
                plt.scatter(x,y,s,c='k', alpha=0.5)
                
            plt.axis('off')
            
            if dg.stim_table.loc[presentation_ids].stimulus_condition_id.unique()[0] == pref_cond:
                    
                plt.title('PREF')
                plt.figure(292+IDX)
                order = np.argsort(np.squeeze(magnitudes.data))
                
                plt.subplot(1,2,1)
                for trial_idx, trial in enumerate(order):
 
                    D = sub_counts.sel(stimulus_presentation_id=sub_counts.stimulus_presentation_id[trial])
                    
                    spike_inds = np.where(np.squeeze(D.data))[0]
                    spike_amplitudes = np.squeeze(D.data)[spike_inds]
                    spike_times = D.time_relative_to_stimulus_onset[spike_inds]
                    
                    x = spike_times
                    y = np.ones(x.shape) + trial_idx
                    
                    s = spike_amplitudes
                    
                    if file_type:
                        s = s*200
                    else:
                        s = s*10
                    
                    plt.scatter(x,y,s,c='k', alpha=0.5)
                    
                    if file_type:
                        plt.scatter([0],[1],s=[200], c='r')
                    
                plt.ylim([-1,16])
                    
                dg, level, fraction = get_sig_fraction(dg, unit_id)
                    
                plt.subplot(1,2,2)
                plt.barh(np.arange(len(magnitudes.data))+1, 
                         width=np.squeeze(magnitudes.data[order]), 
                         height=0.75)
                
                plt.title(fraction)
                plt.xlim([0,np.max(magnitudes.data) * 1.1])
                plt.plot([level,level],[0,16],'--r')
                plt.plot([0,np.max(magnitudes.data) * 1.1],[16-fraction*15, 16-fraction*15], '--b')
                plt.ylim([-1,16])

    plt.figure(192+IDX)
    plt.subplot2grid((6,9),(0,8), colspan=1, rowspan=5)
    
    plt.plot(tf_magnitudes, 5 -np.arange(len(tf_magnitudes)), '.b')
    plt.plot(tf_magnitudes, 5 -np.arange(len(tf_magnitudes)), '-b')
    plt.ylim([0.5,5.5])
    plt.xlim([0,np.max(tf_magnitudes)*1.1])
    
    plt.subplot2grid((6,9),(5,0), colspan=8, rowspan=1)
    
    plt.plot(ori_magnitudes, '.b')
    plt.plot(ori_magnitudes, '-b')
    plt.xlim([-0.5,7.5])
    plt.ylim([0,np.max(ori_magnitudes)*1.1])

