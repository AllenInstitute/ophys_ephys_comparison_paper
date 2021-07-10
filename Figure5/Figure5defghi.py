import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%

df_ophys = pd.read_csv('data/df_ophys_200620.csv', index_col=0, low_memory=False)

df_ephys = pd.read_csv('data/df_ephys_200620.csv', index_col=0, low_memory=False)

df_ephys_synth_ml = pd.read_csv('data/df_synth_ophys.csv', index_col=0)

df_ephys = df_ephys.join(df_ephys_synth_ml, on='unit_id', rsuffix='_ophys_ml', how='inner')

# %%

# compare responsiveness before/after

plt.figure(2104)
plt.clf()

stim = 'dg'

sub_df = df_ephys[(df_ephys['firing_rate_' + stim] > 0)]

a = (sub_df['sig_fraction_spont_' + stim] >= 0.25) 
hit_pre = np.sum(a) 

b = (sub_df['sig_fraction_spont_' + stim + '_ophys_ml'] >= 0.25) 
hit_post = np.sum(b) 
    
c = (sub_df['sig_fraction_spont_' + stim] >= 0.25) & \
    (sub_df['sig_fraction_spont_' + stim + '_ophys_ml'] < 0.25)
hit_change = np.sum(c) 
    
d = (sub_df['sig_fraction_spont_' + stim] < 0.25) 
miss_pre = np.sum(d)
                  
e = (sub_df['sig_fraction_spont_' + stim + '_ophys_ml'] < 0.25) 
miss_post = np.sum(e) 
    
f = (sub_df['sig_fraction_spont_' + stim] < 0.25) & \
    (sub_df['sig_fraction_spont_' + stim + '_ophys_ml'] >= 0.25)
    
miss_change = np.sum(f) 

# ///
plt.bar(0,
        miss_pre - miss_change,
        bottom = 0,
        width=1, color='lightsalmon')

plt.bar(0,
        miss_change,
        bottom = miss_pre - miss_change,
        width=1, color='salmon')

plt.bar(0,
        hit_change,
        bottom = 4000,
        width=1, color='springgreen')

plt.bar(0,
        hit_pre - hit_change,
        bottom = 4000 + hit_change,
        width=1, color='palegreen')

# ////////////

plt.bar(2,
        miss_post - hit_change,
        bottom = 0,
        width=1, color='lightsalmon')

plt.bar(2,
        hit_change,
        bottom = miss_post - hit_change,
        width=1, color='salmon')

plt.bar(2,
        miss_change,
        bottom = 4000,
        width=1, color='springgreen')

plt.bar(2,
        hit_post - miss_change,
        bottom = 4000 + miss_change,
        width=1, color='palegreen')


plt.xticks(ticks=[0, 2], labels=['pre', 'post'])

axis = plt.gca()
[axis.spines[loc].set_visible(False) for loc in ['top', 'right']] 

print((hit_change + miss_change) / len(sub_df))

# %%

ophys_color = 'green'
ephys_color = 'orange'

metric = 'sig_fraction_spont_'
stimuli = ['dg', 'sg', 'ns']

plt.figure(2000)
plt.clf()

bins = np.arange(0,1.1,0.1)

for stim_idx, stim in enumerate(stimuli):
    plt.subplot(1,3,stim_idx+1)

    axis, ophys_values = plot_responsiveness_comparison(metric + stim, 
                                                        df_ophys[df_ophys['firing_rate_' + stim] > 0.0], 
                                                        ophys_color, offset=-0.18)
    axis, ephys_values = plot_responsiveness_comparison(metric + stim + '_ophys_ml', df_ephys[df_ephys['firing_rate_' + stim] > 0.0],
                                                        ephys_color, offset=+0.18)
    
    plt.title(stim)
    
    for area_idx, area in enumerate(areas):
        
        bins = np.arange(0,1.1,0.1)
        S, W, J = compare_distributions(ophys_values[area_idx],
                                      ephys_values[area_idx],
                                      bins)

        plt.text(area_idx-0.2, np.mean(ephys_values[area_idx] > 0.25) + 0.05, str(np.around(J,2)), fontsize=12)
    

# %%

# compare preference before/after

plt.figure(2022)
plt.clf()

sub_df = df_ephys[(df_ephys.sig_fraction_spont_dg > 0.25)]

h, xedges, yedgs, im = plt.hist2d(sub_df['pref_tf_dg'],
                                  sub_df['pref_tf_dg_ophys_ml'], bins=([0.5,1.5,2.5,4.5,8.5,16]))
plt.clf()
plt.imshow(h, cmap='cividis')

a = sub_df['pref_tf_dg'] != sub_df['pref_tf_dg_ophys_ml']

change_df = sub_df[a]

print(1 - np.sum(np.diag(h)) / np.sum(h))

plt.xlabel('Original')
plt.ylabel('After forward model')
plt.colorbar()
plt.xticks(np.arange(5), [1,2,4,8,15])
plt.yticks(np.arange(5), [1,2,4,8,15])

plt.title(len(sub_df))

# %%

ophys_color = 'green'
ephys_color = 'orange'

metric = 'pref_tf_'
stim = 'dg'

plt.figure(1999)
plt.clf()

axis, ophys_values = plot_preference_comparison(metric + stim, 
                           df_ophys[(df_ophys['sig_fraction_spont_' + stim] > 0.25) *
                                    (df_ophys[metric + 'multi_' + stim] == False)],
                           ophys_color, 
                           offset=-0.18)
axis, ephys_values = plot_preference_comparison(metric + stim + '_ophys_ml', 
                           df_ephys[(df_ephys['sig_fraction_spont_' + stim + '_ophys_ml'] > 0.25) *
                                    (df_ephys[metric + 'multi_' + stim + '_ophys_ml'] == False)],
                           ephys_color, 
                           offset=+0.18)
    

bins = np.arange(0,5.1,1.0)

bins = np.arange(0,5.1,1.0)

for area_idx, area in enumerate(areas):
    

    S, W, J = compare_distributions(ophys_values[area_idx],
                                      ephys_values[area_idx],
                                      bins)

    plt.subplot(1,5,area_idx+1)
    plt.text(2, 0.5, 'D = ' + str(np.around(J,4)))
    
# %%

# compare selectivity before/after

plt.figure(219, figsize=(8,8))
plt.clf()

areas = ['VISp', 'VISl', 'VISal', 'VISpm', 'VISam']

sub_df = df_ephys[(df_ephys.sig_fraction_spont_dg > 0.25)]

plt.scatter(sub_df['lifetime_sparseness_dg'],sub_df['lifetime_sparseness_dg_ophys_ml'], s=1, c='darkgray')
plt.xlabel('Original')
plt.ylabel('After forward model')
plt.ylim([0,1])
plt.xlim([0,1])

plt.title(len(sub_df))

# %%

ophys_color = 'green'
ephys_color = 'orange'

metric = 'lifetime_sparseness_'
stimuli = ['dg', 'sg', 'ns', 'nm3']

bins = np.arange(0,1.1,0.1)

JS = []
JS_b =[]

for stim_idx, stim in enumerate(stimuli[0:1]):
    plt.figure(2001+stim_idx, figsize=(10,4))
    plt.clf()

    axis, ophys_values = plot_selectivity_comparison(metric + stim, 
                                df_ophys[df_ophys['sig_fraction_spont_' + stim] > 0.25], 
                                ophys_color)
    
    axis, ephys_values = plot_selectivity_comparison(metric + stim + '_ophys_ml', 
                                df_ephys[(df_ephys['sig_fraction_spont_' + stim + '_ophys_ml'] > 0.25)], 
                                ephys_color
                                )
    

    for area_idx, area in enumerate(areas):
        
        plt.subplot(1,5, area_idx+1)
        bins = np.arange(0,1.1,0.1)
        S, W, J, JbootA, JbootB = compare_distributions(ophys_values[area_idx],
                                      ephys_values[area_idx],
                                      bins, bootstrap=1)
        
        print(np.mean(JbootA > J))
        print(np.mean(JbootA))
        
        JS.append(J)
        JS_b.append(np.mean(JbootA))

        plt.text(0.1, 1, str(np.around(J,2)), fontsize=12)
    
print(JS)

plt.tight_layout()

