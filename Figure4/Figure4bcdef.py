import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common.plotting_functions import *
from common.distance_metrics import *

# %%

df_ephys = pd.read_csv('data/df_ephys_200620.csv', index_col=0, low_memory=False)

df_isi = pd.read_csv('data/LogISI_003.csv', index_col=0)

# %%
genotypes = ['wt/wt',
             'Cux2-CreERT2;Camk2a-tTA;Ai93',
             'Slc17a7-IRES2-Cre;Camk2a-tTA;Ai93', 
             'Vip-IRES-Cre;Ai148',
             'Sst-IRES-Cre;Ai148']

for gt in genotypes:
    print(gt)
    print(len(df_ephys[df_ephys.genotype == gt].specimen_id.unique()))

# %%
# total number of cells

areas = ['VISp', 'VISl', 'VISal', 'VISpm', 'VISam']
common_names = ['V1','LM', 'AL', 'PM', 'AM']

plt.figure(2191, figsize=(9,5))
plt.clf()

matplotlib.rcParams.update({'font.size': 14})

wt_counts = []
gcamp_counts = []

for genotype_idx, genotype in enumerate(genotypes):
    
    for area_idx, area in enumerate(areas):
        
        sub_df = df_ephys[(df_ephys.genotype == genotype) & 
                          (df_ephys.ecephys_structure_acronym == area)]
        
        counts = sub_df.groupby('specimen_id')['waveform_amplitude'].count()
        
        x = np.ones((counts.size)) * (area_idx + genotype_idx*0.1)
        plt.plot(x, counts, '.', color=color_mapping[genotype],
                 alpha=0.25)
        
        M = np.mean(counts)
        E = np.std(counts)
        
        if genotype_idx == 0:
            wt_counts.append(M)
        else:
            gcamp_counts.append(M)
        
        plt.plot(area_idx + genotype_idx*0.1, M, '.', color=color_mapping[genotype], 
                 markersize=10)
        plt.errorbar(area_idx + genotype_idx*0.1, M, E,
                     color=color_mapping[genotype],
                     linewidth=5,
                     alpha=0.5)
        
plt.xticks(ticks=np.arange(5) + 0.2, labels=common_names)
plt.ylabel('Unit count')
axis = plt.gca()
[axis.spines[loc].set_visible(False) for loc in ['top', 'right']] 

print('WT: ' + str(np.mean(wt_counts)) + ' +/- ' + str(np.std(wt_counts)))
print('GCAMP: ' + str(np.mean(gcamp_counts)) + ' +/- ' + str(np.std(gcamp_counts)))

# %%

# firing rate

from scipy.ndimage.filters import gaussian_filter1d

plt.figure(2192, figsize=(18,3))
plt.clf()

filter_width = 4

for area_idx, area in enumerate(areas):

    for genotype_idx, genotype in enumerate(genotypes):
    

        if genotype == 'Cux2-CreERT2;Camk2a-tTA;Ai93' and area == 'VISpm':
            pass
        else:
        
            plt.subplot(1,5,area_idx+1)
            
            sub_df = df_ephys[(df_ephys.genotype == genotype) & 
                              (df_ephys.ecephys_structure_acronym == area)]
        
            
            firing_rate = np.log10(sub_df['firing_rate'])
            
            h, b, = np.histogram(firing_rate, 
                                 bins=np.linspace(-2,2.5,150), 
                                 density=True)
            
            plt.bar(b[:-1], gaussian_filter1d(h,filter_width), 
                    width =np.mean(np.diff(b)),
                    color=color_mapping[genotype], 
                    alpha=0.2)
            
            plt.plot(b[:-1], gaussian_filter1d(h,filter_width), 
                     color=color_mapping[genotype],
                     linewidth=2.0)
            
            if genotype_idx == 0:
                baseline = firing_rate
                plt.title(common_names[area_idx])
            else:
                S, W, J = compare_distributions(baseline,
                                      firing_rate,
                                      np.arange(-2,3.0,0.5))
                
                plt.text(1.5, genotype_idx*0.2, 
                         str(np.around(J,3)), 
                         color=color_mapping[genotype])
    
            
            axis = plt.gca()
            axis.get_yaxis().set_visible(False)
            [axis.spines[loc].set_visible(False) for loc in ['right', 'top', 'left']]  
            
            if area_idx == 0:
                plt.xlabel('Firing rate (Hz)')
        
            plt.xticks(ticks = np.arange(-2,3),
                       labels=['0.01','0.1','1','10','100'])
            
        
        
plt.tight_layout()

# %%

# bursting

from scipy.ndimage.filters import gaussian_filter1d

plt.figure(2199, figsize=(15,3))
plt.clf()

genotypes_isi = ['wt','Vip-Ai148','Sst-Ai148' ,'Cux2-Ai93' ,'Slc17a-Ai93' ]

filter_width = 2
min_value = 0.05

for genotype_idx, genotype in enumerate(genotypes_isi):
    
    for area_idx, area in enumerate(areas):
        
        if genotype == 'Cux2-Ai93' and area == 'VISpm':
            pass
        else:
        
            plt.subplot(1,5,area_idx+1)
            
            L = np.sum((df_isi.genotype == genotype) & 
                       (df_isi.ecephys_structure_acronym == area))
            
            sub_df = df_isi[(df_isi.genotype == genotype) & 
                              (df_isi.ecephys_structure_acronym == area) &
                              (df_isi.burst_fraction > min_value)]
        
            
            burst_fraction = sub_df['burst_fraction']
            
            h, b, = np.histogram(burst_fraction, 
                                 bins=np.linspace(min_value, 1,25), 
                                 density=False)
            h = h / L
            
            plt.bar(b[:-1], gaussian_filter1d(h,filter_width), 
                    width =np.mean(np.diff(b)),
                    color=color_mapping[genotype], 
                    alpha=0.2)
            
            plt.plot(b[:-1], gaussian_filter1d(h,filter_width), 
                     color=color_mapping[genotype],
                     linewidth=2.0)
            
            plt.title(common_names[area_idx])
            
            M = np.median(burst_fraction)
            plt.plot([M],[0.1],'.',color=color_mapping[genotype])
            plt.plot([0.3,0.3],[0,0.12],'k',alpha=0.1)
            
            axis = plt.gca()
            axis.get_yaxis().set_visible(False)
            [axis.spines[loc].set_visible(False) for loc in ['right', 'top', 'left']]  
            
            if area_idx == 0:
                plt.xlabel('Burst fraction')
        
            if genotype_idx == 0:
                baseline = burst_fraction
                plt.title(common_names[area_idx])
            else:
                S, W, J = compare_distributions(baseline,
                                      burst_fraction,
                                      np.arange(0, 1, 0.1))
                
                plt.text(0.75, genotype_idx*0.02 + 0.01, 
                         str(np.around(J,3)), 
                         color=color_mapping[genotype])
                
            plt.ylim([0,0.12])
            plt.xlim([min_value,1.0])
            plt.xticks(ticks=[0.05, 0.5, 1.0])
        
plt.tight_layout()

# %%

# selectivity

plt.figure(num=2193, figsize=(15,3))
plt.clf()

bins = np.arange(0,1.1,0.1)

selectivities = []

for genotype_idx, genotype in enumerate(genotypes):
    
    axes, sel = plot_selectivity_comparison('lifetime_sparseness_dg', 
                                df_ephys[df_ephys.genotype == genotype],
                                color=color_mapping[genotype],
                                num_points=100,
                                filter_window=5)
    
    if genotype_idx == 0:
        baseline_selectivity = sel
    else:
        selectivity = sel
    
        for area_idx, area in enumerate(areas):
            
            if genotype == 'Cux2-CreERT2;Camk2a-tTA;Ai93' and area == 'VISpm':
                pass
            
            else:
            
                plt.subplot(1,5,area_idx+1)
                
                bins = np.arange(0,1.1,0.1)
                S, W, J = compare_distributions(baseline_selectivity[area_idx],
                                          selectivity[area_idx],
                                          bins)
    
                plt.text(0.75,genotype_idx*0.6, str(np.around(J,2)), fontsize=10,
                         color=color_mapping[genotype])
        
plt.tight_layout()

# %%

# responsiveness

plt.figure(num=2193, figsize=(15,4))
plt.clf()

for genotype_idx, genotype in enumerate(genotypes):
    
    axis, resp = plot_responsiveness_comparison('sig_fraction_spont_dg', 
                                df_ephys[(df_ephys.genotype == genotype) &
                                         (df_ephys.firing_rate_dg > 0.0)],
                                color=color_mapping[genotype],
                                offset=(genotype_idx-2)*0.12,
                                barwidth=0.1)


    if genotype_idx == 0:
        baseline_responsiveness = resp
        
        for area_idx, area in enumerate(areas):
            M = np.mean(baseline_responsiveness[area_idx] > 0.25)

            plt.plot([area_idx-0.3,area_idx+0.3],[M,M],'--k',
                     linewidth=1, alpha=0.5)
    else:
        responsiveness = resp
    
        for area_idx, area in enumerate(areas):
            
            if genotype == 'Cux2-CreERT2;Camk2a-tTA;Ai93' and area == 'VISpm':
                pass
            else:
                
                bins = np.arange(0,1.1,0.1)
                S, W, J = compare_distributions(baseline_responsiveness[area_idx],
                                          responsiveness[area_idx],
                                          bins)
                
                plt.text(area_idx +( genotype_idx-2.5) * 0.2, 
                         0.9, 
                         str(np.around(J,3)), color=color_mapping[genotype],
                         fontsize=10)
    
            
plt.tight_layout()
