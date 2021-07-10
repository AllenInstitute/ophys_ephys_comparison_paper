import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%


df_ophys = pd.read_csv('data/df_ophys_200620.csv', index_col=0, low_memory=False)
df_ephys = pd.read_csv('data/df_ephys_200620.csv', index_col=0, low_memory=False)

# %%

specimens = df_ephys.specimen_id.unique()

specimens = [s for s in specimens if s > 1000000]

sub_df = df_ephys[df_ephys.specimen_id.isin(specimens)]

for genotype in sub_df.genotype.unique():
    
    print(genotype)
    print(len(sub_df[sub_df.genotype==genotype].specimen_id.unique()))

# %%

# SOME GENERAL NUMBERS

# EPHYS

num_mice = len(df_ephys.specimen_id.unique())
num_experiments = num_mice
num_cells = len(df_ephys)
num_genotypes = len(df_ephys.genotype.unique())

print ('EPHYS:  \n' + 
       'Total mice: ' + str(num_mice) + '\n' + 
       'Total experiments: ' + str(num_experiments) + '\n' +
       'Total cells: ' + str(num_cells) + '\n' +
       'Total genotypes: ' + str(num_genotypes) + '\n'
       )

# OPHYS

num_mice = len(df_ophys.donor_name.unique())
num_experiments = len(df_ophys.container_id.unique())
num_cells = len(df_ophys)
num_genotypes = len(df_ophys.cre_line.unique())

print ('OPHYS:  \n' + 
       'Total mice: ' + str(num_mice) + '\n' + 
       'Total experiments: ' + str(num_experiments) + '\n' +
       'Total cells: ' + str(num_cells) + '\n' +
       'Total genotypes: ' + str(num_genotypes) 
       )


# %%
ephys_genotypes = df_ephys.genotype.unique()
ophys_genotypes = df_ophys.cre_line.unique()

plt.figure(num=191, figsize=(15,4))
plt.clf()

areas = ['VISp', 'VISl', 'VISal', 'VISpm', 'VISam']
common_names = ['V1','LM', 'AL', 'PM', 'AM']

for modality_idx, modality in enumerate(('ephys','ophys')):
    
    if modality == 'ephys':
        df = df_ephys
        xlim = 1100
    else:
        df = df_ophys
        xlim = 8500
    
    for area_idx, area in enumerate(areas):
        
        plt.subplot(2,5,area_idx+1+modality_idx*5)
        
        for layer_idx, layer in enumerate([2, 4, 5, 6]):
            
            offset = 0
            
            for genotype_idx, genotype in enumerate(color_mapping.keys()):
                
                sub_df = df[(df.ecephys_structure_acronym == area) &
                                  (df.cortical_layer == layer) &
                                  (df.genotype == genotype)]
                
                plt.barh(4-layer_idx, 
                         len(sub_df),
                         height=0.4,
                         left=offset,
                         color=color_mapping[genotype])
                
                offset += len(sub_df)
        
        plt.xlim([0,xlim])
        axis = plt.gca()
        [axis.spines[loc].set_visible(False) for loc in ['top', 'right']] 
        
        if modality_idx == 0:
            plt.title(common_names[area_idx], fontsize=14)
            
        if area_idx == 0:
            plt.yticks(ticks=np.arange(1,5),
                       labels=['L6','L5','L4','L2/3'],
                       fontsize=14)
        else:
            plt.yticks(ticks=[],labels=[])
            
        if modality == 'ephys':
            plt.xticks(ticks=[0,500,1000], fontsize=14)
        else:
            plt.xticks(ticks=[0,5000], fontsize=14)
            if area_idx == 0:
                plt.xlabel('Neuron count')
            
            