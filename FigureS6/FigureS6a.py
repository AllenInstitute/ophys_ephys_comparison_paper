import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common.plotting_functions import *
from common.distance_metrics import *

# %%

fname = 'data/noise_df_no6s.csv'

df = pd.read_csv(fname)
df = df.set_index(df.cell_specimen_ids.values.astype('int'))
df.index.name = 'unit_id'

df = df.groupby('unit_id').mean()

# %%

df_ophys = pd.read_csv('data/df_ophys_200620.csv', index_col=0, low_memory=False)

df = df.join(df_ophys, on='unit_id', how='inner', rsuffix='_orig')

# %%

# Params by Cre line

from scipy.ndimage.filters import gaussian_filter1d

plt.figure(num=1010, figsize=(5,5))
plt.clf()

for idx, genotype in enumerate(df.genotype.unique()):
    
    sub_df = df[df.genotype == genotype]
    
    plt.subplot(1,3,1)
    h,b = np.histogram(sub_df.sigmaest, bins=np.linspace(0,0.120,100), density=True)
    H = gaussian_filter1d(h,2)
    plt.bar(b[:-1], H/40, 
                width =np.mean(np.diff(b)),
                bottom =idx,
                color = color_mapping[genotype],
                alpha=0.2)
        
    plt.plot(b[:-1], H/40+idx, 
                 linewidth=2.0,
                 color = color_mapping[genotype])
    
    ml = np.argmax(H)
    plt.plot([b[ml],b[ml]],[idx,H[ml]/40+idx],'k',alpha=0.2)
    
    plt.yticks(ticks = np.arange(10), labels=df.genotype.unique())
    plt.ylim([-1,11])
    axis = plt.gca()
    [axis.spines[loc].set_visible(False) for loc in ['top', 'right']] 
    plt.title('sigma')
    
    plt.subplot(1,3,2)
    h,b = np.histogram(sub_df.tauest, bins=np.linspace(0,1.25,100), density=True)
    H = gaussian_filter1d(h,2)
    plt.bar(b[:-1], H/3.5, 
                width =np.mean(np.diff(b)),
                color = color_mapping[genotype],
                bottom=idx, 
                alpha=0.2)
        
    plt.plot(b[:-1], H/3.5 + idx, 
                 linewidth=2.0,
                 color = color_mapping[genotype],)
    
    ml = np.argmax(H)
    plt.plot([b[ml],b[ml]],[idx,H[ml]/3.5+idx],'k',alpha=0.2)
    plt.ylim([-1,11])
    axis = plt.gca()
    [axis.spines[loc].set_visible(False) for loc in ['top', 'right','left']] 
    plt.title('tau')
    
    plt.subplot(1,3,3)
    x = np.linspace(0,0.06,100)
    h,b = np.histogram(sub_df.aest, bins=x, density=True)
    H = gaussian_filter1d(h,2)
    plt.bar(b[:-1], H/50, 
                width =np.mean(np.diff(b)),
                bottom=idx,
                alpha=0.2,
                color = color_mapping[genotype])
    ml = np.argmax(H[30:])
    plt.plot([b[ml+30],b[ml+30]],[idx,H[ml+30]/50+idx],'k',alpha=0.2)
        
    plt.plot(b[:-1], H/50+idx, 
                 linewidth=2.0,
                 color = color_mapping[genotype])
    
    plt.ylim([-1,11])
    plt.title('A')
    
    axis = plt.gca()
    [axis.spines[loc].set_visible(False) for loc in ['top', 'right', 'left']] 
    
plt.tight_layout()

