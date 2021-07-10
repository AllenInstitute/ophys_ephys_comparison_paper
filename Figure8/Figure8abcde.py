import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%

f1 = np.load('data/ephys_classes.npz', allow_pickle=True)

f2 = np.load('data/ophys_classes.npz', allow_pickle=True)

colors = ['dimgrey',
          'lightblue',
          'lightslategrey',
          'slategrey',
          'orange',
          'dimgrey',
          'dimgrey',
          'dimgrey',
          'skyblue',
          'royalblue',
          'dimgrey',
          'mediumvioletred',
          'mediumorchid',
          'plum',
          'orchid',
          'mediumaquamarine']


labels = ['None',
          'DG',
          'SG',
          'DG-SG',
          'NS',
          'DG-NS',
          'SG-NS',
          'DG-SG-NS',
          'NM',
          'DG-NM',
          'SG-NM',
          'DG-SG-NM',
          'NS-NM',
          'DG-NS-NM',
          'SG-NS-NM',
          'DG-SG-NS-NM']

# %%

thresholds = [float(a) for a in list(f1.keys())]

import matplotlib

matplotlib.rcParams.update({'font.size': 14})

# %%

plt.figure(19111)
plt.clf()

plt.subplot(2,2,1)

from scipy.spatial.distance import jensenshannon

b = np.mean(f2['0.25'], 0)

d = []

for t in thresholds:
    M = np.mean(f1[str(t)],0)
    #plt.plot(M,'.')
    d.append(jensenshannon(b, M))
    
    
D = f1['0.25'] * 100
for i in range(16):
    plt.bar(i, np.mean(D[:,i]), color=colors[i], alpha=0.7)
    x = np.ones((100,)) * i
    plt.plot(x,D[:,i],'.', color=colors[i])
    
plt.xticks(ticks=np.arange(16),labels=labels,rotation=90)
plt.ylabel('Percent of neurons')
axis = plt.gca()
[axis.spines[loc].set_visible(False) for loc in ['top', 'right']] 
plt.ylim([0,50])
plt.title('Ephys (threshold = 25%)')

plt.subplot(2,2,2)
    
D = f1['0.4'] * 100
for i in range(16):
    plt.bar(i, np.mean(D[:,i]), color=colors[i], alpha=0.7)
    x = np.ones((100,)) * i
    plt.plot(x,D[:,i],'.', color=colors[i])
    
plt.xticks(ticks=np.arange(16),labels=labels,rotation=90)
plt.ylabel('Percent of neurons')
axis = plt.gca()
plt.ylim([0,50])
[axis.spines[loc].set_visible(False) for loc in ['top', 'right']] 
    
plt.title('Ephys (threshold = 40%)')
    
plt.subplot(2,2,3)

D = f2['0.25'] * 100

for i in range(16):
    plt.bar(i, np.mean(D[:,i]), color=colors[i], alpha=0.7)
    x = np.ones((100,)) * i
    plt.plot(x,D[:,i],'.', color=colors[i])
    
plt.title('Ophys (threshold = 25%)')
plt.xticks(ticks=np.arange(16),labels=labels,rotation=90)
plt.ylabel('Percent of neurons')
axis = plt.gca()
plt.ylim([0,50])
[axis.spines[loc].set_visible(False) for loc in ['top', 'right']] 

plt.subplot(2,2,4)

plt.plot(thresholds, d)
plt.plot(thresholds, d, '.', color='tab:blue')
plt.xlabel('Threshold')
plt.ylabel('J-S Distance')
plt.plot(thresholds[5], d[5], 'ok', markersize=10, fillstyle='none') #, linewidth=1.0)
plt.plot(thresholds[8], d[8], 'ok', markersize=10, fillstyle='none') #=None, linewidth=1.0)
plt.ylim([0,0.8])

axis = plt.gca()
[axis.spines[loc].set_visible(False) for loc in ['top', 'right']] 
    
plt.tight_layout()
    
# %%

ephys_class_means = np.load('data/ephys_means_all.npz', allow_pickle=True)

ophys_class_means = np.load('data/ophys_means_all.npz', allow_pickle=True)

# %%

keys = list(ophys_class_means)

plt.figure(11123)
plt.clf()

from matplotlib.colors import LinearSegmentedColormap

ephys_ind = 24
ophys_ind = 77
    
def get_class_index(means, threshold=0.25):
    if threshold is None:
        threshold = get_threshold_from_means(means)
    responsive = means>threshold
    basis = np.array([1,2,4,8])  # dg, sg, ns, nm
    class_index = np.sum(responsive*basis, axis=1)
    return class_index

def get_colormap(threshold):
    return LinearSegmentedColormap.from_list('cmap', [(0.0, 'dodgerblue'), 
                                                      (threshold, 'white'),
                                                      (1.0, 'red')])

def plot_heatmap(index,class_means, threshold):
    
    key = keys[index]
    class_inds = get_class_index(class_means[key], threshold=threshold)
    order = np.argsort(class_inds)
    class_labels = np.array(labels)[np.sort(class_inds)]
    plt.imshow(class_means[key][order,:], cmap=get_colormap(threshold), vmin=0, vmax=1.0)
    plt.colorbar()
    plt.yticks(ticks=np.arange(len(class_labels)), labels=class_labels)

for i, t in enumerate([0.0,0.1,0.2,0.3,0.4]):

    plt.subplot(1,5,i+1)
    plot_heatmap(ephys_ind, ephys_class_means, t)
    plt.axis('off')

plt.figure(11124)
plt.clf()


plt.subplot(1,2,1)
plot_heatmap(ephys_ind, ephys_class_means, 0.4)

plt.subplot(1,2,2)
plot_heatmap(ophys_ind, ophys_class_means, 0.25)

    
# %%
    
df = pd.read_hdf('data/ephys_tsne.h5')

df_o = pd.read_hdf('data/ophys_tsne.h5')

clusters = df['sample_' + keys[ephys_ind]].values

clusters_o = df_o['sample_' + keys[ephys_ind]].values

selection= 13

plt.figure(191)
plt.clf()

plt.subplot(1,2,1)
plt.scatter(df.y, df.x, c=(80-clusters), s=2, cmap='gray', vmin=0, vmax=80) #,cmap='gist_stern')
plt.scatter(df.y[clusters == selection], df.x[clusters == selection], c='brown', s=2) #,cmap='gist_stern')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')

plt.subplot(1,2,2)

means = ephys_class_means[keys[ephys_ind]][selection,:]

plt.bar(np.arange(4), means)
plt.xlim([-2,5])
plt.plot([-2,5],[0.25,0.25],'--k')
plt.plot([-2,5],[0.25,0.25],'--k')
plt.ylim([0,1])
plt.xticks(ticks=np.arange(4), labels=['DG','SG', 'NS', 'NM'])
plt.ylabel('Mean responsive fraction')

class_inds = get_class_index(ephys_class_means[keys[ephys_ind]], threshold=0.25)

class_labels = class_inds[clusters]

L, counts = np.unique(class_labels, return_counts=True)

fractions = counts / np.sum(counts)

for i in range(len(L)):
    print(np.array(labels)[L[i]])
    print(fractions[i])
