import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from scipy.stats import norm, pearsonr, linregress

# %%

gt = np.load('data/ground_truth_upsampled_dict.npy', allow_pickle=True)
nnd = np.load('data/NND_upsampled_dict.npy', allow_pickle=True)

noise_df = pd.read_hdf('data/noise_std_oephys.h5')
noise_df = noise_df.set_index(noise_df.tid)

# %%


cell_ids = [103467, 103928, 103444, 103406, 102959, 103438, 103927, 102958,
 103953, 103991, 103736, 103971, 103932, 102978, 103986, 103933, 103931,
 103985, 103947, 103950, 103477, 103988, 102969, 103992, 103979, 102956,
 103402, 103958, 103934, 102960, 103920, 102974]

# %%

def plot_corr(gt_vals, est_vals, plot_on =True):
    
    r, p = pearsonr(gt_vals, est_vals)
    
    baseline_rate = np.around(np.mean(est_vals),4)
    
    if plot_on:
        gt_plot = gt_vals + np.random.rand(len(gt_vals)) * 0.5
        plt.plot(gt_plot, est_vals, '.k', alpha=0.3)
        plt.text(5,baseline_rate, str(baseline_rate))
    
        res = linregress(gt_vals, est_vals)
        x = np.arange(6)
        plt.plot(x, res.intercept + res.slope*x, 'r', label='fitted line')
        plt.title(np.around(r,2))
    
    return r, baseline_rate

# %%
    
thresholds = np.arange(0, 10.1, 0.5)

correlations = np.zeros((len(cell_ids), len(thresholds)))
rates = np.zeros((len(cell_ids), len(thresholds)))
fp_rates = np.zeros((len(cell_ids), len(thresholds)))

for cell_idx, cell_id in enumerate(cell_ids):
    
    info = noise_df.loc[cell_id]
    
    t_gt = gt.item()[cell_id]
    
    total_inds = len(t_gt)
    
    bins = np.arange(0, total_inds, 15)
  
    gt_vals = np.zeros((bins.shape))
    
    nnd_vals = np.zeros((bins.shape))
    
    for thresh_idx, threshold in enumerate(thresholds):
        
        t_nnd = np.copy(nnd.item()[cell_id])
        
        t_nnd = t_nnd / np.max(t_nnd) * info.max_dff
        
        thresh = info.noise_std * threshold 

        gt_yes = 0
        gt_no = 0
        false_pos = 0
        true_neg = 0
        hit = 0
        miss = 0
        
        for i in range(len(bins)-1):
            gt_vals[i] = np.sum(t_gt[bins[i]:bins[i+1]])
            nnd_vals[i] = np.sum(t_nnd[bins[i]:bins[i+1]])
            if (nnd_vals[i] < thresh):
                nnd_vals[i] = 0
            
            if (gt_vals[i]) > 0:
                gt_yes += 1
                if (nnd_vals[i] > 0):
                    hit += 1
                else:
                    miss += 1
                    
            else:
                gt_no += 1
                if (nnd_vals[i] > 0):
                    false_pos += 1
                else:
                    true_neg += 1
                    
        r, rate = plot_corr(gt_vals, nnd_vals, False)
        
        correlations[cell_idx, thresh_idx] = r
        rates[cell_idx, thresh_idx] = rate
        fp_rates[cell_idx, thresh_idx] = false_pos / gt_no

# %%
        
cmap = matplotlib.cm.get_cmap('cividis')

plt.figure(1992)
plt.clf()

plt.subplot(1,3,1)

for i in range(len(cell_ids)):
    plt.plot(thresholds, correlations[i,:], c='k', alpha=0.1)
    plt.plot(thresholds, correlations[i,:], '.', c='k', alpha=0.1)
    
for i in range(len(thresholds)):    
    plt.plot(thresholds[i], np.nanmean(correlations[:,i]), '.', color=cmap(i / 11),
             markersize=20,
             markeredgecolor='k')
    
plt.title('correlation with ground truth')
plt.xlabel('Threshold level')
plt.ylim([0,0.85])

axis = plt.gca()
[axis.spines[loc].set_visible(False) for loc in ['top', 'right']] 

plt.subplot(1,3,2)
for i in range(len(cell_ids)):
    plt.plot(thresholds, rates[i,:] / rates[i,0], c='k', alpha=0.1)
    plt.plot(thresholds, rates[i,:] / rates[i,0],'.', c='k', alpha=0.1)
    
for i in range(len(thresholds)):    
    plt.plot(thresholds[i], np.nanmean(rates[:,i] / rates[:,0]), '.', color=cmap(i / 11),
             markersize=20,
             markeredgecolor='k')
    
plt.title('average response (relative)')

axis = plt.gca()
[axis.spines[loc].set_visible(False) for loc in ['top', 'right']] 

plt.subplot(1,3,3)
for i in range(len(cell_ids)):
    plt.plot(thresholds, fp_rates[i,:], c='k', alpha=0.1)
    plt.plot(thresholds, fp_rates[i,:], '.',c='k', alpha=0.1)
    
for i in range(len(thresholds)):    
    plt.plot(thresholds[i], np.nanmean(fp_rates[:,i]), '.', color=cmap(i / 11),
             markersize=20,
             markeredgecolor='k')

plt.title('false positive rate')

axis = plt.gca()
[axis.spines[loc].set_visible(False) for loc in ['top', 'right']] 
