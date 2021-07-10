from scipy.ndimage.filters import gaussian_filter1d
import matplotlib.pyplot as plt
import numpy as np

ephys_color = 'gray'
ophys_color = 'green'
forward_model_color = 'gold'

areas = ['VISp', 'VISl', 'VISal', 'VISpm', 'VISam']
common_names = ['V1','LM', 'AL', 'PM', 'AM']

color_mapping = {'wt/wt' : 'black',
                 'Pvalb-IRES-Cre/wt;Ai32(RCL-ChR2(H134R)_EYFP)/wt' : '#F7B0F7',
                 'Sst-IRES-Cre/wt;Ai32(RCL-ChR2(H134R)_EYFP)/wt' : '#FFD445',
                 'Vip-IRES-Cre/wt;Ai32(RCL-ChR2(H134R)_EYFP)/wt' : '#FF9C45',
                 'Slc17a7-IRES2-Cre;Camk2a-tTA;Ai93' : '#5C5C5C',
                 'Cux2-CreERT2;Camk2a-tTA;Ai93' : '#A92E66',
                 'wt' : 'black',
                 'Vip-Ai148' : '#B49139',
                 'Sst-Ai148' : '#7B5217',
                 'Cux2-Ai93' : '#A92E66',
                 'Slc17a-Ai93' : '#5C5C5C',
                 'Vip-IRES-Cre;Ai148' : '#B49139',
                 'Sst-IRES-Cre;Ai148' : '#7B5217',
                 'Emx1-IRES-Cre' :  '#9F9F9F',
                 'Slc17a7-IRES2-Cre' :'#5C5C5C',
                 'Cux2-CreERT2' : '#A92E66',
                 'Scnn1a-Tg3-Cre' : '#4F63C2',
                 'Rbp4-Cre_KL100' : '#5CAD53',
                 'Rorb-IRES2-Cre' : '#7841BE',
                 'Nr5a1-Cre' : '#5BB0B0',
                 'Ntsr1-Cre_GN220' : '#FF3B39',
                 'Fezf2-CreER' : '#3A6604',
                 'Tlx3-Cre_PL56' : '#99B20D'}

genotype_shorthand = {'wt/wt' : 'Wild type',
                 'Pvalb-IRES-Cre/wt;Ai32(RCL-ChR2(H134R)_EYFP)/wt' : 'Pvalb-ChR2',
                 'Sst-IRES-Cre/wt;Ai32(RCL-ChR2(H134R)_EYFP)/wt' : 'Sst-ChR2',
                 'Vip-IRES-Cre/wt;Ai32(RCL-ChR2(H134R)_EYFP)/wt' : 'Vip-ChR2',
                 'Cux2-CreERT2;Camk2a-tTA;Ai93' : 'Cux2-GCaMP6f',
                 'Slc17a7-IRES2-Cre;Camk2a-tTA;Ai93' : 'Slc17a7-GCaMP6f',
                 'Vip-IRES-Cre;Ai148' : 'Vip-GCaMP6f',
                 'Sst-IRES-Cre;Ai148' : 'Sst-GCaMP6f',
                 'Scnn1a-Tg3-Cre' : 'Scnn1a-GCaMP6f',
                 'Rbp4-Cre_KL100' : 'Rbp4-GCaMP6f',
                 'Rorb-IRES2-Cre' : 'Rorb-GCaMP6f',
                 'Cux2-CreERT2' : 'Cux2-GCaMP6f',
                 'Emx1-IRES-Cre' :  'Emx1-GCaMP6sf',
                 'Nr5a1-Cre' : 'Nr5a1-GCaMP6f',
                 'Ntsr1-Cre_GN220' : 'Ntsr1-GCaMP6f',
                 'Slc17a7-IRES2-Cre' :'Slc17a7-GCaMP6f',
                 'Fezf2-CreER' : 'Fezf2-GCaMP6f',
                 'Tlx3-Cre_PL56' : 'Tlx3-GCaMP6f'}   

def smoothed_hist(data, bins, linecolor, fillcolor, linestyle='-', filter_window=5): 
    
    h, b, = np.histogram(data, 
                             bins=bins, 
                             density=True)
    
    if fillcolor is not None:
        plt.bar(b[:-1], gaussian_filter1d(h,filter_window), 
            width =np.mean(np.diff(b)),
            color=fillcolor, 
            alpha= 0.6)
    
    plt.plot(b[:-1], gaussian_filter1d(h,filter_window), 
             linestyle,
             color=linecolor,
             alpha=0.8,
             linewidth=2.0)
    
    
def cumulative_histogram(values, bins, color, style='-'):
    
    h, b = np.histogram(values, bins, density=True)
    
    M = np.mean(np.diff(bins))
    
    plt.plot(b[:-1] + M/2, np.cumsum(h)*M, style, color=color)
    plt.plot(b[:-1] + M/2, np.cumsum(h)*M, '.', color=color)
    axis = plt.gca()
    [axis.spines[loc].set_visible(False) for loc in ['top', 'right']] 

    return h

def plot_responsiveness_comparison(metric, df, color, offset=-0.2, threshold=0.25, barwidth=0.25, areas = ['VISp', 'VISl', 'VISal', 'VISpm', 'VISam']):
    
    """
    Compare responsive fraction for one stimulus type

    Parameters:
    ----------
    metric : str (e.g., 'sig_fraction_spont_dg')
    df : pd.DataFrame
    color : matplotlib color
    offset : float (offset of bars)
    threshold : float (fraction of significant trials)

    Returns:
    ---------
    axis : current axis
    values : list of np.ndarrays of responsive fractions
    
    """

    values = []
    all_values = []

    for area_idx, area in enumerate(areas):
        
        sub_df = df[(df.ecephys_structure_acronym == area)]
        values.append(np.sum(sub_df[metric] > threshold) / len(sub_df))
        all_values.append(sub_df[metric].values)
  
    plt.bar(np.arange(len(areas))+offset, values, width=barwidth, color=color, alpha=0.7)
    plt.ylim([0,1])
    plt.grid('on',axis='y')

    plt.xticks(ticks=np.arange(len(areas)), labels=common_names)
    plt.ylabel('Responsive fraction')
    
    axis = plt.gca()
    [axis.spines[loc].set_visible(False) for loc in ['top', 'right']] 
    
    return axis, all_values
    
    
def plot_selectivity_comparison(metric, df, color, filter_window=4, num_points=150, areas=['VISp', 'VISl', 'VISal', 'VISpm', 'VISam']):

    """
    Compare distribution of a selectivity metric across areas

    Parameters:
    ----------
    metric : str (e.g., 'lifetime_sparseness_dg')
    df : pd.DataFrame
    color : matplotlib color
    filter_window : int (width of Gaussian filter window)
    num_points : int (number of points for histogram)

    Returns:
    ---------
    axes : list of axes for all plots
    values : list of np.ndarrays of values in distribution
    
    """
    
    axes = []
    values = []
    
    for area_idx, area in enumerate(areas):
        
        if len(areas) > 1:
            plt.subplot(1,len(areas),area_idx+1)
        
        sub_df = df[(df.ecephys_structure_acronym == area)]
            
        h, b, = np.histogram(sub_df[metric], 
                             bins=np.linspace(0,1,num_points), 
                             density=True)
        
        plt.bar(b[:-1], gaussian_filter1d(h,filter_window), 
                width =np.mean(np.diff(b)),
                color=color, 
                alpha=0.2)
        
        plt.plot(b[:-1], gaussian_filter1d(h,filter_window), 
                 color=color,
                 linewidth=2.0)
        
        plt.title(common_names[area_idx])
        
        axis = plt.gca()
        axis.get_yaxis().set_visible(False)
        [axis.spines[loc].set_visible(False) for loc in ['right', 'top', 'left']]  
        
        if area_idx == 0:
            plt.xlabel(metric)
        
        axes.append(axis)
        
        values.append(sub_df[metric].values)
    
    return axes, values
        

def plot_preference_comparison(metric, df, color, offset=-0.18, barwidth=0.3, areas=['VISp', 'VISl', 'VISal', 'VISpm', 'VISam']):

    """
    Compare distribution of a preference metrics across areas

    Parameters:
    ----------
    metric : str (e.g., 'pref_tf_dg')
    df : pd.DataFrame
    color : matplotlib color
    offset : float (offset of bars)

    Returns:
    ---------
    axes : list of axes for all plots
    values : list of np.ndarrays containing preference values
    
    """   
    
    axes = []
    all_values = []
    
    for area_idx, area in enumerate(areas):
        
        plt.subplot(1, len(areas), area_idx+1)
        
        sub_df = df[(df.ecephys_structure_acronym == area)]
            
        M = sub_df[metric].values
        
        all_values.append(M)
        
        h,b = np.histogram(M, bins=20, density=True)
        y = h[h>0]
        plt.bar(np.arange(len(y))+offset,y,width=barwidth,color=color,alpha=0.8)
        
        axis = plt.gca()
        axis.get_yaxis().set_visible(False)
        [axis.spines[loc].set_visible(False) for loc in ['right', 'top', 'left']] 
        
        axes.append(axis)
        
        plt.xticks(ticks=np.arange(len(y)), labels=np.sort(np.unique(M)))
        
        plt.title(common_names[area_idx])
        
        if area_idx == 0:
            plt.xlabel(metric)

    return axes, all_values
        

def get_color_palette(area, name='Allen CCF'):
    
    default_color = '#C9C9C9'
    
    if name == 'Steinmetz':
        
        palette = {'VISp' : '#1d2836',
                'VISl' : '#2d405a',
                'VISal' : '#4c6d9e',
                'VISrl' : '#3c587f',
                'VISpm' : '#56769d',
                'VISam' : '#7395bc',

                'DG' : '#432135',
                'CA3' : '#703457',
                'CA1' : '#9a4376',
                'CA' : '#9a4376',
                'POST' : '#bc568e',
                'SUB' : '#d26d9c',
                
                'LGd' : '#3c6636',
                'LP' : '#73ad6c',
                'LD' : '#31522f',
                
                'APN' : '#c15355',
                'MRN' : '#984445'
                }
        
    elif name == 'Allen CCF' :
        
        palette = {'VISp' : '#08858C',
                'VISl' : '#08858C',
                'VISal' : '#08858C',
                'VISrl' : '#009FAC',
                'VISpm' : '#08858C',
                'VISam' : '#08858C',
                
                'DG' : '#7ED04B',
                'CA3' : '#7ED04B',
                'CA1' : '#7ED04B',
                'CA' : '#7ED04B',
                'POST' : '#48C83C',
                'SUB' : '#4FC244',
                
                'LGd' : '#FF8084',
                'LP' : '#FF909F',
                'LD' : '#FF909F',
                
                'APN' : '#FF90FF',
                'MRN' : '#FF90FF'
                }
        
    elif name == 'Rainbow' :
        
        palette = {'VISp' : '#F6BB42',
                'VISl' : '#37BC9B',
                'VISal' : '#967ADC',
                'VISrl' : '#4A89DC',
                'VISpm' : '#E9573F',
                'VISam' : '#DA4453',
                
                'DG' : '#37BC9B',
                'CA3' : '#37BC9B',
                'CA1' : '#37BC9B',
                'CA' : '#7ED04B',
                'POST' : '#48CFAD',
                'SUB' : '#48CFAD',
                
                'LGd' : '#D770AD',
                'LP' : '#EC87C0',
                'LD' : '#EC87C0',
                
                'APN' : '#434A54',
                'MRN' : '#656D78'
                }
        
    elif name == 'cmocean':
                
        import cmocean
        
        hierarchy_colors = cmocean.cm.phase(np.arange(1.0,0.1,-0.124))
            
        palette = {
                'VISp' : hierarchy_colors[1],
                'VISl' : hierarchy_colors[2],
                'VISal' : hierarchy_colors[5],
                'VISrl' : hierarchy_colors[3],
                'VISpm' : hierarchy_colors[6],
                'VISam' : hierarchy_colors[7],
                
                'DG' : '#A4A4A4',
                'CA3' : '#6D6D6D',
                'CA1' : '#5B5B5B',
                'CA2' : '#5B5B5B',
                'CA' : '#7ED04B',
                'POST' : '#A4A4A4',
                'SUB' : '#A4A4A4',
                'HPC' : '#A4A4A4',
                
                'LGd' : hierarchy_colors[0],
                'LP' : hierarchy_colors[4]
                }
        
    elif name == 'seaborn':

        colors = [[217,141,194],
                  [129,116,177],
                  [78,115,174],
                  [101,178,201],
                  [88,167,106],
                  [202,183,120],
                  [219,132,87],
                  [194,79,84]]
        
        def scale_colors(color):
            return [col/255 for col in color]
        
        hierarchy_colors = [scale_colors(col) for col in colors]
        
        palette = {
                'VISp' : hierarchy_colors[1],
                'VISl' : hierarchy_colors[2],
                'VISal' : hierarchy_colors[5],
                'VISrl' : hierarchy_colors[3],
                'VISpm' : hierarchy_colors[6],
                'VISam' : hierarchy_colors[7],
                
                
                'DG' : '#A4A4A4',
                'CA3' : '#6D6D6D',
                'CA1' : '#5B5B5B',
                'CA2' : '#5B5B5B',
                'CA' : '#7ED04B',
                'POST' : '#A4A4A4',
                'SUB' : '#A4A4A4',
                'HPC' : '#A4A4A4',
                
                'LGd' : hierarchy_colors[0],
                'LP' : hierarchy_colors[4]
                }
        
    else:
        raise Error('No matching palette name')
        
    if area in palette.keys():
        
        return palette[area]
    
    else:
        return default_color
    
    