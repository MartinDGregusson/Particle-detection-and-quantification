from process2 import MultipleImages
import os
import pickle
import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as stats

def multiple_histograms(bins_range, num_bins, dict_list, names=None, norm=True):
    """
    Plots two graphs based on the given dictionaries

    The first will be frequency weighted with gaussian curves,
    the second will be area weighted, which essentially means each
    bar is multiplied by x^2

    args:
    bins_range (tuple)      :   min and max value for the bins
    num_bins (int)          :   How many bins in the histogram
    dict_list (list)        :   List containing the dictionaries for each dataset
    names (list)            :   Name to be used for labelling the graphs, if not None
                                then must be at least as long as dict_list
    norm (bool)             :   Whether to normalize the frequency weighted data
    """
    bar_width = 0.5*(1/len(dict_list))*(bins_range[1]-bins_range[0])/num_bins
    bar_align = 'edge'

    #This is the frequency weighted graph
    fig1, ax1 = plt.subplots(figsize=(12, 5))
    ax1.set_xlabel(r'Particle size $\delta_{\mathrm{ECD}}$ [nm]')
    ax1.set_ylabel('Frequency weighted')
    offsets = np.arange(len(dict_list))-(len(dict_list)/2)
    colors = ["r", "g", "b", "c", "m", "y"]
    for i in range(len(dict_list)):
        if names is None:
            name is None
        else:
            name = names[i]
        if norm is True:
            divisor = np.sum(dict_list[i]["hist"])
        elif norm is False:
            divisor = 1
        else:
            divisor = norm
        ax1.bar(dict_list[i]["centre"]+bar_width*offsets[i], dict_list[i]["hist"]/divisor, width=bar_width, align=bar_align, label=name, color=colors[i%len(colors)], alpha=0.3)
        points = np.linspace(dict_list[i]["bins"][0], dict_list[i]["bins"][-1], 200)
        mean_val = dict_list[i]["ecd_mean"]
        std_val = dict_list[i]["ecd_std"]
        ax1.plot(points, ((bins_range[1]-bins_range[0])/num_bins)*stats.norm.pdf(points, mean_val, std_val), colors[i%len(colors)]+"--")
    ax1.legend()

    #This is the area weighted graph
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    ax2.set_xlabel(r'Particle size $\delta_{\mathrm{ECD}}$ [nm]')
    ax2.set_ylabel('Area weighted volume fraction')
    offsets = np.arange(len(dict_list))-len(dict_list)/2
    for i in range(len(dict_list)):
        if names is None:
            name is None
        else:
            name = names[i]
        ax2.bar(dict_list[i]["centre"]+bar_width*offsets[i], dict_list[i]["hist_norm_area"], width=bar_width, align=bar_align, label=name, color=colors[i%len(colors)])
        #points = np.linspace(dict_list[i]["bins"][0], dict_list[i]["bins"][-1], 200)
        #mean_val = dict_list[i]["ecd_mean"]                                                                                #I found no clear way of getting this data to fit
        #std_val = dict_list[i]["ecd_std"]                                                                                  #Code is left commented out in case someone needs it
        #ax2.plot(points, stats.norm.pdf(points, mean_val, std_val)*(points**2)/500, colors[i%len(colors)]+"--")
    ax2.legend()
    
    return fig1, ax1, fig2, ax2

def extract_dicts_from_pickled_objects(file_names, bins_range=(0,100), num_bins=50):
    """
    Extracts the dictionaries conatining relevant statistics
    from the pickled MultipleImages objects
    
    args:
    file_names (list)       :   List of paths to pickled objects
    bins_range (tuple)      :   min and max value for the bins
    num_bins (int)          :   How many bins in the histogram
    """
    dict_list = []
    name_list = []
    for file in file_names:
        process = MultipleImages()
        process.load_current(file)
        stats = process.overall_stats(bins_range=bins_range, num_bins=num_bins, show=False, return_values=True)
        dict_list.append(stats)
        name, ext = os.path.splitext(file)
        name_list.append(name)
    return dict_list, name_list

def extract_dicts_from_pickled_dicts(file_names):
    """
    Retrieves the pickled dictionaries from the given
    locations
    
    args:
    file_names (list)       :   List of paths to pickled dictionaries
    """
    dict_list = []
    name_list = []
    for filename in file_names:
        with open(filename, 'rb') as f:
            stats = pickle.load(f)
            dict_list.append(stats)
        name, ext = os.path.splitext(filename)
        name_list.append(name)
    return dict_list, name_list

def extract_dicts_from_objects(processes, bins_range=(0,100), num_bins=50):
    """
    Extracts the dictionaries conatining relevant statistics
    from the given MultipleImages objects
    
    args:
    processes (list)        :   List of MultipledImages objects
    bins_range (tuple)      :   min and max value for the bins
    num_bins (int)          :   How many bins in the histogram
    """
    dict_list = []
    name_list = None
    for process in processes:
        stats = process.overall_stats(bins_range=bins_range, num_bins=num_bins, show=False, return_values=True)
        dict_list.append(stats)
    return dict_list, name_list