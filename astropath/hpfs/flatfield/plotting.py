#imports
from .utilities import RectangleThresholdTableEntry
from ...utilities.tableio import readtable
from ...utilities.misc import cd, crop_and_overwrite_image
from ...utilities.config import CONST as UNIV_CONST
import numpy as np
import matplotlib.pyplot as plt
import pathlib

def save_figure_in_dir(pyplot_inst,figname,save_dirpath=None) :
    """
    Helper function to save the current figure with a given name and crop it. 
    If save_dirpath is given the figure is saved in that directory (possibly creating it)
    """
    if save_dirpath is not None :
        if not save_dirpath.is_dir() :
            save_dirpath.mkdir()
        with cd(save_dirpath) :
            pyplot_inst.savefig(figname)
            pyplot_inst.close()
            crop_and_overwrite_image(figname)
    else :
        pyplot_inst.savefig(figname)
        pyplot_inst.close()
        crop_and_overwrite_image(figname)

def plot_tissue_edge_rectangle_locations(all_rects,edge_rects,root_dir,slideID,save_dirpath=None) :
    """
    Make and save the plot of the edge field locations next to the qptiff for reference

    all_rects    = list of all Rectangle objects to plot
    edge_rects   = list that is a subset of the above of any Rectangles that are on the edges of the tissue
    root_dir     = path to the root directory for the cohort the slide comes from
    slideID      = ID of the slide whose Rectangle locations are being plotted
    save_dirpath = path to directory to save the plot in (if None the plot is saved in the current directory)
    """
    #some constants
    SINGLE_FIG_SIZE = (9.6,7.2)
    FONTSIZE = 13.5
    FIGURE_NAME = f'{slideID}_rectangle_locations.png'
    #make and save the plot
    edge_rect_xs = [r.x for r in edge_rects]
    edge_rect_ys = [r.y for r in edge_rects]
    bulk_rect_xs = [r.x for r in all_rects if r not in edge_rects]
    bulk_rect_ys = [r.y for r in all_rects if r not in edge_rects]
    has_qptiff = (root_dir / f'{slideID}' / f'{UNIV_CONST.DBLOAD_DIR_NAME}' / f'{slideID}{UNIV_CONST.QPTIFF_SUFFIX}').is_file()
    if has_qptiff :
        f,(ax1,ax2) = plt.subplots(1,2,figsize=(2*SINGLE_FIG_SIZE[0],SINGLE_FIG_SIZE[1]))
    else :
        f,ax1 = plt.subplots(figsize=SINGLE_FIG_SIZE)
    ax1.scatter(edge_rect_xs,edge_rect_ys,marker='o',color='r',label='edges')
    ax1.scatter(bulk_rect_xs,bulk_rect_ys,marker='o',color='b',label='bulk')
    ax1.invert_yaxis()
    ax1.set_title(f'{slideID} rectangles, ({len(edge_rect_xs)} edge and {len(bulk_rect_xs)} bulk)',fontsize=FONTSIZE)
    ax1.legend(loc='best',fontsize=FONTSIZE)
    ax1.set_xlabel('x position',fontsize=FONTSIZE)
    ax1.set_ylabel('y position',fontsize=FONTSIZE)
    if has_qptiff :
        ax2.imshow(mpimg.imread(root_dir / f'{slideID}' / f'{UNIV_CONST.DBLOAD_DIR_NAME}' / f'{slideID}{UNIV_CONST.QPTIFF_SUFFIX}'))
        ax2.set_title('reference qptiff',fontsize=FONTSIZE)
    save_figure_in_dir(plt,FIGURE_NAME,save_dirpath)

def plot_image_layer_thresholds_with_histograms(image_background_thresholds_by_layer,slide_thresholds,hists_by_layer,save_dirpath=None) :
    """
    Make and save plots of a slide's distributions of background thresholds found/overall best thresholds 
    and comparisons with the pixel histograms, one plot for every image layer
    
    image_background_thresholds_by_layer = an array of all background thresholds found in every image layer 
    slide_thresholds = a list of the optimal thresholds found for the slide by layer
    hists_by_layer = an array of layer pixel histograms summed over all images used to find the background thresholds
    save_dirpath = path to directory to save the plot in (if None the plot is saved in the current directory)
    """
    assert(image_background_thresholds_by_layer.shape[-1]==len(slide_thresholds)==hists_by_layer.shape[-1])
    for threshold in slide_thresholds :
        layer_n = threshold.layer_n
        thresholds_to_plot = image_background_thresholds_by_layer[:,layer_n-1][image_background_thresholds_by_layer[:,layer_n-1]!=0]
        f,(ax1,ax2) = plt.subplots(1,2,figsize=(2*6.4,4.6))
        max_threshold_found = np.max(thresholds_to_plot)
        ax1.hist(thresholds_to_plot,max_threshold_found+11,(0,max_threshold_found+11))            
        mean = np.mean(thresholds_to_plot); med = np.median(thresholds_to_plot)
        ax1.plot([mean,mean],[0.8*y for y in ax1.get_ylim()],linewidth=2,color='m',label=f'mean={mean:.2f}')
        ax1.plot([med,med],[0.8*y for y in ax1.get_ylim()],linewidth=2,color='r',label=f'median={int(med)}')
        ax1.set_title(f'optimal thresholds for images in layer {layer_n}')
        ax1.set_xlabel('pixel intensity (counts)')
        ax1.set_ylabel('# of images')
        ax1.legend(loc='best')
        chosen_t = threshold.counts_threshold
        axis_min = 0; axis_max = hists_by_layer.shape[0]-1
        while hists_by_layer[axis_min,layer_n-1]==0 :
            axis_min+=1
        while hists_by_layer[axis_max,layer_n-1]==0 :
            axis_max-=1
        log_bins = np.logspace(np.log10(axis_min),np.log10(axis_max),101)
        background_layer_log_hist = np.zeros((len(log_bins)-1),dtype=np.uint64)
        signal_layer_log_hist = np.zeros((len(log_bins)-1),dtype=np.uint64)
        log_bin_i = 0
        for lin_bin in range(axis_min,axis_max+1) :
            while lin_bin>log_bins[log_bin_i+1] and log_bin_i<len(log_bins)-2 :
                log_bin_i+=1
            if lin_bin>chosen_t :
                signal_layer_log_hist[log_bin_i]+=hists_by_layer[lin_bin,layer_n-1]
            else :
                background_layer_log_hist[log_bin_i]+=hists_by_layer[lin_bin,layer_n-1]
        ax2.bar(log_bins[:-1],background_layer_log_hist,width=np.diff(log_bins),log=True,align='edge',alpha=0.7,label='background')
        ax2.bar(log_bins[:-1],signal_layer_log_hist,width=np.diff(log_bins),log=True,align='edge',alpha=0.7,label='signal')
        ax2.plot([chosen_t,chosen_t],[ax2.get_ylim()[0],0.8*ax2.get_ylim()[1]],linewidth=2,color='r',label=f'threshold = {chosen_t} counts')
        ax2.set_xscale('log')
        ax2.set_title('pixel histogram (summed over all images)')
        ax2.set_xlabel('log pixel intensity (counts)')
        ax2.set_ylabel('log(# of image pixels)')
        ax2.legend(loc='best')
        save_figure_in_dir(plt,f'layer_{layer_n}_background_threshold_plots.png',save_dirpath)

def plot_background_thresholds_by_layer(datatable_filepath,chosen_threshold_table_entries,save_dirpath=None) :
    """
    Plot a slide's 10th and 90th percentile thresholds found, and the final chosen thresholds, in each image layer
    Based on a datatable that's saved when the optimal thresholds are found

    datatable_filepath = path to the datatable file containing RectangleThresholdTableEntry objects stored after 
                         all the individual tissue edge image thresholds were found
    chosen_threshold_table_entries = the list of overall optimal background thresholds stored as ThresholdTableEntry objects
    save_dirpath = path to directory to save the plot in (if None the plot is saved in the current directory)
    """
    rectangle_thresholds = readtable(datatable_filepath,RectangleThresholdTableEntry)
    assert len(chosen_threshold_table_entries) == len(set([rt.layer_n for rt in rectangle_thresholds]))
    nlayers = len(chosen_threshold_table_entries)
    chosen_cts_by_layer = []; chosen_cpmsts_by_layer = []
    for li in range(nlayers) :
        layer_tobjs = [ct for ct in chosen_threshold_table_entries if ct.layer_n==li+1]
        if len(layer_tobjs)!=1 :
            return
        chosen_cts_by_layer.append(layer_tobjs[0].counts_threshold)
        chosen_cpmsts_by_layer.append(layer_tobjs[0].counts_per_ms_threshold)
    cts_by_layer = [[rt.counts_threshold for rt in rectangle_thresholds if rt.layer_n==li+1] for li in range(nlayers)]
    cpmsts_by_layer = [[rt.counts_per_ms_threshold for rt in rectangle_thresholds if rt.layer_n==li+1] for li in range(nlayers)]
    low_ct_pctiles_by_layer = []; high_ct_pctiles_by_layer = []
    low_cpmst_pctiles_by_layer = []; high_cpmst_pctiles_by_layer = []
    for li in range(nlayers) :
        layer_cts = cts_by_layer[li]; layer_cpmsts = cpmsts_by_layer[li]
        layer_cts.sort(); layer_cpmsts.sort()
        low_ct_pctiles_by_layer.append(layer_cts[int(0.1*len(layer_cts))])
        low_cpmst_pctiles_by_layer.append(layer_cpmsts[int(0.1*len(layer_cts))])
        high_ct_pctiles_by_layer.append(layer_cts[int(0.9*len(layer_cts))])
        high_cpmst_pctiles_by_layer.append(layer_cpmsts[int(0.9*len(layer_cts))])
    xvals=list(range(1,nlayers+1))
    f,ax = plt.subplots(2,1,figsize=(12.8,2*4.6))
    plt.suptitle('Thresholds chosen from tissue edge HPFs by image layer')
    ax[0].plot(xvals,low_ct_pctiles_by_layer,marker='v',color='r',linewidth=2,label='10th %ile thresholds')
    ax[0].plot(xvals,high_ct_pctiles_by_layer,marker='^',color='b',linewidth=2,label='90th %ile thresholds')
    ax[0].plot(xvals,chosen_cts_by_layer,marker='o',color='k',linewidth=2,label='final chosen thresholds')
    ax[0].set_xlabel('image layer')
    ax[0].set_ylabel('pixel intensity (counts)')
    ax[0].legend(loc='best')
    ax[1].plot(xvals,low_cpmst_pctiles_by_layer,marker='v',color='r',linewidth=2,label='10th %ile thresholds')
    ax[1].plot(xvals,high_cpmst_pctiles_by_layer,marker='^',color='b',linewidth=2,label='90th %ile thresholds')
    ax[1].plot(xvals,chosen_cpmsts_by_layer,marker='o',color='k',linewidth=2,label='chosen thresholds (counts/ms)')
    ax[1].set_xlabel('image layer')
    ax[1].set_ylabel('pixel intensity (counts/ms)')
    ax[1].legend(loc='best')
    slideID = datatable_filepath.name.split('-')[0]
    save_figure_in_dir(plt,f'{slideID}_background_thresholds_by_layer.png',save_dirpath)

def plot_image_layers(image,name_stem,save_dirpath=None) :
    """
    Save .png reference plots of all layers of a given image

    image = the image whose layers should be plotted (assumed to be stored with shape [height,width,nlayers])
    name_stem = prefix to each image that will be saved ("_layer_[n].png" is added)
    save_dirpath = path to directory to save the plots in (if None the plot is saved in the current directory)
    """
    fig_size=(6.4,6.4*(image.shape[0]/image.shape[1]))
    for li in range(image.shape[-1]) :
        layer_title = f'{name_stem} layer {li+1}'
        layer_fn = f'{name_stem}_layer_{li+1}.png'
        f,ax = plt.subplots(figsize=fig_size)
        pos = ax.imshow(image[:,:,li])
        ax.set_title(layer_title)
        cax = f.add_axes([ax.get_position().x1+0.003,ax.get_position().y0,0.006,ax.get_position().height])
        f.colorbar(pos,cax=cax)
        save_figure_in_dir(plt,layer_fn,save_dirpath)
