#imports
from ...utilities.misc import cd, crop_and_overwrite_image
from ...utilities.config import CONST as UNIV_CONST
import numpy as np
import matplotlib.pyplot as plt
import pathlib

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
    if save_dirpath is not None :
        if not save_dirpath.is_dir() :
            save_dirpath.mkdir()
        with cd(save_dirpath) :
            plt.savefig(FIGURE_NAME)
            plt.close()
            crop_and_overwrite_image(FIGURE_NAME)
    else :
        plt.savefig(FIGURE_NAME)
        plt.close()
        crop_and_overwrite_image(FIGURE_NAME)

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
        ax1.plot([mean,mean],[0.8*y for y in ax1.get_ylim()],linewidth=2,color='m',label=f'mean={mean}')
        ax1.plot([med,med],[0.8*y for y in ax1.get_ylim()],linewidth=2,color='r',label=f'median={med}')
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
        print(f'axis_min = {axis_min}, hists_by_layer[{axis_min},{layer_n-1}] = {hists_by_layer[axis_min,layer_n-1]}, axis_max = {axis_max}')
        log_bins = np.logspace(np.log10(axis_min),np.log10(axis_max),61)
        background_layer_log_hist = np.zeros((len(log_bins)-1),dtype=np.uint64)
        signal_layer_log_hist = np.zeros((len(log_bins)-1),dtype=np.uint64)
        log_bin_i = 0
        for lin_bin in range(axis_min,axis_max+1) :
            if lin_bin>log_bins[log_bin_i+1] :
                log_bin_i+=1
            if lin_bin>chosen_t :
                signal_layer_log_hist[log_bin_i]+=hists_by_layer[lin_bin,layer_n-1]
            else :
                background_layer_log_hist[log_bin_i]+=hists_by_layer[lin_bin,layer_n-1]
        ax2.bar(log_bins[:-1],background_layer_log_hist,width=np.diff(log_bins),log=True,align='edge',label='background')
        ax2.bar(log_bins[:-1],signal_layer_log_hist,width=np.diff(log_bins),log=True,align='edge',label='signal')
        ax2.set_xscale('log')
        ax2.set_title('pixel histogram (summed over all images)')
        ax2.set_xlabel('log pixel intensity (counts)')
        ax2.set_ylabel('log(# of image pixels)')
        ax2.legend(loc='best')
        fn = f'layer_{layer_n}_background_threshold_plots.png'
        if save_dirpath is not None :
            with cd(save_dirpath) :
                plt.savefig(fn)
                plt.close()
                crop_and_overwrite_image(fn)
        else :
            plt.savefig(fn)
            plt.close()
            crop_and_overwrite_image(fn)


