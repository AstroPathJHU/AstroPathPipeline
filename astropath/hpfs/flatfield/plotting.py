#imports
from .utilities import RectangleThresholdTableEntry
from ..image_masking.config import CONST as MASKING_CONST
from ...utilities.tableio import readtable
from ...utilities.misc import cd, save_figure_in_dir
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

def plot_flagged_HPF_locations(sid,rectangles,lmrs,save_dirpath=None) :
    """
    Plot all of the HPF locations for a slide with their reasons for being flagged

    sid = slideID
    rectangles = the full list of rectangles for the slide
    lmrs = the list of LabelledMaskRegion objects for the slide
    save_dirpath = path to directory to save the plots in (if None the plot is saved in the current directory)
    """
    all_flagged_hpf_keys = [lmr.image_key for lmr in lmrs]
    hpf_identifiers = []
    for r in rectangles :
        key=r.file.rstrip(UNIV_CONST.IM3_EXT)
        key_x = float(key.split(',')[0].split('[')[1])
        key_y = float(key.split(',')[1].split(']')[0])
        if key in all_flagged_hpf_keys :
            key_strings = set([lmr.reason_flagged for lmr in lmrs if lmr.image_key==key])
            blur_flagged = 1 if MASKING_CONST.BLUR_FLAG_STRING in key_strings else 0
            saturation_flagged = 1 if MASKING_CONST.SATURATION_FLAG_STRING in key_strings else 0
            flagged_int = 1*blur_flagged+2*saturation_flagged
        else :
            flagged_int = 0
        hpf_identifiers.append({'x':key_x,'y':key_y,'flagged':flagged_int})
    colors_by_flag_int = ['gray','royalblue','gold','limegreen']
    labels_by_flag_int = ['not flagged','blur flagged','saturation flagged','blur and saturation']
    f,ax = plt.subplots(figsize=(9.6,7.2))
    for i in range(len(colors_by_flag_int)) :
        hpf_ids_to_plot = [identifier for identifier in hpf_identifiers if identifier['flagged']==i]
        if len(hpf_ids_to_plot)<1 :
            continue
        ax.scatter([hpfid['x'] for hpfid in hpf_ids_to_plot],
                   [hpfid['y'] for hpfid in hpf_ids_to_plot],
                   marker='o',
                   color=colors_by_flag_int[i],
                   label=labels_by_flag_int[i])
    ax.invert_yaxis()
    title_text = f'{sid} HPF center locations, ({len(rectangles)} in slide '
    title_text+=f'{len([hpfid for hpfid in hpf_identifiers if hpfid["flagged"]!=0])} flagged)'
    ax.set_title(title_text,fontsize=13.5)
    ax.legend(loc='best',fontsize=10)
    ax.set_xlabel('HPF local x position',fontsize=13.5)
    ax.set_ylabel('HPF local y position',fontsize=13.5)
    fn = f'{sid}_flagged_hpf_locations.png'
    save_figure_in_dir(plt,fn,save_dirpath)

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
        ax.axis('off')
        cax = f.add_axes([ax.get_position().x1+0.006,ax.get_position().y0,0.02,ax.get_position().height])
        f.colorbar(pos,cax=cax)
        save_figure_in_dir(plt,layer_fn,save_dirpath)

def flatfield_image_pixel_intensity_plot(flatfield_image,batchID=None,save_dirpath=None) :
    """
    Plot the max/min, 5th/95th %ile, and std. dev. of a flatfield image's correction factors by layer 

    flatfield_image = the flatfield image array for which the plot should be made
    batchID = the batchID for the given flatfield model (used in titles and names, optional)
    save_dirpath = path to directory to save the plots in (if None the plot is saved in the current directory)
    """
    #figure out the number of layers and the filter breaks
    nlayers=flatfield_image.shape[-1]
    if nlayers==35 :
        last_filter_layers = [lg[1] for lg in UNIV_CONST.LAYER_GROUPS_35[:-1]] 
    elif nlayers==43 :
        last_filter_layers = [lg[1] for lg in UNIV_CONST.LAYER_GROUPS_43[:-1]]
    else :
        raise ValueError(f'ERROR: number of layers {nlayers} is not a recognized option!') 
    yclip = int(flatfield_image.shape[0]*0.1)
    xclip = int(flatfield_image.shape[1]*0.1)
    flatfield_image_clipped=flatfield_image[yclip:-yclip,xclip:-xclip,:]
    u_mins=[]; u_lows=[]; u_maxs=[]; u_highs=[]
    c_mins=[]; c_lows=[]; c_maxs=[]; c_highs=[]
    u_std_devs=[]; c_std_devs=[]
    plt.figure(figsize=(16.8,(27./64.)*16.8))
    xaxis_vals = list(range(1,nlayers+1))
    #iterate over the layers
    for layer_i in range(nlayers) :
        #find the min, max, and 5/95%ile pixel intensities for this uncorrected and corrected image layer
        sorted_u_layer = np.sort((flatfield_image[:,:,layer_i]).flatten())/np.mean(flatfield_image[:,:,layer_i])
        u_mins.append(sorted_u_layer[0]); u_lows.append(sorted_u_layer[int(0.05*len(sorted_u_layer))])
        u_stddev = np.std(sorted_u_layer); u_std_devs.append(u_stddev)
        u_maxs.append(sorted_u_layer[-1]); u_highs.append(sorted_u_layer[int(0.95*len(sorted_u_layer))])
        sorted_c_layer = np.sort((flatfield_image_clipped[:,:,layer_i]).flatten())/np.mean(flatfield_image_clipped[:,:,layer_i])
        c_mins.append(sorted_c_layer[0]); c_lows.append(sorted_c_layer[int(0.05*len(sorted_c_layer))])
        c_stddev = np.std(sorted_c_layer); c_std_devs.append(c_stddev)
        c_maxs.append(sorted_c_layer[-1]); c_highs.append(sorted_c_layer[int(0.95*len(sorted_c_layer))])
        if layer_i==0 :
            plt.fill([layer_i+0.5,layer_i+0.5,layer_i+1.5,layer_i+1.5],[1.-u_stddev,1.+u_stddev,1.+u_stddev,1.-u_stddev],'mediumseagreen',alpha=0.5,label='overall std. dev.')
            plt.fill([layer_i+0.5,layer_i+0.5,layer_i+1.5,layer_i+1.5],[1.-c_stddev,1.+c_stddev,1.+c_stddev,1.-c_stddev],'goldenrod',alpha=0.5,label='std. dev. (central 64%)')
        else :
            plt.fill([layer_i+0.5,layer_i+0.5,layer_i+1.5,layer_i+1.5],[1.-u_stddev,1.+u_stddev,1.+u_stddev,1.-u_stddev],'mediumseagreen',alpha=0.5)
            plt.fill([layer_i+0.5,layer_i+0.5,layer_i+1.5,layer_i+1.5],[1.-c_stddev,1.+c_stddev,1.+c_stddev,1.-c_stddev],'goldenrod',alpha=0.5)
    #plot the relative intensity plots together, with the broadband filter breaks
    plt.plot([xaxis_vals[0],xaxis_vals[-1]],[1.0,1.0],color='darkgreen',linestyle='dashed',label='mean')
    totalmin=min(min(u_lows),min(c_lows))
    totalmax=max(max(u_highs),max(c_highs))
    for i in range(len(last_filter_layers)+1) :
        f_i = 0 if i==0 else last_filter_layers[i-1]
        l_i = xaxis_vals[-1] if i==len(last_filter_layers) else last_filter_layers[i]
        if i==0 :
            plt.plot(xaxis_vals[f_i:l_i],u_lows[f_i:l_i],color='darkred',marker='v',label=r'overall 5th %ile')
            plt.plot(xaxis_vals[f_i:l_i],u_highs[f_i:l_i],color='darkred',marker='^',label=r'overall 95th %ile')
            plt.plot(xaxis_vals[f_i:l_i],c_lows[f_i:l_i],color='darkblue',marker='v',label=r'5th %ile (central 64%)')
            plt.plot(xaxis_vals[f_i:l_i],c_highs[f_i:l_i],color='darkblue',marker='^',label=r'95th %ile (central 64%)')
            plt.plot([l_i+0.5,l_i+0.5],[totalmin-0.01,totalmax+0.01],color='black',linewidth=2,linestyle='dotted',label='broadband filter changeover')
        else :
            plt.plot(xaxis_vals[f_i:l_i],u_lows[f_i:l_i],color='darkred',marker='v')
            plt.plot(xaxis_vals[f_i:l_i],u_highs[f_i:l_i],color='darkred',marker='^')
            plt.plot(xaxis_vals[f_i:l_i],c_lows[f_i:l_i],color='darkblue',marker='v')
            plt.plot(xaxis_vals[f_i:l_i],c_highs[f_i:l_i],color='darkblue',marker='^')
            if i!=len(last_filter_layers) :
                plt.plot([l_i+0.5,l_i+0.5],[totalmin-0.01,totalmax+0.01],color='black',linewidth=2,linestyle='dotted')
    plt.title('flatfield image pixel intensity relative to layer mean',fontsize=14)
    plt.xlabel('layer number',fontsize=14)
    #fix the range on the x-axis to accommodate the legend
    plt.xlim(0,nlayers+12)
    plt.ylabel('pixel intensity relative to layer mean',fontsize=14)
    plt.legend(loc='lower right')
    #write out the figure
    fn = 'flatfield'
    if batchID is not None :
        fn+=f'_BatchID_{batchID:02d}'
    fn+='_pixel_intensities.png'
    save_figure_in_dir(plt,fn,save_dirpath)

def mask_stack_whole_image_vs_central_region(mask_stack,save_dirpath=None) :
    """
    Plot the max/min, 5th/95th %ile, and std. dev. of a mask stack's number of images stacked by layer 
    in the whole image and in the central region of the image only

    mask_stack = the mask stack to plot
    save_dirpath = path to directory to save the plots in (if None the plot is saved in the current directory)
    """
    nlayers=mask_stack.shape[-1]
    if nlayers==35 :
        last_filter_layers = [lg[1] for lg in UNIV_CONST.LAYER_GROUPS_35[:-1]] 
    elif nlayers==43 :
        last_filter_layers = [lg[1] for lg in UNIV_CONST.LAYER_GROUPS_43[:-1]]
    else :
        raise ValueError(f'ERROR: number of layers {nlayers} is not a recognized option!') 
    yclip = int(mask_stack.shape[0]*0.1)
    xclip = int(mask_stack.shape[1]*0.1)
    clipped_mask_stack = mask_stack[yclip:-yclip,xclip:-xclip,:]
    u_mins=[]; u_lows=[]; u_maxs=[]; u_highs=[]
    c_mins=[]; c_lows=[]; c_maxs=[]; c_highs=[]
    u_std_devs=[]; c_std_devs=[]
    plt.figure(figsize=(16.8,7.2))
    xaxis_vals = list(range(1,nlayers+1))
    #iterate over the layers
    for layer_i in range(nlayers) :
        #find the min, max, and 5/95%ile pixel intensities for this uncorrected and corrected image layer
        sorted_u_layer = np.sort((mask_stack[:,:,layer_i]).flatten())/np.mean(mask_stack[:,:,layer_i])
        u_mins.append(sorted_u_layer[0]); u_lows.append(sorted_u_layer[int(0.05*len(sorted_u_layer))])
        u_stddev = np.std(sorted_u_layer); u_std_devs.append(u_stddev)
        u_maxs.append(sorted_u_layer[-1]); u_highs.append(sorted_u_layer[int(0.95*len(sorted_u_layer))])
        sorted_c_layer = np.sort((clipped_mask_stack[:,:,layer_i]).flatten())/np.mean(clipped_mask_stack[:,:,layer_i])
        c_mins.append(sorted_c_layer[0]); c_lows.append(sorted_c_layer[int(0.05*len(sorted_c_layer))])
        c_stddev = np.std(sorted_c_layer); c_std_devs.append(c_stddev)
        c_maxs.append(sorted_c_layer[-1]); c_highs.append(sorted_c_layer[int(0.95*len(sorted_c_layer))])
        if layer_i==0 :
            plt.fill([layer_i+0.5,layer_i+0.5,layer_i+1.5,layer_i+1.5],[1.-u_stddev,1.+u_stddev,1.+u_stddev,1.-u_stddev],'mediumseagreen',alpha=0.5,label='overall std. dev.')
            plt.fill([layer_i+0.5,layer_i+0.5,layer_i+1.5,layer_i+1.5],[1.-c_stddev,1.+c_stddev,1.+c_stddev,1.-c_stddev],'goldenrod',alpha=0.5,label='std. dev. (central 64%)')
        else :
            plt.fill([layer_i+0.5,layer_i+0.5,layer_i+1.5,layer_i+1.5],[1.-u_stddev,1.+u_stddev,1.+u_stddev,1.-u_stddev],'mediumseagreen',alpha=0.5)
            plt.fill([layer_i+0.5,layer_i+0.5,layer_i+1.5,layer_i+1.5],[1.-c_stddev,1.+c_stddev,1.+c_stddev,1.-c_stddev],'goldenrod',alpha=0.5)
    #plot the relative intensity plots together, with the broadband filter breaks
    plt.plot([xaxis_vals[0],xaxis_vals[-1]],[1.0,1.0],color='darkgreen',linestyle='dashed',label='mean')
    totalmin=min(min(u_lows),min(c_lows))
    totalmax=max(max(u_highs),max(c_highs))
    for i in range(len(last_filter_layers)+1) :
        f_i = 0 if i==0 else last_filter_layers[i-1]
        l_i = xaxis_vals[-1] if i==len(last_filter_layers) else last_filter_layers[i]
        if i==0 :
            plt.plot(xaxis_vals[f_i:l_i],u_lows[f_i:l_i],color='darkred',marker='v',label=r'overall 5th %ile')
            plt.plot(xaxis_vals[f_i:l_i],u_highs[f_i:l_i],color='darkred',marker='^',label=r'overall 95th %ile')
            plt.plot(xaxis_vals[f_i:l_i],c_lows[f_i:l_i],color='darkblue',marker='v',label=r'5th %ile (central 64%)')
            plt.plot(xaxis_vals[f_i:l_i],c_highs[f_i:l_i],color='darkblue',marker='^',label=r'95th %ile (central 64%)')
            plt.plot([l_i+0.5,l_i+0.5],[totalmin-0.01,totalmax+0.01],color='black',linewidth=2,linestyle='dotted',label='broadband filter changeover')
        else :
            plt.plot(xaxis_vals[f_i:l_i],u_lows[f_i:l_i],color='darkred',marker='v')
            plt.plot(xaxis_vals[f_i:l_i],u_highs[f_i:l_i],color='darkred',marker='^')
            plt.plot(xaxis_vals[f_i:l_i],c_lows[f_i:l_i],color='darkblue',marker='v')
            plt.plot(xaxis_vals[f_i:l_i],c_highs[f_i:l_i],color='darkblue',marker='^')
            if i!=len(last_filter_layers) :
                plt.plot([l_i+0.5,l_i+0.5],[totalmin-0.01,totalmax+0.01],color='black',linewidth=2,linestyle='dotted')
    plt.title('relative spread in number of images stacked per layer for whole and central 64% images',fontsize=14)
    plt.xlabel('layer number',fontsize=14)
    #fix the range on the x-axis to accommodate the legend
    plt.xlim(0,nlayers+(10 if nlayers==35 else 12))
    plt.ylabel('number of images stacked relative to layer mean',fontsize=14)
    plt.legend(loc='best')
    save_figure_in_dir(plt,'mask_stack_whole_image_vs_central_region.png',save_dirpath)
    u_hi_lo_spread = [u_highs[li]-u_lows[li] for li in range(nlayers)]
    c_hi_lo_spread = [c_highs[li]-c_lows[li] for li in range(nlayers)]
    print(f'Mean whole image 5th-95th %ile = {np.mean(np.array(u_hi_lo_spread))}')
    print(f'Mean central 64% 5th-95th %ile = {np.mean(np.array(c_hi_lo_spread))}')
    print(f'Mean whole image std. dev. = {np.mean(np.array(u_std_devs))}')
    print(f'Mean central 64% std. dev. = {np.mean(np.array(c_std_devs))}')

def corrected_mean_image_PI_and_IV_plots(smoothed_mean_image,smoothed_corrected_mean_image,central_region=False,save_dirpath=None) :
    """
    Plot the max/min, 5th/95th %ile, and std. dev. of a mean image's pixel intensities by layer, before and after correction by a flatfield model
    Also creates a second plot of the pre/post correction std. dev. and 5th-95th percentile relative intensity variations
    Plots can be created for the entire image region or for the central region only.

    smoothed_mean_image = the smoothed pre-correction mean image array
    smoothed_corrected_mean_image = the smoothed post-correction mean image array
    central_region = True if only the central 64% should be used to calculate the plotted statistics, False otherwise.
    save_dirpath = path to directory to save the plots in (if None the plot is saved in the current directory)
    """
    assert smoothed_mean_image.shape == smoothed_corrected_mean_image.shape
    #clip the outer edges off if the plots are for the central region only
    if central_region :
        yclip = int(smoothed_mean_image.shape[0]*0.1)
        xclip = int(smoothed_mean_image.shape[1]*0.1)
        smoothed_mean_image = smoothed_mean_image[yclip:-yclip,xclip:-xclip,:]
        smoothed_corrected_mean_image = smoothed_corrected_mean_image[yclip:-yclip,xclip:-xclip,:]
    #figure out the number of layers and filter breaks
    nlayers=smoothed_mean_image.shape[-1]
    if nlayers==35 :
        last_filter_layers = [lg[1] for lg in UNIV_CONST.LAYER_GROUPS_35[:-1]] 
    elif nlayers==43 :
        last_filter_layers = [lg[1] for lg in UNIV_CONST.LAYER_GROUPS_43[:-1]]
    else :
        raise ValueError(f'ERROR: number of layers {nlayers} is not a recognized option!') 
    #keep track of the uncorrected and corrected images' minimum and maximum (and 5/95%ile) pixel intensities while the other plots are made
    u_low_pixel_intensities=[]; u_high_pixel_intensities=[]
    c_low_pixel_intensities=[]; c_high_pixel_intensities=[]
    u_std_devs=[]; c_std_devs=[]
    plt.figure(figsize=(16.8,(27./64.)*16.8))
    xaxis_vals = list(range(1,nlayers+1))
    #iterate over the layers
    for layer_i in range(nlayers) :
        #find the min, max, and 5/95%ile pixel intensities for this uncorrected and corrected image layer
        sorted_u_layer = np.sort((smoothed_mean_image[:,:,layer_i]).flatten())/np.mean(smoothed_mean_image[:,:,layer_i])
        u_low_pixel_intensities.append(sorted_u_layer[int(0.05*len(sorted_u_layer))])
        u_stddev = np.std(sorted_u_layer); u_std_devs.append(u_stddev)
        u_high_pixel_intensities.append(sorted_u_layer[int(0.95*len(sorted_u_layer))])
        sorted_c_layer = np.sort((smoothed_corrected_mean_image[:,:,layer_i]).flatten())/np.mean(smoothed_corrected_mean_image[:,:,layer_i])
        c_low_pixel_intensities.append(sorted_c_layer[int(0.05*len(sorted_c_layer))])
        c_stddev = np.std(sorted_c_layer); c_std_devs.append(c_stddev)
        c_high_pixel_intensities.append(sorted_c_layer[int(0.95*len(sorted_c_layer))])
        if layer_i==0 :
            plt.fill([layer_i+0.5,layer_i+0.5,layer_i+1.5,layer_i+1.5],[1.-u_stddev,1.+u_stddev,1.+u_stddev,1.-u_stddev],'mediumseagreen',alpha=0.5,label='uncorrected std. dev.')
            plt.fill([layer_i+0.5,layer_i+0.5,layer_i+1.5,layer_i+1.5],[1.-c_stddev,1.+c_stddev,1.+c_stddev,1.-c_stddev],'goldenrod',alpha=0.5,label='corrected std. dev.')
        else :
            plt.fill([layer_i+0.5,layer_i+0.5,layer_i+1.5,layer_i+1.5],[1.-u_stddev,1.+u_stddev,1.+u_stddev,1.-u_stddev],'mediumseagreen',alpha=0.5)
            plt.fill([layer_i+0.5,layer_i+0.5,layer_i+1.5,layer_i+1.5],[1.-c_stddev,1.+c_stddev,1.+c_stddev,1.-c_stddev],'goldenrod',alpha=0.5)
    #plot the relative intensity plots together, with the broadband filter breaks
    plt.plot([xaxis_vals[0],xaxis_vals[-1]],[1.0,1.0],color='darkgreen',linestyle='dashed',label='mean')
    totalmin=min(min(u_low_pixel_intensities),min(c_low_pixel_intensities))
    totalmax=max(max(u_high_pixel_intensities),max(c_high_pixel_intensities))
    for i in range(len(last_filter_layers)+1) :
        f_i = 0 if i==0 else last_filter_layers[i-1]
        l_i = xaxis_vals[-1] if i==len(last_filter_layers) else last_filter_layers[i]
        if i==0 :
            plt.plot(xaxis_vals[f_i:l_i],u_low_pixel_intensities[f_i:l_i],color='darkred',marker='v',label=r'uncorrected 5th %ile')
            plt.plot(xaxis_vals[f_i:l_i],u_high_pixel_intensities[f_i:l_i],color='darkred',marker='^',label=r'uncorrected 95th %ile')
            plt.plot(xaxis_vals[f_i:l_i],c_low_pixel_intensities[f_i:l_i],color='darkblue',marker='v',label=r'corrected 5th %ile')
            plt.plot(xaxis_vals[f_i:l_i],c_high_pixel_intensities[f_i:l_i],color='darkblue',marker='^',label=r'corrected 95th %ile')
            plt.plot([l_i+0.5,l_i+0.5],[totalmin-0.01,totalmax+0.01],color='black',linewidth=2,linestyle='dotted',label='broadband filter changeover')
        else :
            plt.plot(xaxis_vals[f_i:l_i],u_low_pixel_intensities[f_i:l_i],color='darkred',marker='v')
            plt.plot(xaxis_vals[f_i:l_i],u_high_pixel_intensities[f_i:l_i],color='darkred',marker='^')
            plt.plot(xaxis_vals[f_i:l_i],c_low_pixel_intensities[f_i:l_i],color='darkblue',marker='v')
            plt.plot(xaxis_vals[f_i:l_i],c_high_pixel_intensities[f_i:l_i],color='darkblue',marker='^')
            if i!=len(last_filter_layers) :
                plt.plot([l_i+0.5,l_i+0.5],[totalmin-0.01,totalmax+0.01],color='black',linewidth=2,linestyle='dotted')
    plt.title('uncorrected/corrected smoothed mean image relative pixel intensities',fontsize=14)
    plt.xlabel('layer number',fontsize=14)
    #fix the range on the x-axis to accommodate the legend
    plt.xlim(0,nlayers+(10 if nlayers==35 else 12))
    plt.ylabel('pixel intensity relative to layer mean',fontsize=14)
    plt.legend(loc='lower right')
    fn = 'smoothed_mean_image_pixel_intensities'
    if central_region :
        fn+='_central_region'
    fn+='.png'
    save_figure_in_dir(plt,fn,save_dirpath)
    #plot the reduction in illumination variation
    u_hi_lo_spread = [u_high_pixel_intensities[li]-u_low_pixel_intensities[li] for li in range(nlayers)]
    c_hi_lo_spread = [c_high_pixel_intensities[li]-c_low_pixel_intensities[li] for li in range(nlayers)]
    f,ax=plt.subplots(figsize=(9.6,0.5*9.6))
    ax.plot(xaxis_vals,u_hi_lo_spread,marker='v',linestyle='dashed',linewidth=2,label='uncorrected 5th-95th %ile')
    ax.plot(xaxis_vals,c_hi_lo_spread,marker='^',linestyle='dashed',linewidth=2,label='corrected 5th-95th %ile')
    ax.plot(xaxis_vals,u_std_devs,marker='<',linestyle='dotted',linewidth=2,label='uncorrected std. dev.')
    ax.plot(xaxis_vals,c_std_devs,marker='>',linestyle='dotted',linewidth=2,label='corrected std. dev.')
    ax.set_ylim(0,(ax.get_ylim())[-1]*1.25)
    ax.set_title('illumination variation reduction by layer')
    ax.set_xlabel('image layer')
    ax.set_ylabel('relative flux')
    ax.legend(loc='best')
    fn = 'illumination_variation_reduction'
    if central_region :
        fn+='_central_region'
    fn+='.png'
    save_figure_in_dir(plt,fn,save_dirpath)
