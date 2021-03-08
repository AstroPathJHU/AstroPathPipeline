#imports
from .config import CONST
from ..utilities.misc import cd, cropAndOverwriteImage
from ..utilities.config import CONST as UNIV_CONST
import numpy as np, matplotlib.pyplot as plt
import os, glob, statistics

#helper function to plot the max/min, 5th/95th %ile, and std. dev. of a flatfield image's correction factors by layer 
def flatfieldImagePixelIntensityPlot(flatfield_image,savename=None) :
    #figure out the number of layers and the filter breaks
    nlayers=flatfield_image.shape[-1]
    if nlayers==35 :
        LAST_FILTER_LAYERS = [lg[1] for lg in UNIV_CONST.LAYER_GROUPS_35[:-1]] 
    elif nlayers==43 :
        LAST_FILTER_LAYERS = [lg[1] for lg in UNIV_CONST.LAYER_GROUPS_43[:-1]]
    else :
        raise ValueError(f'ERROR: number of layers {nlayers} is not a recognized option!') 
    yclip = int(flatfield_image.shape[0]*0.1)
    xclip = int(flatfield_image.shape[1]*0.1)
    flatfield_image_clipped=flatfield_image[yclip:-yclip,xclip:-xclip,:]
    u_mins=[]; u_lows=[]; u_maxs=[]; u_highs=[]
    c_mins=[]; c_lows=[]; c_maxs=[]; c_highs=[]
    u_std_devs=[]; c_std_devs=[]
    plt.figure(figsize=(CONST.INTENSITY_FIG_WIDTH,(27./64.)*CONST.INTENSITY_FIG_WIDTH))
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
    for i in range(len(LAST_FILTER_LAYERS)+1) :
        f_i = 0 if i==0 else LAST_FILTER_LAYERS[i-1]
        l_i = xaxis_vals[-1] if i==len(LAST_FILTER_LAYERS) else LAST_FILTER_LAYERS[i]
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
            if i!=len(LAST_FILTER_LAYERS) :
                plt.plot([l_i+0.5,l_i+0.5],[totalmin-0.01,totalmax+0.01],color='black',linewidth=2,linestyle='dotted')
    plt.title('flatfield image pixel intensity relative to layer mean',fontsize=14)
    plt.xlabel('layer number',fontsize=14)
    #fix the range on the x-axis to accommodate the legend
    plt.xlim(0,nlayers+12)
    plt.ylabel('pixel intensity relative to layer mean',fontsize=14)
    plt.legend(loc='lower right')
    if savename is not None :
        plt.savefig(savename)
        plt.close()
        cropAndOverwriteImage(savename)
    else :
        plt.show()
    print(f'Overall max, whole image = {np.max(flatfield_image)}')
    print(f'Overall min, whole image = {np.min(flatfield_image)}')
    print(f'Overall max, central 64% = {np.max(flatfield_image_clipped)}')
    print(f'Overall min, central 64% = {np.min(flatfield_image_clipped)}')
    u_hi_lo_spread = [u_highs[li]-u_lows[li] for li in range(nlayers)]
    c_hi_lo_spread = [c_highs[li]-c_lows[li] for li in range(nlayers)]
    print(f'Mean whole image 5th-95th %ile = {np.mean(np.array(u_hi_lo_spread))}')
    print(f'Mean central 64% 5th-95th %ile = {np.mean(np.array(c_hi_lo_spread))}')
    print(f'Mean whole image std. dev. = {np.mean(np.array(u_std_devs))}')
    print(f'Mean central 64% std. dev. = {np.mean(np.array(c_std_devs))}')

#helper function to plot how the average intensities of the mean image change in each layer after application of the flatfield corrections
def correctedMeanImagePIandIVplots(smoothed_mean_image,smoothed_corrected_mean_image,central_region=False,pi_savename=None,iv_plot_name=None,iv_csv_name=None) :
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
        LAST_FILTER_LAYERS = [lg[1] for lg in UNIV_CONST.LAYER_GROUPS_35[:-1]] 
    elif nlayers==43 :
        LAST_FILTER_LAYERS = [lg[1] for lg in UNIV_CONST.LAYER_GROUPS_43[:-1]]
    else :
        raise ValueError(f'ERROR: number of layers {nlayers} is not a recognized option!') 
    #keep track of the uncorrected and corrected images' minimum and maximum (and 5/95%ile) pixel intensities while the other plots are made
    u_low_pixel_intensities=[]; u_high_pixel_intensities=[]
    c_low_pixel_intensities=[]; c_high_pixel_intensities=[]
    u_std_devs=[]; c_std_devs=[]
    plt.figure(figsize=(CONST.INTENSITY_FIG_WIDTH,(27./64.)*CONST.INTENSITY_FIG_WIDTH))
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
    for i in range(len(LAST_FILTER_LAYERS)+1) :
        f_i = 0 if i==0 else LAST_FILTER_LAYERS[i-1]
        l_i = xaxis_vals[-1] if i==len(LAST_FILTER_LAYERS) else LAST_FILTER_LAYERS[i]
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
            if i!=len(LAST_FILTER_LAYERS) :
                plt.plot([l_i+0.5,l_i+0.5],[totalmin-0.01,totalmax+0.01],color='black',linewidth=2,linestyle='dotted')
    plt.title('uncorrected/corrected smoothed mean image relative pixel intensities',fontsize=14)
    plt.xlabel('layer number',fontsize=14)
    #fix the range on the x-axis to accommodate the legend
    plt.xlim(0,nlayers+(10 if nlayers==35 else 12))
    plt.ylabel('pixel intensity relative to layer mean',fontsize=14)
    plt.legend(loc='lower right')
    if pi_savename is not None :
        plt.savefig(pi_savename)
        plt.close()
        cropAndOverwriteImage(pi_savename)
    else :
        plt.show()
    #plot the reduction in illumination variation and save the values to a csv file as well
    u_hi_lo_spread = [u_high_pixel_intensities[li]-u_low_pixel_intensities[li] for li in range(nlayers)]
    c_hi_lo_spread = [c_high_pixel_intensities[li]-c_low_pixel_intensities[li] for li in range(nlayers)]
    print(f'Mean uncorrected 5th-95th %ile = {np.mean(np.array(u_hi_lo_spread))}')
    print(f'Mean corrected 5th-95th %ile = {np.mean(np.array(c_hi_lo_spread))}')
    print(f'Mean uncorrected std. dev. = {np.mean(np.array(u_std_devs))}')
    print(f'Mean corrected std. dev. = {np.mean(np.array(c_std_devs))}')
    f,ax=plt.subplots(figsize=(CONST.ILLUMINATION_VARIATION_PLOT_WIDTH,0.5*CONST.ILLUMINATION_VARIATION_PLOT_WIDTH))
    ax.plot(xaxis_vals,u_hi_lo_spread,marker='v',linestyle='dashed',linewidth=2,label='uncorrected 5th-95th %ile')
    ax.plot(xaxis_vals,c_hi_lo_spread,marker='^',linestyle='dashed',linewidth=2,label='corrected 5th-95th %ile')
    ax.plot(xaxis_vals,u_std_devs,marker='<',linestyle='dotted',linewidth=2,label='uncorrected std. dev.')
    ax.plot(xaxis_vals,c_std_devs,marker='>',linestyle='dotted',linewidth=2,label='corrected std. dev.')
    ax.set_ylim(0,(ax.get_ylim())[-1]*1.25)
    ax.set_title('illumination variation reduction by layer')
    ax.set_xlabel('image layer')
    ax.set_ylabel('relative flux')
    ax.legend(loc='best')
    if iv_plot_name is not None :
        plt.savefig(iv_plot_name)
        plt.close()
        cropAndOverwriteImage(iv_plot_name)
    else :
        plt.show()
    if iv_csv_name is not None :
        with open(iv_csv_name,'w') as ivrcfp :
            ivrcfp.write('layer,uncorrected_5th_to_95th_percentile,corrected_5th_to_95th_percentile,uncorrected_std_dev,corrected_std_dev\n')
            for li,(uhls,chls,usd,csd) in enumerate(zip(u_hi_lo_spread,c_hi_lo_spread,u_std_devs,c_std_devs),start=1) :
                ivrcfp.write(f'{li:d},{uhls:.5f},{chls:.5f},{usd:.5f},{csd:.5f}\n')

#helper function to #plot the max, min, 10th/90th %ile, mean, and std. dev. of the thresholds for each layer
def slideBackgroundThresholdsPlot(flatfield_top_dir,nlayers,savename=None) :
    #get all the background thresholds for every slide by layer
    all_bgts_by_layer = [[] for _ in range(nlayers)]
    slide_names = []
    with cd(os.path.join(flatfield_top_dir,CONST.THRESHOLDING_PLOT_DIR_NAME)) :
        all_threshold_fns = glob.glob(f'*_{UNIV_CONST.BACKGROUND_THRESHOLD_TEXT_FILE_NAME_STEM}')
        for tfn in all_threshold_fns :
            sn = tfn.split('_')[0]
            slide_names.append(sn)
            with open(tfn,'r') as fp :
                vals=[int(l.rstrip()) for l in fp.readlines() if l!='']
                for li in range(nlayers) :
                    all_bgts_by_layer[li].append(vals[li])
    xvals=list(range(1,nlayers+1))
    maxvals=[]; minvals=[]; lowvals=[]; hivals=[]; meanvals=[]; medianvals=[]; stddevs=[]
    for li in range(nlayers) :
        all_layer_thresholds=[all_bgts_by_layer[li][si] for si in range(len(all_bgts_by_layer[li]))]
        all_layer_thresholds.sort()
        maxvals.append(max(all_layer_thresholds)); minvals.append(min(all_layer_thresholds))
        hivals.append(all_layer_thresholds[int(0.9*len(all_layer_thresholds))]); lowvals.append(all_layer_thresholds[int(0.1*len(all_layer_thresholds))])
        meanvals.append(statistics.mean(all_layer_thresholds))
        medianvals.append(statistics.median(all_layer_thresholds))
        stddevs.append(statistics.pstdev(all_layer_thresholds))
    #make the figure
    f,ax=plt.subplots(figsize=(16.8,7.2))
    #plot the standard deviation fills
    for li in range(nlayers) :
        boxlow=meanvals[li]-stddevs[li]
        boxhi=meanvals[li]+stddevs[li]
        if li==0 :
            ax.fill([li+0.5,li+0.5,li+1.5,li+1.5],[boxlow,boxhi,boxhi,boxlow],'skyblue',alpha=0.5,label='std. dev.')
        else :
            ax.fill([li+0.5,li+0.5,li+1.5,li+1.5],[boxlow,boxhi,boxhi,boxlow],'skyblue',alpha=0.5)
    #plot the other statistics for each filter region
    if nlayers==35 :
        LAST_FILTER_LAYERS = [lg[1] for lg in UNIV_CONST.LAYER_GROUPS_35[:-1]]  
        microscope_name_stem = 'Vectra 3.0'
    elif nlayers==43 :
        LAST_FILTER_LAYERS = [lg[1] for lg in UNIV_CONST.LAYER_GROUPS_43[:-1]]
        microscope_name_stem = 'Vectra Polaris'
    else :
        raise ValueError(f'ERROR: number of layers {nlayers} is not a recognized option!') 
    for i in range(len(LAST_FILTER_LAYERS)+1) :
        f_i = 0 if i==0 else LAST_FILTER_LAYERS[i-1]
        l_i = xvals[-1] if i==len(LAST_FILTER_LAYERS) else LAST_FILTER_LAYERS[i]
        if i==0 :
            ax.plot(xvals[f_i:l_i],lowvals[f_i:l_i],color='limegreen',marker='v',linestyle='dashed',alpha=0.5,label='10th %ile')
            ax.plot(xvals[f_i:l_i],hivals[f_i:l_i],color='orchid',marker='^',linestyle='dashed',alpha=0.5,label='90th %ile')
            ax.plot(xvals[f_i:l_i],minvals[f_i:l_i],color='darkgreen',marker='v',label='minimum')
            ax.plot(xvals[f_i:l_i],maxvals[f_i:l_i],color='darkmagenta',marker='^',label='maximum')
            ax.plot(xvals[f_i:l_i],medianvals[f_i:l_i],color='darkorange',marker='s',linewidth=2,label='median')
            ax.plot(xvals[f_i:l_i],meanvals[f_i:l_i],color='k',marker='o',linewidth=2,label='mean')
            ax.plot([l_i+0.5,l_i+0.5],[2,max(maxvals)+2],color='dimgrey',linewidth=4,linestyle='dotted',label='filter changeover')
        else :
            ax.plot(xvals[f_i:l_i],lowvals[f_i:l_i],color='limegreen',marker='v',linestyle='dashed',alpha=0.5)
            ax.plot(xvals[f_i:l_i],hivals[f_i:l_i],color='orchid',marker='^',linestyle='dashed',alpha=0.5)
            ax.plot(xvals[f_i:l_i],minvals[f_i:l_i],color='darkgreen',marker='v')
            ax.plot(xvals[f_i:l_i],maxvals[f_i:l_i],color='darkmagenta',marker='^')
            ax.plot(xvals[f_i:l_i],medianvals[f_i:l_i],color='darkorange',marker='s',linewidth=2)
            ax.plot(xvals[f_i:l_i],meanvals[f_i:l_i],color='k',marker='o',linewidth=2)
            if i!=len(LAST_FILTER_LAYERS) :
                ax.plot([l_i+0.5,l_i+0.5],[2,max(maxvals)+2],color='dimgrey',linewidth=4,linestyle='dotted')
    #reset the axis limits and add the legend
    ax.set_xlim(0,nlayers+1)
    ax.set_ylim(0,ax.get_ylim()[1])
    ax.legend(loc='best',fontsize=13)
    ax.set_xlabel('image layer',fontsize=14)
    ax.set_ylabel('threshold flux',fontsize=14)
    ax.set_title(f'background thresholds found for {len(slide_names)} {microscope_name_stem} slides',fontsize=14)
    if savename is not None :
        plt.savefig(savename)
        plt.close()
        cropAndOverwriteImage(savename)
    else :
        plt.show()

def maskStackEdgeVsCentralRegion(mask_stack,savename=None) :
    nlayers=mask_stack.shape[-1]
    if nlayers==35 :
        LAST_FILTER_LAYERS = [lg[1] for lg in UNIV_CONST.LAYER_GROUPS_35[:-1]] 
    elif nlayers==43 :
        LAST_FILTER_LAYERS = [lg[1] for lg in UNIV_CONST.LAYER_GROUPS_43[:-1]]
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
    for i in range(len(LAST_FILTER_LAYERS)+1) :
        f_i = 0 if i==0 else LAST_FILTER_LAYERS[i-1]
        l_i = xaxis_vals[-1] if i==len(LAST_FILTER_LAYERS) else LAST_FILTER_LAYERS[i]
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
            if i!=len(LAST_FILTER_LAYERS) :
                plt.plot([l_i+0.5,l_i+0.5],[totalmin-0.01,totalmax+0.01],color='black',linewidth=2,linestyle='dotted')
    plt.title('relative spread in number of images stacked per layer for whole and central 64% images',fontsize=14)
    plt.xlabel('layer number',fontsize=14)
    #fix the range on the x-axis to accommodate the legend
    plt.xlim(0,nlayers+(10 if nlayers==35 else 12))
    plt.ylabel('number of images stacked relative to layer mean',fontsize=14)
    plt.legend(loc='best')
    if savename is not None :
        plt.savefig(savename)
        plt.close()
        cropAndOverwriteImage(savename)
    else :
        plt.show()
    u_hi_lo_spread = [u_highs[li]-u_lows[li] for li in range(nlayers)]
    c_hi_lo_spread = [c_highs[li]-c_lows[li] for li in range(nlayers)]
    print(f'Mean whole image 5th-95th %ile = {np.mean(np.array(u_hi_lo_spread))}')
    print(f'Mean central 64% 5th-95th %ile = {np.mean(np.array(c_hi_lo_spread))}')
    print(f'Mean whole image std. dev. = {np.mean(np.array(u_std_devs))}')
    print(f'Mean central 64% std. dev. = {np.mean(np.array(c_std_devs))}')

#helper function to write out a sheet of masking information plots for an image
def doMaskingPlotsForImage(image_key,tissue_mask,plot_dict_lists,compressed_full_mask,workingdir=None) :
    #figure out how many rows/columns will be in the sheet and set up the plots
    n_rows = len(plot_dict_lists)+1
    n_cols = max(n_rows,len(plot_dict_lists[0]))
    for pdi in range(1,len(plot_dict_lists)) :
        if len(plot_dict_lists[pdi]) > n_cols :
            n_cols = len(plot_dict_lists[pdi])
    f,ax = plt.subplots(n_rows,n_cols,figsize=(n_cols*6.4,n_rows*tissue_mask.shape[0]/tissue_mask.shape[1]*6.4))
    #add the masking plots for each layer group
    for row,plot_dicts in enumerate(plot_dict_lists) :
        for col,pd in enumerate(plot_dicts) :
            dkeys = pd.keys()
            #imshow plots
            if 'image' in dkeys :
                #edit the overlay based on the full mask
                if 'title' in dkeys and 'overlay (clipped)' in pd['title'] :
                    pd['image'][:,:,1][compressed_full_mask[:,:,row+1]>1]=pd['image'][:,:,0][compressed_full_mask[:,:,row+1]>1]
                    pd['image'][:,:,0][(compressed_full_mask[:,:,row+1]>1) & (pd['image'][:,:,2]!=0)]=0
                    pd['image'][:,:,2][(compressed_full_mask[:,:,row+1]>1) & (pd['image'][:,:,2]!=0)]=0
                imshowkwargs = {}
                possible_keys = ['cmap','vmin','vmax']
                for pk in possible_keys :
                    if pk in dkeys :
                        imshowkwargs[pk]=pd[pk]
                pos = ax[row][col].imshow(pd['image'],**imshowkwargs)
                f.colorbar(pos,ax=ax[row][col])
                if 'title' in dkeys :
                    title_text = pd['title'].replace('IMAGE',image_key)
                    ax[row][col].set_title(title_text)
            #histogram plots
            elif 'hist' in dkeys :
                binsarg=100
                logarg=False
                if 'bins' in dkeys :
                    binsarg=pd['bins']
                if 'log_scale' in dkeys :
                    logarg=pd['log_scale']
                ax[row][col].hist(pd['hist'],binsarg,log=logarg)
                if 'xlabel' in dkeys :
                    xlabel_text = pd['xlabel'].replace('IMAGE',image_key)
                    ax[row][col].set_xlabel(xlabel_text)
                if 'line_at' in dkeys :
                    ax[row][col].plot([pd['line_at'],pd['line_at']],
                                    [0.8*y for y in ax[row][col].get_ylim()],
                                    linewidth=2,color='tab:red',label=pd['line_at'])
                    ax[row][col].legend(loc='best')
            #bar plots
            elif 'bar' in dkeys :
                ax[row][col].bar(pd['bins'][:-1],pd['bar'],width=1.0)
                if 'xlabel' in dkeys :
                    xlabel_text = pd['xlabel'].replace('IMAGE',image_key)
                    ax[row][col].set_xlabel(xlabel_text)
                if 'line_at' in dkeys :
                    ax[row][col].plot([pd['line_at'],pd['line_at']],
                                    [0.8*y for y in ax[row][col].get_ylim()],
                                    linewidth=2,color='tab:red',label=pd['line_at'])
                    ax[row][col].legend(loc='best')
            #remove axes from any extra slots
            if col==len(plot_dicts)-1 and col<n_cols-1 :
                for ci in range(col+1,n_cols) :
                    ax[row][ci].axis('off')
    #add the plots of the full mask layer groups
    enumerated_mask_max = np.max(compressed_full_mask)
    for lgi in range(compressed_full_mask.shape[-1]-1) :
        pos = ax[n_rows-1][lgi].imshow(compressed_full_mask[:,:,lgi+1],vmin=0.,vmax=enumerated_mask_max,cmap='rainbow')
        f.colorbar(pos,ax=ax[n_rows-1][lgi])
        ax[n_rows-1][lgi].set_title(f'full mask, layer group {lgi+1}')
    #empty the other unused axes in the last row
    for ci in range(n_rows-1,n_cols) :
        ax[n_rows-1][ci].axis('off')
    #show/save the plot
    if workingdir is None :
        plt.show()
    else :
        with cd(workingdir) :
            fn = f'{image_key}_masking_plots.png'
            plt.savefig(fn); plt.close(); cropAndOverwriteImage(fn)

#helper function to plot all of the hpf locations for a slide with their reasons for being flagged
def plotFlaggedHPFLocations(sid,all_rfps,rfps_added,lmrs,plotdir_path=None) :
    all_flagged_hpf_keys = [lmr.image_key for lmr in lmrs]
    hpf_identifiers = []
    for rfp in all_rfps :
        key = (os.path.basename(rfp)).rstrip(UNIV_CONST.RAW_EXT)
        key_x = float(key.split(',')[0].split('[')[1])
        key_y = float(key.split(',')[1].split(']')[0])
        if key in all_flagged_hpf_keys :
            key_strings = set([lmr.reason_flagged for lmr in lmrs if lmr.image_key==key])
            blur_flagged = 1 if CONST.BLUR_FLAG_STRING in key_strings else 0
            saturation_flagged = 1 if CONST.SATURATION_FLAG_STRING in key_strings else 0
            flagged_int = 1*blur_flagged+2*saturation_flagged
        else :
            flagged_int = 0
        #elif rfp in rfps_added :
        #    flagged_int = 0
        #else :
        #    flagged_int = 4
        hpf_identifiers.append({'x':key_x,'y':key_y,'flagged':flagged_int})
    colors_by_flag_int = ['gray','royalblue','gold','limegreen','black']
    labels_by_flag_int = ['not flagged','blur flagged','saturation flagged','blur and saturation','not read/stacked']
    w = max([identifier['x'] for identifier in hpf_identifiers])-min([identifier['x'] for identifier in hpf_identifiers])
    h = max([identifier['y'] for identifier in hpf_identifiers])-min([identifier['y'] for identifier in hpf_identifiers])
    if h>w :
        f,ax = plt.subplots(figsize=(((1.1*w)/(1.1*h))*9.6,9.6))
    else :
        f,ax = plt.subplots(figsize=(9.6,9.6*((1.1*h)/(1.1*w))))
    for i in range(len(colors_by_flag_int)) :
        hpf_ids_to_plot = [identifier for identifier in hpf_identifiers if identifier['flagged']==i]
        if len(hpf_ids_to_plot)<1 :
            continue
        ax.scatter([hpfid['x'] for hpfid in hpf_ids_to_plot],
                   [hpfid['y'] for hpfid in hpf_ids_to_plot],
                   marker='o',
                   color=colors_by_flag_int[i],
                   label=labels_by_flag_int[i])
    ax.set_xlim(ax.get_xlim()[0]-0.05*w,ax.get_xlim()[1]+0.05*w)
    ax.set_ylim(ax.get_ylim()[0]-0.05*h,ax.get_ylim()[1]+0.05*h)
    ax.invert_yaxis()
    title_text = f'{sid} HPF center locations, ({len(all_rfps)} in slide, {len(rfps_added)} read, {len([hpfid for hpfid in hpf_identifiers if hpfid["flagged"] not in (0,4)])} flagged)'
    ax.set_title(title_text,fontsize=16)
    ax.legend(loc='best',fontsize=10)
    ax.set_xlabel('HPF local x position',fontsize=16)
    ax.set_ylabel('HPF local y position',fontsize=16)
    fn = f'{sid}_flagged_hpf_locations.png'
    if plotdir_path is not None :
        with cd(plotdir_path) :
            plt.savefig(fn)
            cropAndOverwriteImage(fn)
    else :
        plt.show()

