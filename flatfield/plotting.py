#imports
from .config import CONST
from ..utilities.misc import cd, cropAndOverwriteImage
import numpy as np, matplotlib.pyplot as plt
import os, glob, statistics

#helper function to plot the max/min, 5th/95th %ile, and std. dev. of a flatfield image's correction factors by layer 
def flatfieldImagePixelIntensityPlot(flatfield_image,savename=None) :
    #figure out the number of layers and the filter breaks
    nlayers=flatfield_image.shape[-1]
    if nlayers==35 :
        LAST_FILTER_LAYERS = CONST.LAST_FILTER_LAYERS_35 
    elif nlayers==43 :
        LAST_FILTER_LAYERS = CONST.LAST_FILTER_LAYERS_43
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
        LAST_FILTER_LAYERS = CONST.LAST_FILTER_LAYERS_35 
    elif nlayers==43 :
        LAST_FILTER_LAYERS = CONST.LAST_FILTER_LAYERS_43
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
        all_threshold_fns = glob.glob(f'*_{CONST.THRESHOLD_TEXT_FILE_NAME_STEM}')
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
        LAST_FILTER_LAYERS = CONST.LAST_FILTER_LAYERS_35 
        microscope_name_stem = 'Vectra 3.0'
    elif nlayers==43 :
        LAST_FILTER_LAYERS = CONST.LAST_FILTER_LAYERS_43
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
        LAST_FILTER_LAYERS = CONST.LAST_FILTER_LAYERS_35 
    elif nlayers==43 :
        LAST_FILTER_LAYERS = CONST.LAST_FILTER_LAYERS_43
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
