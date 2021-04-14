#imports
from astropath.hpfs.flatfield.utilities import chunkListOfFilepaths, readImagesMT
from astropath.hpfs.flatfield.image_mask import getSizeFilteredMask, getImageTissueMask, getExclusiveMask
from astropath.hpfs.flatfield.image_mask import getMorphedAndFilteredMask, getImageLayerLocalVarianceOfNormalizedLaplacian
from astropath.hpfs.flatfield.config import CONST
from astropath.utilities.img_file_io import getImageHWLFromXMLFile, getSlideMedianExposureTimesByLayer, LayerOffset
from astropath.utilities.img_file_io import smoothImageWorker, getExposureTimesByLayer
#from astropath.utilities.img_file_io import writeImageToFile
from astropath.utilities.tableio import readtable, writetable
from astropath.utilities.misc import cd, addCommonArgumentsToParser, cropAndOverwriteImage
#from astropath.utilities import units
#from astropath.baseclasses.csvclasses import constantsdict
from astropath.utilities.dataclasses import MyDataClass
from argparse import ArgumentParser
import numpy as np, matplotlib.pyplot as plt, multiprocessing as mp
import logging, pathlib, glob, cv2

#constants
RAWFILE_EXT                = '.Data.dat'
LOCAL_MEAN_KERNEL          = np.array([[0.0,0.2,0.0],
                                       [0.2,0.2,0.2],
                                       [0.0,0.2,0.0]])
WINDOW_EL                  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(45,45))
SMALLER_WINDOW_EL          = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
SMALLER_WINDOW_N_PIXELS    = np.sum(SMALLER_WINDOW_EL)
#MASK_LAYER_GROUPS          = [(1,9),(10,18),(19,25),(26,32),(33,35)]
MASK_LAYER_GROUPS           = [(1,9),(10,11),(12,17),(18,20),(21,29),(30,36),(37,43)]
#BRIGHTEST_LAYERS           = [5,11,21,29,34]
BRIGHTEST_LAYERS           = [5,10,16,19,22,31,41]
DAPI_LAYER_GROUP_INDEX     = 0
#RBC_LAYER_GROUP_INDEX      = 1
RBC_LAYER_GROUP_INDEX      = 4
FOLD_MIN_PIXELS            = 30000
FOLD_MIN_SIZE              = 5000
#FOLD_NLV_CUT               = 0.0035
FOLD_NLV_CUTS_BY_LAYER_GROUP  = [0.12, 0.0, 0.07, 0.07, 0.12, 0.10,  0.10]
#FOLD_MAX_MEAN              = 0.0030
FOLD_MAX_MEANS_BY_LAYER_GROUP = [0.10, 0.0, 0.06, 0.06, 0.10, 0.085, 0.085]
#FOLD_MASK_FLAG_CUTS        = [3,3,1,1,0]
FOLD_MASK_FLAG_CUTS        = [3,0,1,0,3,1,1]
FOLD_FLAG_STRING           = 'tissue fold or bright dust'
DUST_MIN_PIXELS            = 30000
DUST_MIN_SIZE              = 20000
#DUST_NLV_CUT               = 0.00085
DUST_NLV_CUT               = 0.07
#DUST_MAX_MEAN              = 0.00065
DUST_MAX_MEAN              = 0.0525
DUST_STRING                = 'likely dust'
SATURATION_MIN_PIXELS      = 4500
SATURATION_MIN_SIZE        = 1000
#SATURATION_INTENSITY_CUTS  = [100,100,250,400,150]
SATURATION_INTENSITY_CUTS  = [100,50,250,250,150,100,50]
SATURATION_FLAG_STRING     = 'saturated likely skin red blood cells or stain'

#logger
logger = logging.getLogger("flagging_hpf_regions")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s  [%(funcName)s]","%Y-%m-%d %H:%M:%S"))
logger.addHandler(handler)

#mask region information helper class
class LabelledMaskRegion(MyDataClass) :
    image_key          : str
    cellview_x         : float
    cellview_y         : float
    region_index       : int
    layers             : str
    n_pixels           : int
    reason_flagged     : str

#################### IMAGE HANDLING UTILITIY FUNCTIONS ####################

#function to return a blur mask for a given image layer group, along with a dictionary of plots to add to the group for this image
def getImageLayerGroupBlurMask(img_array,exp_times,layer_group_bounds,nlv_cut,n_layers_flag_cut,max_mean,brightest_layer_n,ethistandbins,return_plots=True) :
    #start by making a mask for every layer in the group
    stacked_masks = np.zeros(img_array.shape[:-1],dtype=np.uint8)
    brightest_layer_nlv = None
    for ln in range(layer_group_bounds[0],layer_group_bounds[1]+1) :
        #get the local variance of the normalized laplacian image
        img_nlv = getImageLayerLocalVarianceOfNormalizedLaplacian(img_array[:,:,ln-1])
        if ln==brightest_layer_n :
            brightest_layer_nlv = img_nlv
        #get the mean of those local normalized laplacian variance values in the window size
        img_nlv_loc_mean = (cv2.filter2D(img_nlv,cv2.CV_32F,SMALLER_WINDOW_EL,borderType=cv2.BORDER_REFLECT))/SMALLER_WINDOW_N_PIXELS
        #threshold on the local variance of the normalized laplacian and the local mean of those values to make a binary mask
        layer_mask = (np.where((img_nlv>nlv_cut) | (img_nlv_loc_mean>max_mean),1,0)).astype(np.uint8)
        #small open/close to refine it
        layer_mask = (cv2.morphologyEx(layer_mask,cv2.MORPH_OPEN,CONST.SMALL_CO_EL,borderType=cv2.BORDER_REPLICATE))
        layer_mask = (cv2.morphologyEx(layer_mask,cv2.MORPH_CLOSE,CONST.SMALL_CO_EL,borderType=cv2.BORDER_REPLICATE))
        #erode by the smaller window element
        layer_mask = (cv2.morphologyEx(layer_mask,cv2.MORPH_ERODE,SMALLER_WINDOW_EL,borderType=cv2.BORDER_REPLICATE))
        #add it to the stack 
        stacked_masks+=layer_mask
    #determine the final mask for this group by thresholding on how many individual layers contribute
    group_blur_mask = (np.where(stacked_masks>n_layers_flag_cut,1,0)).astype(np.uint8)    
    #medium sized open/close to refine it
    group_blur_mask = (cv2.morphologyEx(group_blur_mask,cv2.MORPH_OPEN,CONST.MEDIUM_CO_EL,borderType=cv2.BORDER_REPLICATE))
    group_blur_mask = (cv2.morphologyEx(group_blur_mask,cv2.MORPH_CLOSE,CONST.MEDIUM_CO_EL,borderType=cv2.BORDER_REPLICATE))
    #set up the plots to return
    if return_plots :
        plot_img_layer = img_array[:,:,brightest_layer_n-1]
        #im_gs = (plot_img_layer*group_blur_mask).astype(np.float32); im_gs /= np.max(im_gs)
        #overlay_gs = np.array([im_gs,im_gs,0.15*group_blur_mask]).transpose(1,2,0)
        sorted_pil = np.sort(plot_img_layer[group_blur_mask==1].flatten())
        if len(sorted_pil)>0 :
            pil_max = sorted_pil[int(0.95*len(sorted_pil))]; pil_min = sorted_pil[0]
        else :
            pil_max=255; pil_min=0
        norm = 255./(pil_max-pil_min)
        im_c = (np.clip(norm*(plot_img_layer-pil_min),0,255)).astype(np.uint8)
        overlay_c = np.array([im_c,im_c*group_blur_mask,im_c*group_blur_mask]).transpose(1,2,0)
        plots = [{'image':plot_img_layer,'title':f'raw IMAGE layer {brightest_layer_n}'},
                 {'image':overlay_c,'title':f'layer {layer_group_bounds[0]}-{layer_group_bounds[1]} blur mask overlay (clipped)'},
                 #{'image':overlay_gs,'title':f'layer {layer_group_bounds[0]}-{layer_group_bounds[1]} blur mask overlay (grayscale)'},
                 {'bar':ethistandbins[0],'bins':ethistandbins[1],'xlabel':f'layer {layer_group_bounds[0]}-{layer_group_bounds[1]} exposure times (ms)','line_at':exp_times[layer_group_bounds[0]-1]},
                 {'image':brightest_layer_nlv,'title':'local variance of normalized laplacian'},
                 {'hist':brightest_layer_nlv.flatten(),'xlabel':'variance of normalized laplacian','line_at':nlv_cut},
                 {'image':stacked_masks,'title':f'stacked layer masks (cut at {n_layers_flag_cut})','cmap':'gist_ncar','vmin':0,'vmax':layer_group_bounds[1]-layer_group_bounds[0]+1},
                 #{'image':group_blur_mask,'title':f'layer {layer_group_bounds[0]}-{layer_group_bounds[1]} blur mask','vmin':0,'vmax':1},
                ]
    else :
        plots = None
    #return the blur mask for the layer group and the list of plot dictionaries
    return group_blur_mask, plots

#return the tissue fold mask for an image combining information from all layer groups
def getImageTissueFoldMask(img_array,exp_times,tissue_mask,exp_t_hists,return_plots=False) :
    ##smooth the image EXTREMELY gently
    #img_array = smoothImageWorker(img_array,1)
    #get the fold masks for each layer group
    fold_masks_by_layer_group = []; fold_mask_plots_by_layer_group = []
    for lgi,lgb in enumerate(MASK_LAYER_GROUPS) :
        lgtfm, lgtfmps = getImageLayerGroupBlurMask(img_array,
                                                    exp_times,
                                                    lgb,
                                                    FOLD_NLV_CUTS_BY_LAYER_GROUP[lgi],
                                                    FOLD_MASK_FLAG_CUTS[lgi],
                                                    FOLD_MAX_MEANS_BY_LAYER_GROUP[lgi],
                                                    BRIGHTEST_LAYERS[lgi],
                                                    exp_t_hists[lgi],
                                                    return_plots)
        fold_masks_by_layer_group.append(lgtfm)
        fold_mask_plots_by_layer_group.append(lgtfmps)
    #combine the layer group blur masks to get the final mask for all layers
    stacked_fold_masks = np.zeros_like(fold_masks_by_layer_group[0])
    for lgi,layer_group_fold_mask in enumerate(fold_masks_by_layer_group) :
        to_add = 1
        if lgi in (DAPI_LAYER_GROUP_INDEX,RBC_LAYER_GROUP_INDEX) :
            to_add = 10
        stacked_fold_masks[layer_group_fold_mask==0]+=to_add
    overall_fold_mask = (np.where(stacked_fold_masks>11,0,1)).astype(np.uint8)
    #morph and filter the mask using the common operations
    overall_fold_mask = getMorphedAndFilteredMask(overall_fold_mask,tissue_mask,FOLD_MIN_PIXELS,FOLD_MIN_SIZE)
    if return_plots :
        return overall_fold_mask,fold_mask_plots_by_layer_group
    else :
        return overall_fold_mask,None

#function to return a blur mask for a given image layer group, along with a dictionary of plots to add to the group for this image
def getImageLayerGroupSaturationMask(img_array,exp_times,layer_group_bounds,intensity_cut,n_layers_flag_cut,brightest_layer_n,ethistandbins,return_plots=True) :
    #normalize the image by its exposure time
    normalized_img_array = np.zeros(img_array.shape,dtype=np.float32)
    for li in range(img_array.shape[-1]) :
        normalized_img_array[:,:,li] = img_array[:,:,li]/exp_times[li]
    #smooth the exposure time-normalized image
    sm_n_img_array = smoothImageWorker(normalized_img_array,CONST.TISSUE_MASK_SMOOTHING_SIGMA)
    #then make a mask for every layer in the group
    stacked_masks = np.zeros(normalized_img_array.shape[:-1],dtype=np.uint8)
    for ln in range(layer_group_bounds[0],layer_group_bounds[1]+1) :
        #threshold the image layer to make a binary mask
        layer_mask = (np.where(sm_n_img_array[:,:,ln-1]>intensity_cut,0,1)).astype(np.uint8)
        #large close to keep only the biggest areas
        #layer_mask = (cv2.morphologyEx(layer_mask,cv2.MORPH_CLOSE,CONST.C3_EL,borderType=cv2.BORDER_REPLICATE))
        stacked_masks+=layer_mask
    #determine the final mask for this group by thresholding on how many individual layers contribute
    group_mask = (np.where(stacked_masks>n_layers_flag_cut,1,0)).astype(np.uint8)    
    #medium sized open/close to refine it
    group_mask = (cv2.morphologyEx(group_mask,cv2.MORPH_OPEN,CONST.MEDIUM_CO_EL,borderType=cv2.BORDER_REPLICATE))
    group_mask = (cv2.morphologyEx(group_mask,cv2.MORPH_CLOSE,CONST.MEDIUM_CO_EL,borderType=cv2.BORDER_REPLICATE))
    #filter the mask for the total number of pixels and regions by the minimum size
    group_mask = getSizeFilteredMask(group_mask,SATURATION_MIN_SIZE)
    if np.sum(group_mask==0)<SATURATION_MIN_PIXELS :
        group_mask = np.ones_like(group_mask)
    #set up the plots to return
    if return_plots :
        #plot_img_layer = sm_n_img_array[:,:,brightest_layer_n-1]
        #sorted_pil = np.sort(plot_img_layer[group_mask==1].flatten())
        #pil_max = sorted_pil[int(0.95*len(sorted_pil))]; pil_min = sorted_pil[0]
        #norm = 255./(pil_max-pil_min)
        #im_c = (np.clip(norm*(plot_img_layer-pil_min),0,255)).astype(np.uint8)
        #overlay_c = np.array([im_c,im_c*group_mask,im_c*group_mask]).transpose(1,2,0)
        plots = [#{'image':plot_img_layer,'title':f'smoothed normalized IMAGE layer {brightest_layer_n}'},
                 #{'image':overlay_c,'title':f'layer {layer_group_bounds[0]}-{layer_group_bounds[1]} saturation mask overlay (clipped)'},
                 {'hist':sm_n_img_array[:,:,brightest_layer_n-1].flatten(),'xlabel':'pixel intensity (counts/ms)','line_at':intensity_cut,'log_scale':True},
                 #{'bar':ethistandbins[0],'bins':ethistandbins[1],'xlabel':f'layer {layer_group_bounds[0]}-{layer_group_bounds[1]} exposure times (ms)','line_at':exp_times[layer_group_bounds[0]-1]},
                 #{'image':stacked_masks,'title':f'stacked layer masks (cut at {n_layers_flag_cut})','cmap':'gist_ncar','vmin':0,'vmax':layer_group_bounds[1]-layer_group_bounds[0]+1},
                 #{'image':group_mask,'title':f'layer {layer_group_bounds[0]}-{layer_group_bounds[1]} saturation mask','vmin':0,'vmax':1},
                ]
    else :
        plots = None
    #return the saturation mask for the layer group and the list of plot dictionaries
    return group_mask, plots

#################### HELPER FUNCTIONS ####################

#helper function to write out a sheet of masking information plots for an image
def doMaskingPlotsForImage(image_key,tissue_mask,plot_dict_lists,full_mask,workingdir=None) :
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
                    pd['image'][:,:,1][full_mask[:,:,MASK_LAYER_GROUPS[row][0]]>1]=pd['image'][:,:,0][full_mask[:,:,MASK_LAYER_GROUPS[row][0]]>1]
                    pd['image'][:,:,0][(full_mask[:,:,MASK_LAYER_GROUPS[row][0]]>1) & (pd['image'][:,:,2]!=0)]=0
                    pd['image'][:,:,2][(full_mask[:,:,MASK_LAYER_GROUPS[row][0]]>1) & (pd['image'][:,:,2]!=0)]=0
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
    #add the plot of the overlaid tissue and layer group masks
    #superimposed_masks = tissue_mask
    #max_sim_val = 1
    #for mci,plot_dicts in enumerate(plot_dict_lists) :
    #    for pd in plot_dicts :
    #        if 'image' in pd.keys() and 'title' in pd.keys() and pd['title'].endswith('blur mask') :
    #            superimposed_masks+=(2**(mci+1))*pd['image']
    #            max_sim_val+=(2**(mci+1))
    #pos = ax[n_rows-1][0].imshow(superimposed_masks,vmin=0.,vmax=max_sim_val,cmap='gist_ncar')
    #f.colorbar(pos,ax=ax[n_rows-1][0])
    #ax[n_rows-1][0].set_title('superimposed masks')
    #add the plots of the enumerated mask layer groups
    enumerated_mask_max = np.max(full_mask)
    for lgi,lgb in enumerate(MASK_LAYER_GROUPS) :
        pos = ax[n_rows-1][lgi].imshow(full_mask[:,:,lgb[0]-1],vmin=0.,vmax=enumerated_mask_max,cmap='rainbow')
        f.colorbar(pos,ax=ax[n_rows-1][lgi])
        ax[n_rows-1][lgi].set_title(f'full mask, layers {lgb[0]}-{lgb[1]}')
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
#def plotFlaggedHPFLocations(sid,all_rfps,all_lmrs,pscale,xpos,ypos,truncated,workingdir) :
def plotFlaggedHPFLocations(sid,all_rfps,all_lmrs,truncated,workingdir) :
    all_flagged_hpf_keys = [lmr.image_key for lmr in all_lmrs]
    hpf_identifiers = []
    for rfp in all_rfps :
        key = ((pathlib.Path(rfp)).name).rstrip(RAWFILE_EXT)
        key_x = float(key.split(',')[0].split('[')[1])
        key_y = float(key.split(',')[1].split(']')[0])
        #cvx = pscale*key_x-xpos
        #cvy = pscale*key_y-ypos
        cvx = key_x
        cvy = key_y
        if key in all_flagged_hpf_keys :
            key_strings = set([lmr.reason_flagged for lmr in all_lmrs if lmr.image_key==key])
            fold_flagged = 1 if FOLD_FLAG_STRING in key_strings else 0
            dust_flagged = 1 if DUST_STRING in key_strings else 0
            saturation_flagged = 1 if SATURATION_FLAG_STRING in key_strings else 0
            flagged_int = 1*fold_flagged+2*dust_flagged+4*saturation_flagged
        else :
            flagged_int = 0
        hpf_identifiers.append({'x':cvx,'y':cvy,'flagged':flagged_int})
    colors_by_flag_int = ['gray','royalblue','gold','limegreen','firebrick','mediumorchid','darkorange','aqua']
    labels_by_flag_int = ['not flagged','tissue fold flagged','single-layer dust flagged','dust and tissue folds',
                            'saturation flagged','saturation and tissue folds','saturation and dust','saturation and dust and tissue folds']
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
    title_text = f'{sid} HPF center locations, ({len([hpfid for hpfid in hpf_identifiers if hpfid["flagged"]!=0])} flagged out of {len(all_rfps)} read)'
    if truncated :
        title_text+=' (stopped early)'
    ax.set_title(title_text,fontsize=16)
    ax.legend(loc='best',fontsize=10)
    ax.set_xlabel('HPF CellView x position',fontsize=16)
    ax.set_ylabel('HPF CellView y position',fontsize=16)
    fn = f'{sid}_flagged_hpf_locations.png'
    with cd(workingdir) :
        plt.savefig(fn)
        cropAndOverwriteImage(fn)

#helper function to change a mask from zeroes and ones to region indices and zeroes
def getEnumeratedMask(layer_mask,start_i) :
    #first invert the mask to get the "bad" regions as "signal"
    inverted_mask = np.zeros_like(layer_mask); inverted_mask[layer_mask==0] = 1; inverted_mask[layer_mask==1] = 0
    #label each connected region uniquely starting at the supplied index
    n_labels, labels_im = cv2.connectedComponents(inverted_mask)
    return_mask = np.zeros_like(layer_mask)
    for label_i in range(1,n_labels) :
        return_mask[labels_im==label_i] = start_i+label_i-1
    #return the mask
    return return_mask

#helper function to calculate and add the subimage infos for a single image to a shared dictionary (run in parallel)
#def getLabelledMaskRegionsWorker(img_array,exposure_times,key,thresholds,xpos,ypos,pscale,workingdir,exp_time_hists,return_list) :
def getLabelledMaskRegionsWorker(img_array,exposure_times,key,thresholds,workingdir,exp_time_hists,return_list) :
    #start by creating the tissue mask
    tissue_mask = getImageTissueMask(img_array,thresholds)
    #next get the tissue fold mask and its associated plots
    tissue_fold_mask,tissue_fold_plots_by_layer_group = getImageTissueFoldMask(img_array,exposure_times,tissue_mask,exp_time_hists,return_plots=True)
    #tissue_fold_mask,tissue_fold_plots_by_layer_group = np.ones(img_array.shape,dtype=np.uint8),None
    #get masks for the blurriest areas of the DAPI layer group
    #sm_img_array = smoothImageWorker(img_array,1)
    sm_img_array = img_array
    dapi_dust_mask,dapi_dust_plots = getImageLayerGroupBlurMask(sm_img_array,
                                                                exposure_times,
                                                                MASK_LAYER_GROUPS[DAPI_LAYER_GROUP_INDEX],
                                                                DUST_NLV_CUT,
                                                                0.5*(MASK_LAYER_GROUPS[DAPI_LAYER_GROUP_INDEX][1]-MASK_LAYER_GROUPS[DAPI_LAYER_GROUP_INDEX][0]+1),
                                                                DUST_MAX_MEAN,
                                                                BRIGHTEST_LAYERS[DAPI_LAYER_GROUP_INDEX],
                                                                exp_time_hists[DAPI_LAYER_GROUP_INDEX],
                                                                False)
    #same morphology transformations as before
    dapi_dust_mask = getMorphedAndFilteredMask(dapi_dust_mask,tissue_mask,DUST_MIN_PIXELS,DUST_MIN_SIZE)
    #make sure any regions in that mask are sufficiently exclusive w.r.t. what's already flagged
    dapi_dust_mask = getExclusiveMask(dapi_dust_mask,tissue_fold_mask,0.25)
    #dapi_dust_mask,dapi_dust_plots = np.ones(img_array.shape,dtype=np.uint8),None
    #get masks for the saturated regions in each layer group
    layer_group_saturation_masks = []; layer_group_saturation_mask_plots = []
    for lgi,lgb in enumerate(MASK_LAYER_GROUPS) :
        lgsm,saturation_mask_plots = getImageLayerGroupSaturationMask(img_array,
                                                                      exposure_times,
                                                                      lgb,
                                                                      SATURATION_INTENSITY_CUTS[lgi],
                                                                      (MASK_LAYER_GROUPS[lgi][1]-MASK_LAYER_GROUPS[lgi][0]),
                                                                      BRIGHTEST_LAYERS[lgi],
                                                                      exp_time_hists[lgi],
                                                                      return_plots=True)
        #lgsm,saturation_mask_plots = np.ones(img_array.shape,dtype=np.uint8), None
        layer_group_saturation_masks.append(lgsm)
        layer_group_saturation_mask_plots.append(saturation_mask_plots)
    #if there is anything flagged in the final masks, write out some plots and the mask file/csv file lines
    is_masked = np.min(tissue_fold_mask)<1 or np.min(dapi_dust_mask)<1
    if not is_masked :
        for lgsm in layer_group_saturation_masks :
            if np.min(lgsm)<1 :
                is_masked=True
                break
    if is_masked :
        #figure out where the image is in cellview
        key_x = float(key.split(',')[0].split('[')[1])
        key_y = float(key.split(',')[1].split(']')[0])
        #cvx = pscale*key_x-xpos
        #cvy = pscale*key_y-ypos
        cvx = key_x
        cvy = key_y
        #the mask starts as all ones (0=background, 1=good tissue, >=2 is a flagged region)
        output_mask = np.ones(img_array.shape,dtype=np.uint8)
        #add in the tissue fold mask, starting with index 2
        start_i = 2
        if np.min(tissue_fold_mask)<1 :
            layers_string = '-1'
            enumerated_fold_mask = getEnumeratedMask(tissue_fold_mask,start_i)
            for li in range(img_array.shape[-1]) :
                output_mask[:,:,li] = np.where(enumerated_fold_mask!=0,enumerated_fold_mask,output_mask[:,:,li])
            start_i = np.max(enumerated_fold_mask)+1
            region_indices = list(range(np.min(enumerated_fold_mask[enumerated_fold_mask!=0]),np.max(enumerated_fold_mask)+1))
            for ri in region_indices :
                r_size = np.sum(enumerated_fold_mask==ri)
                return_list.append(LabelledMaskRegion(key,cvx,cvy,ri,layers_string,r_size,FOLD_FLAG_STRING))
        #add in the mask for the dapi layer group
        if np.min(dapi_dust_mask)<1 :
            layers_string = f'{MASK_LAYER_GROUPS[DAPI_LAYER_GROUP_INDEX][0]}-{MASK_LAYER_GROUPS[DAPI_LAYER_GROUP_INDEX][1]}'
            enumerated_dapi_mask = getEnumeratedMask(dapi_dust_mask,start_i)
            for ln in range(MASK_LAYER_GROUPS[DAPI_LAYER_GROUP_INDEX][0],MASK_LAYER_GROUPS[DAPI_LAYER_GROUP_INDEX][1]+1) :
                output_mask[:,:,ln-1] = np.where(enumerated_dapi_mask!=0,enumerated_dapi_mask,output_mask[:,:,ln-1])
            start_i = np.max(enumerated_dapi_mask)+1
            region_indices = list(range(np.min(enumerated_dapi_mask[enumerated_dapi_mask!=0]),np.max(enumerated_dapi_mask)+1))
            for ri in region_indices :
                r_size = np.sum(enumerated_dapi_mask==ri)
                return_list.append(LabelledMaskRegion(key,cvx,cvy,ri,layers_string,r_size,DUST_STRING))
        #add in the saturation masks 
        for lgi,lgsm in enumerate(layer_group_saturation_masks) :
            if np.min(lgsm)<1 :
                layers_string = f'{MASK_LAYER_GROUPS[lgi][0]}-{MASK_LAYER_GROUPS[lgi][1]}'
                enumerated_sat_mask = getEnumeratedMask(lgsm,start_i)
                for ln in range(MASK_LAYER_GROUPS[lgi][0],MASK_LAYER_GROUPS[lgi][1]+1) :
                    output_mask[:,:,ln-1] = np.where(enumerated_sat_mask!=0,enumerated_sat_mask,output_mask[:,:,ln-1])
                start_i = np.max(enumerated_sat_mask)+1
                region_indices = list(range(np.min(enumerated_sat_mask[enumerated_sat_mask!=0]),np.max(enumerated_sat_mask)+1))
                for ri in region_indices :
                    r_size = np.sum(enumerated_sat_mask==ri)
                    return_list.append(LabelledMaskRegion(key,cvx,cvy,ri,layers_string,r_size,SATURATION_FLAG_STRING))
        #finally add in the tissue mask (all the background is zero in every layer, unless already flagged otherwise)
        for li in range(img_array.shape[-1]) :
            output_mask[:,:,li] = np.where(output_mask[:,:,li]==1,tissue_mask,output_mask[:,:,li])
        #make and write out the plots for this image
        all_plot_dict_lists = []
        if tissue_fold_plots_by_layer_group is not None :
            all_plot_dict_lists += tissue_fold_plots_by_layer_group
        for lgi,saturation_mask_plot_dicts in enumerate(layer_group_saturation_mask_plots) :
            if saturation_mask_plot_dicts is not None :
                all_plot_dict_lists[lgi].append(saturation_mask_plot_dicts[0])
        if dapi_dust_plots is not None :
            all_plot_dict_lists += [dapi_dust_plots]
        #sat_plots = []
        #for saturation_mask_plot_dicts in layer_group_saturation_mask_plots :
        #    if saturation_mask_plot_dicts is not None :
        #        sat_plots.append(saturation_mask_plot_dicts)
        #all_plot_dict_lists += sat_plots
        doMaskingPlotsForImage(key,tissue_mask,all_plot_dict_lists,output_mask,workingdir)
        ##write out the mask in the working directory
        #with cd(workingdir) :
        #    writeImageToFile(output_mask,f'{key}_mask.png',dtype=np.uint8)

#helper function to get a list of all the labelled mask regions for a chunk of files
#def getLabelledMaskRegionsForChunk(fris,metsbl,etcobl,thresholds,xpos,ypos,pscale,workingdir,root_dir,exp_time_hists) :
def getLabelledMaskRegionsForChunk(fris,metsbl,etcobl,thresholds,workingdir,root_dir,exp_time_hists) :
    #get the image arrays
    img_arrays = readImagesMT(fris,med_exposure_times_by_layer=metsbl,et_corr_offsets_by_layer=etcobl)
    #get all of the labelled mask region objects as the images are masked
    manager = mp.Manager()
    return_list = manager.list()
    procs = []
    for i,im_array in enumerate(img_arrays) :
        msg = f'Masking {fris[i].rawfile_path} {fris[i].sequence_print}'
        logger.info(msg)
        exp_times = getExposureTimesByLayer(fris[i].rawfile_path,root_dir)
        key = ((pathlib.Path(fris[i].rawfile_path)).name).rstrip(RAWFILE_EXT)
        #p = mp.Process(target=getLabelledMaskRegionsWorker,args=(im_array,exp_times,key,thresholds,xpos,ypos,pscale,workingdir,exp_time_hists,return_list))
        p = mp.Process(target=getLabelledMaskRegionsWorker,args=(im_array,exp_times,key,thresholds,workingdir,exp_time_hists,return_list))
        procs.append(p)
        p.start()
    for proc in procs:
        proc.join()
    #return just a regular list of all the labelled region info objects
    return list(return_list)

#helper function to make sure all the necessary information is available from the command line arguments
def checkArgs(args) :
    #rawfile_top_dir/[slideID] must exist
    rawfile_dir = pathlib.Path(f'{args.rawfile_top_dir}/{args.slideID}')
    if not pathlib.Path.is_dir(rawfile_dir) :
        raise ValueError(f'ERROR: rawfile directory {rawfile_dir} does not exist!')
    #root dir must exist
    if not pathlib.Path.is_dir(pathlib.Path(args.root_dir)) :
        raise ValueError(f'ERROR: root_dir argument ({args.root_dir}) does not point to a valid directory!')
    #images must be corrected for exposure time, and exposure time correction file must exist
    if (args.skip_exposure_time_correction) :   
        raise ValueError('ERROR: exposure time offset file must be provided.')
    if not pathlib.Path.is_file(pathlib.Path(args.exposure_time_offset_file)) :
        raise ValueError(f'ERROR: exposure_time_offset_file {args.exposure_time_offset_file} does not exist!')
    #need the threshold file
    if args.threshold_file_dir is None :
        raise ValueError('ERROR: must provide a threshold file dir.')
    tfp = pathlib.Path(f'{args.threshold_file_dir}/{args.slideID}_background_thresholds.txt')
    if not pathlib.Path.is_file(tfp) :
        raise ValueError(f'ERROR: threshold file path {tfp} does not exist!')
    #create the working directory if it doesn't already exist
    if not pathlib.Path.is_dir(pathlib.Path(args.workingdir)) :
        pathlib.Path.mkdir(pathlib.Path(args.workingdir))

#################### MAIN SCRIPT ####################

def main(args=None) :
    parser = ArgumentParser()
    #add the common options to the parser
    addCommonArgumentsToParser(parser,flatfielding=False,warping=False)
    #threshold file directory
    parser.add_argument('--threshold_file_dir',
                        help='Path to the directory holding the slide [slideID]_background_thresholds.txt file')
    #group for other run options
    run_option_group = parser.add_argument_group('run options', 'other options for this run')
    run_option_group.add_argument('--n_threads', default=10, type=int,
                                  help='Maximum number of threads to use in reading/processing files')
    run_option_group.add_argument('--max_files', default=-1, type=int,
                                  help='Maximum number of files to use (default = -1 runs all files for the requested slide)')
    run_option_group.add_argument('--max_masks', default=-1, type=int,
                                  help='Maximum number of masks to write out (prevents writing too much while testing, default=-1 writes out everything)')
    args = parser.parse_args(args=args)
    #check the arguments
    checkArgs(args)
    #get all the rawfile paths
    with cd(pathlib.Path(f'{args.rawfile_top_dir}/{args.slideID}')) :
        all_rfps = [pathlib.Path(f'{args.rawfile_top_dir}/{args.slideID}/{fn}') for fn in glob.glob(f'*{RAWFILE_EXT}')]
    #get the correction details and other slide information stuff
    dims   = getImageHWLFromXMLFile(args.root_dir,args.slideID)
    all_exp_times = []
    for lgi in range(len(MASK_LAYER_GROUPS)) :
        all_exp_times.append([])
    for rfp in all_rfps :
        etsbl = getExposureTimesByLayer(rfp,args.root_dir)
        for lgi,lgb in enumerate(MASK_LAYER_GROUPS) :
            all_exp_times[lgi].append(etsbl[lgb[0]-1])
    exp_time_hists = []
    for lgi in range(len(MASK_LAYER_GROUPS)) :
        newhist,newbins = np.histogram(all_exp_times[lgi],bins=100)
        exp_time_hists.append((newhist,newbins))
    metsbl = getSlideMedianExposureTimesByLayer(args.root_dir,args.slideID)
    etcobl = []
    for li in range(dims[-1]) :
        etcobl.append(None)
    layer_offsets_from_file = readtable(args.exposure_time_offset_file,LayerOffset)
    for ln in range(1,len(etcobl)+1) :
        this_layer_offset = [lo.offset for lo in layer_offsets_from_file if lo.layer_n==ln]
        if len(this_layer_offset)==1 :
            etcobl[ln-1]=this_layer_offset[0]
        elif len(this_layer_offset)==0 :
            msg = f'WARNING: LayerOffset file {args.exposure_time_offset_file} does not have an entry for layer {ln}'
            msg+=  ', so that offset will be set to zero!'
            logger.warning(msg)
            etcobl[ln-1]=0.
        else :
            raise RuntimeError(f'ERROR: more than one entry found in LayerOffset file {args.exposure_time_offset_file} for layer {ln}!')
    with open(pathlib.Path(f'{args.threshold_file_dir}/{args.slideID}_background_thresholds.txt')) as fp :
        bgtbl = [int(v) for v in fp.readlines() if v!='']
    if len(bgtbl)!=dims[-1] :
        raise RuntimeError(f'ERROR: found {len(bgtbl)} background thresholds but images have {dims[-1]} layers!')
    thresholds = [bgtbl[li] for li in range(dims[-1])]
    #slide_constants_dict = constantsdict(pathlib.Path(f'{args.root_dir}/{args.slideID}/dbload/{args.slideID}_constants.csv'))
    #xpos = float(units.pixels(slide_constants_dict['xposition']))
    #ypos = float(units.pixels(slide_constants_dict['yposition']))
    #pscale = slide_constants_dict['pscale']
    #chunk up the rawfile read information 
    if args.max_files!=-1 :
        all_rfps = all_rfps[:args.max_files]
    fri_chunks = chunkListOfFilepaths(all_rfps,dims,args.root_dir,args.n_threads)
    #get the masked regions for each chunk
    truncated=False
    all_lmrs = []
    for fri_chunk in fri_chunks :
        #new_lmrs = getLabelledMaskRegionsForChunk(fri_chunk,metsbl,etcobl,thresholds,xpos,ypos,pscale,args.workingdir,args.root_dir,exp_time_hists)
        new_lmrs = getLabelledMaskRegionsForChunk(fri_chunk,metsbl,etcobl,thresholds,args.workingdir,args.root_dir,exp_time_hists)
        all_lmrs+=new_lmrs
        if args.max_masks!=-1 and len(set(lmr.image_key for lmr in all_lmrs))>=args.max_masks :
            truncated=True
            break
    #write out the table with all of the labelled region information
    if len(all_lmrs)>0 :
        fn = f'{args.slideID}_labelled_mask_regions.csv'
        with cd(args.workingdir) :
            writetable(fn,sorted(all_lmrs,key=lambda x: f'{x.image_key}_{x.region_index}'))
    #make the plot of all the HPF locations and whether they have something masked/flagged
    #plotFlaggedHPFLocations(args.slideID,all_rfps,all_lmrs,pscale,xpos,ypos,truncated,args.workingdir)
    plotFlaggedHPFLocations(args.slideID,all_rfps,all_lmrs,truncated,args.workingdir)

if __name__=='__main__' :
    main()
