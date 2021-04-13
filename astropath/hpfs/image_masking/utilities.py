#imports
from .config import CONST
from ...utilities.dataclasses import MyDataClass
import numpy as np
import os, cv2

#mask region information helper class
class LabelledMaskRegion(MyDataClass) :
    image_key          : str
    region_index       : int
    layers             : str
    n_pixels           : int
    reason_flagged     : str

#helper function to unpack, reshape, and return a tissue mask from the packed mask file
def unpackTissueMask(filepath,dimensions) :
    if not os.path.isfile(filepath) :
        raise ValueError(f'ERROR: tissue mask file {filepath} does not exist!')
    packed_mask = np.memmap(filepath,dtype=np.uint8,mode='r')
    return (np.unpackbits(packed_mask)).reshape(dimensions)

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

#return a binary mask with all of the areas smaller than min_size removed
def getSizeFilteredMask(mask,min_size,both=True,invert=False) :
    """
    both   = True if regions of ones and zeros should both be filtered (just recursively calls the function with !invert)
    invert = True if regions of zeros instead of ones should be filtered
    """
    if invert :
        mask = (np.where(mask==1,0,1)).astype(mask.dtype)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    sizes = stats[1:, -1]; nb_components = nb_components - 1
    new_mask = np.zeros_like(mask)
    for i in range(0, nb_components):
        if sizes[i] >= min_size :
            new_mask[output == i + 1] = 1
    if invert :
        new_mask = (np.where(new_mask==1,0,1)).astype(mask.dtype)
    if both :
        return getSizeFilteredMask(new_mask,min_size,both=False,invert=(not invert))
    return new_mask

#return a binary mask with any areas that are already flagged in a prior mask removed
def getExclusiveMask(mask_to_check,prior_mask,min_independent_pixel_frac,invert=True) :
    prior_mask_hot = 1
    if invert :
        mask_to_check = (np.where(mask_to_check==1,0,1)).astype(mask_to_check.dtype)
        prior_mask_hot = 0
    n_regions, regions_im = cv2.connectedComponents(mask_to_check)
    new_mask = np.zeros_like(mask_to_check)
    for region_i in range(1,n_regions) :
        total_region_size = np.sum(regions_im==region_i)
        n_already_selected_pixels = np.sum(prior_mask[regions_im==region_i]==prior_mask_hot)
        if (total_region_size-n_already_selected_pixels)/total_region_size < min_independent_pixel_frac :
            continue
        new_mask[regions_im==region_i] = mask_to_check[regions_im==region_i]
    if invert :
        new_mask = (np.where(new_mask==1,0,1)).astype(mask_to_check.dtype)
    return new_mask

#erode a binary mask near a corresponding tissue mask and filter for size
def getMorphedAndFilteredMask(mask,tissue_mask,min_pixels,min_size) :
    if np.min(mask)<1 and np.max(mask)!=np.min(mask) :
        #a window-sized open incorporating the tissue mask to get rid of any remaining thin borders
        mask_to_transform = cv2.UMat(np.where((mask==0) | (tissue_mask==0),0,1).astype(mask.dtype))
        twice_eroded_fold_mask = cv2.UMat(np.empty_like(mask))
        cv2.morphologyEx(mask,cv2.MORPH_ERODE,CONST.WINDOW_EL,twice_eroded_fold_mask,iterations=2,borderType=cv2.BORDER_REPLICATE)
        cv2.morphologyEx(mask_to_transform,cv2.MORPH_ERODE,CONST.MEDIUM_CO_EL,mask_to_transform,borderType=cv2.BORDER_REPLICATE)
        cv2.morphologyEx(mask_to_transform,cv2.MORPH_OPEN,CONST.WINDOW_EL,mask_to_transform,borderType=cv2.BORDER_REPLICATE)
        cv2.morphologyEx(mask_to_transform,cv2.MORPH_DILATE,CONST.MEDIUM_CO_EL,mask_to_transform,borderType=cv2.BORDER_REPLICATE)
        mask_to_transform = mask_to_transform.get()
        twice_eroded_fold_mask = twice_eroded_fold_mask.get()
        mask[(mask==1) & (tissue_mask==1) & (twice_eroded_fold_mask==0)] = mask_to_transform[(mask==1) & (tissue_mask==1) & (twice_eroded_fold_mask==0)]
        #remove any remaining small spots after the tissue mask incorporation
        mask = getSizeFilteredMask(mask,min_size)
        #make sure there are at least the minimum number of pixels selected
        if np.sum(mask==0)<min_pixels :
            return np.ones_like(mask)
    return mask

#function to compute and return the variance of the normalized laplacian for a given image layer
def getImageLayerLocalVarianceOfNormalizedLaplacian(img_layer) :
    #build the laplacian image and normalize it to get the curvature
    img_laplacian = cv2.Laplacian(img_layer,cv2.CV_32F,borderType=cv2.BORDER_REFLECT)
    img_lap_norm = cv2.UMat(np.empty(img_layer.shape,dtype=np.float32))
    cv2.filter2D(img_layer,cv2.CV_32F,CONST.LOCAL_MEAN_KERNEL,img_lap_norm,borderType=cv2.BORDER_REFLECT)
    img_lap_norm = img_lap_norm.get()
    img_norm_lap = img_laplacian
    img_norm_lap[img_lap_norm!=0] /= img_lap_norm[img_lap_norm!=0]
    img_norm_lap[img_lap_norm==0] = 0
    #find the variance of the normalized laplacian in the neighborhood window
    norm_lap_loc_mean = cv2.UMat(np.empty_like(img_norm_lap))
    cv2.filter2D(img_norm_lap,cv2.CV_32F,CONST.WINDOW_EL,norm_lap_loc_mean,borderType=cv2.BORDER_REFLECT)
    norm_lap_2_loc_mean = cv2.UMat(np.empty_like(img_norm_lap))
    cv2.filter2D(np.power(img_norm_lap,2),cv2.CV_32F,CONST.WINDOW_EL,norm_lap_2_loc_mean,borderType=cv2.BORDER_REFLECT)
    npix = np.sum(CONST.WINDOW_EL)
    norm_lap_loc_mean = norm_lap_loc_mean.get()
    norm_lap_2_loc_mean = norm_lap_2_loc_mean.get()
    norm_lap_loc_mean /= npix
    norm_lap_2_loc_mean /= npix
    local_norm_lap_var = np.abs(norm_lap_2_loc_mean-np.power(norm_lap_loc_mean,2))
    #return the local variance of the normalized laplacian
    return local_norm_lap_var

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
        img_nlv_loc_mean = cv2.UMat(np.empty_like(img_nlv))
        cv2.filter2D(img_nlv,cv2.CV_32F,CONST.SMALLER_WINDOW_EL,img_nlv_loc_mean,borderType=cv2.BORDER_REFLECT)
        img_nlv_loc_mean=img_nlv_loc_mean.get()
        img_nlv_loc_mean/=np.sum(CONST.SMALLER_WINDOW_EL)
        #threshold on the local variance of the normalized laplacian and the local mean of those values to make a binary mask
        layer_mask = (np.where((img_nlv>nlv_cut) | (img_nlv_loc_mean>max_mean),1,0)).astype(np.uint8)
        if np.min(layer_mask) != np.max(layer_mask) :
            #convert to UMat
            layer_mask = cv2.UMat(layer_mask)
            #small open/close to refine it
            cv2.morphologyEx(layer_mask,cv2.MORPH_OPEN,CONST.SMALL_CO_EL,layer_mask,borderType=cv2.BORDER_REPLICATE)
            cv2.morphologyEx(layer_mask,cv2.MORPH_CLOSE,CONST.SMALL_CO_EL,layer_mask,borderType=cv2.BORDER_REPLICATE)
            #erode by the smaller window element
            cv2.morphologyEx(layer_mask,cv2.MORPH_ERODE,CONST.SMALLER_WINDOW_EL,layer_mask,borderType=cv2.BORDER_REPLICATE)
            layer_mask = layer_mask.get()
        #add it to the stack 
        stacked_masks+=layer_mask
    #determine the final mask for this group by thresholding on how many individual layers contribute
    group_blur_mask = (np.where(stacked_masks>n_layers_flag_cut,1,0)).astype(np.uint8)    
    if np.min(group_blur_mask) != np.max(group_blur_mask) :
        #medium sized open/close to refine it
        group_blur_mask = cv2.UMat(group_blur_mask)
        cv2.morphologyEx(group_blur_mask,cv2.MORPH_OPEN,CONST.MEDIUM_CO_EL,group_blur_mask,borderType=cv2.BORDER_REPLICATE)
        cv2.morphologyEx(group_blur_mask,cv2.MORPH_CLOSE,CONST.MEDIUM_CO_EL,group_blur_mask,borderType=cv2.BORDER_REPLICATE)
        group_blur_mask = group_blur_mask.get()
    #set up the plots to return
    if return_plots :
        plot_img_layer = img_array[:,:,brightest_layer_n-1]
        sorted_pil = np.sort(plot_img_layer[group_blur_mask==1].flatten())
        pil_max = sorted_pil[int(0.95*len(sorted_pil))]; pil_min = sorted_pil[0]
        norm = 255./(pil_max-pil_min)
        im_c = (np.clip(norm*(plot_img_layer-pil_min),0,255)).astype(np.uint8)
        overlay_c = np.array([im_c,im_c*group_blur_mask,im_c*group_blur_mask]).transpose(1,2,0)
        plots = [{'image':plot_img_layer,'title':f'raw IMAGE layer {brightest_layer_n}'},
                 {'image':overlay_c,'title':f'layer {layer_group_bounds[0]}-{layer_group_bounds[1]} blur mask overlay (clipped)'}]
        if exp_times is not None and ethistandbins is not None :
            plots.append({'bar':ethistandbins[0],'bins':ethistandbins[1],
                          'xlabel':f'layer {layer_group_bounds[0]}-{layer_group_bounds[1]} exposure times (ms)',
                          'line_at':exp_times[layer_group_bounds[0]-1]})
        plots.append({'image':brightest_layer_nlv,'title':'local variance of normalized laplacian'})
        plots.append({'hist':brightest_layer_nlv.flatten(),'xlabel':'variance of normalized laplacian','line_at':nlv_cut})
        plots.append({'image':stacked_masks,'title':f'stacked layer masks (cut at {n_layers_flag_cut})','cmap':'gist_ncar',
                      'vmin':0,'vmax':layer_group_bounds[1]-layer_group_bounds[0]+1})
    else :
        plots = None
    #return the blur mask for the layer group and the list of plot dictionaries
    return group_blur_mask, plots

#return the tissue fold mask for an image combining information from all layer groups
def getImageTissueFoldMask(sm_img_array,exp_times,tissue_mask,exp_t_hists,layer_groups,brightest_layers,dapi_lgi,rbc_lgi,nlv_cuts,nlv_max_means,return_plots=False) :
    #make the list of flag cuts (how many layers can miss being flagged to include the region in the layer group mask)
    fold_flag_cuts = []
    for lgi,lgb in enumerate(layer_groups) :
        if lgi==dapi_lgi or lgi==rbc_lgi :
            fold_flag_cuts.append(3)
        elif lgb[1]-lgb[0]<3 :
            fold_flag_cuts.append(0)
        else :
            fold_flag_cuts.append(1)
    #get the fold masks for each layer group
    fold_masks_by_layer_group = []; fold_mask_plots_by_layer_group = []
    for lgi,lgb in enumerate(layer_groups) :
        lgeth = exp_t_hists[lgi] if exp_t_hists is not None else None
        lgtfm, lgtfmps = getImageLayerGroupBlurMask(sm_img_array,
                                                    exp_times,
                                                    lgb,
                                                    nlv_cuts[lgi],
                                                    fold_flag_cuts[lgi],
                                                    nlv_max_means[lgi],
                                                    brightest_layers[lgi],
                                                    lgeth,
                                                    return_plots)
        fold_masks_by_layer_group.append(lgtfm)
        fold_mask_plots_by_layer_group.append(lgtfmps)
    #combine the layer group blur masks to get the final mask for all layers
    stacked_fold_masks = np.zeros_like(fold_masks_by_layer_group[0])
    for lgi,layer_group_fold_mask in enumerate(fold_masks_by_layer_group) :
        stacked_fold_masks[layer_group_fold_mask==0]+=10 if lgi in (dapi_lgi,rbc_lgi) else 1
    overall_fold_mask = (np.where(stacked_fold_masks>11,0,1)).astype(np.uint8)
    #morph and filter the mask using the common operations
    overall_fold_mask = getMorphedAndFilteredMask(overall_fold_mask,tissue_mask,CONST.FOLD_MIN_PIXELS,CONST.FOLD_MIN_SIZE)
    if return_plots :
        return overall_fold_mask,fold_mask_plots_by_layer_group
    else :
        return overall_fold_mask,None