#imports
from .config import CONST
from ...utilities.dataclasses import MyDataClass
import numpy as np
import pathlib, cv2

#mask region information helper class
class LabelledMaskRegion(MyDataClass) :
    image_key          : str
    region_index       : int
    layers             : str
    n_pixels           : int
    reason_flagged     : str

def return_new_mask_labelled_regions(im_array,im_key,bg_thresholds,norm_ets,savedir=None) :
    """
    Create an ImageMask, write out the files it creates, and return its list of labelled mask regions
    This function writes out the ImageMask it creates and doesn't return it in order to have the lowest possible memory footprint
    And it also doesn't depend on passing any complex objects outside of it
    Both of which are useful for running many iterations of the function in parallel processes
    
    arguments are the same as ImageMask.__init__
    """
    mask = ImageMask(im_array,im_key,bg_thresholds,norm_ets)
    mask.save_mask_files(savedir)
    return mask.labelled_mask_regions

def save_plots_for_image(im_array,im_key,bg_thresholds,norm_ets,orig_ets,exp_time_hists_and_bins,savedir) :
    """
    Create the masks for a given image and write out plots of the process
    Useful if all you care about is getting the plots
    This function also has the lowest possible memory footprint/simplest possible I/O to be run in parallel processes

    arguments are the same as ImageMask.__init__ + ImageMask.save_plots
    """
    mask = ImageMask(im_array,im_key,bg_thresholds,norm_ets)
    mask.save_plots(orig_ets,exp_time_hists_and_bins,savedir=None)

#helper function to unpack, reshape, and return a tissue mask from its packed mask file
def unpack_tissue_mask(filepath,dimensions) :
    if not pathlib.Path(filepath).is_file() :
        raise ValueError(f'ERROR: tissue mask file {filepath} does not exist!')
    packed_mask = np.memmap(filepath,dtype=np.uint8,mode='r')
    return (np.unpackbits(packed_mask)).reshape(dimensions)

#helper function to change a mask from zeroes and ones to region indices and zeroes
def get_enumerated_mask(layer_mask,start_i) :
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
def get_size_filtered_mask(mask,min_size,both=True,invert=False) :
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
def get_exclusive_mask(mask_to_check,prior_mask,min_independent_pixel_frac,invert=True) :
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
def get_morphed_and_filtered_mask(mask,tissue_mask,min_pixels,min_size) :
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
def get_image_layer_local_variance_of_normalized_laplacian(img_layer) :
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
