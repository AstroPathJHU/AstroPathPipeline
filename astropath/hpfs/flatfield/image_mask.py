#imports
from .utilities import flatfield_logger
from .plotting import doMaskingPlotsForImage
from .config import CONST
from ...utilities.dataclasses import MyDataClass
from ...utilities.img_file_io import smoothImageWorker, getExposureTimesByLayer, im3writeraw, writeImageToFile
from ...utilities.misc import cd
from ...utilities.config import CONST as UNIV_CONST
import numpy as np
import os, cv2

#mask region information helper class
class LabelledMaskRegion(MyDataClass) :
    image_key          : str
    region_index       : int
    layers             : str
    n_pixels           : int
    reason_flagged     : str

class ImageMask() :
    """
    Class to store and work with a mask for an image
    """

    #################### PROPERTIES ####################

    @property
    def packed_tissue_mask(self):
        if self._tissue_mask is None :
            raise RuntimeError('ERROR: pasked_tissue_mask called before setting a tissue mask')
        return np.packbits(self._tissue_mask)
    @property
    def compressed_mask(self): #the compressed mask with (# layers groups)+1 layers
        return self._compressed_mask
    @property
    def onehot_mask(self) : #the mask with good tissue=1, everything else=0
        uncompressed_full_mask = self.uncompressed_full_mask
        return (np.where(uncompressed_full_mask==1,1,0)).astype(np.uint8)
    @property
    def uncompressed_full_mask(self): #the uncompressed mask with the real number of image layers
        if self._compressed_mask is None :
            raise RuntimeError('ERROR: uncompressed_full_mask called without first creating a mask!')
        uncompressed_mask = np.ones((*self._compressed_mask.shape[:-1],self._layer_groups[-1][1]),dtype=np.uint8)
        for lgi,lgb in enumerate(self._layer_groups) :
            for ln in range(lgb[0],lgb[1]+1) :
                uncompressed_mask[:,:,ln-1] = self._compressed_mask[:,:,lgi+1]
        return uncompressed_mask
    @property
    def labelled_mask_regions(self):
        return self._labelled_mask_regions #the list of labelled mask region objects for this mask

    #################### PUBLIC FUNCTIONS ####################

    def __init__(self,image_key) :
        """
        image_key = the filename (minus extension) of the file whose mask this is
        """
        self._image_key = image_key
        #start the list of LabelledMaskRegion objects for this mask
        self._labelled_mask_regions = []
        #initialize the compresssed mask and some associated info as None for now
        self._layer_groups = None
        self._compressed_mask = None
        self._tissue_mask = None

    def addCreatedMasks(self,tissue_mask,blur_mask,saturation_masks) :
        """
        tissue_mask      = the tissue (1) vs. background (0) mask that should be added to the file
        blur_mask        = the blurred region mask (1=not blurred, 0=blurred) that should be added to the file
        saturation_masks = the list of saturation masks (1=not saturated, 0=saturated) to add to the file, one per broadband filter layer group
        """
        #set the tissue mask
        self._tissue_mask = tissue_mask
        #figure out the layer groups to use from the dimensions of the masks passed in
        if len(saturation_masks)==len(UNIV_CONST.LAYER_GROUPS_35) :
            self._layer_groups = UNIV_CONST.LAYER_GROUPS_35
        elif len(saturation_masks)==len(UNIV_CONST.LAYER_GROUPS_43) :
            self._layer_groups = UNIV_CONST.LAYER_GROUPS_43
        else :
            raise RuntimeError(f'ERROR: no defined list of broadband filter breaks for images with {len(saturation_masks)} layer groups!')
        #make the compressed mask, which has (# of layer groups)+1 layers total
        #the first layer holds just the tissue and blur masks; the other layers have the tissue and blur masks plus the saturation mask for each layer group
        self._compressed_mask = np.ones((*tissue_mask.shape,len(self._layer_groups)+1),dtype=np.uint8)
        #add in the blur mask, starting with index 2
        start_i = 2
        if np.min(blur_mask)<1 :
            layers_string = 'all'
            enumerated_blur_mask = getEnumeratedMask(blur_mask,start_i)
            for li in range(self._compressed_mask.shape[-1]) :
                self._compressed_mask[:,:,li][enumerated_blur_mask!=0] = enumerated_blur_mask[enumerated_blur_mask!=0]
            start_i = np.max(enumerated_blur_mask)+1
            region_indices = list(range(np.min(enumerated_blur_mask[enumerated_blur_mask!=0]),np.max(enumerated_blur_mask)+1))
            for ri in region_indices :
                r_size = np.sum(enumerated_blur_mask==ri)
                self._labelled_mask_regions.append(LabelledMaskRegion(self._image_key,ri,layers_string,r_size,CONST.BLUR_FLAG_STRING))
        #add in the saturation masks 
        for lgi,lgsm in enumerate(saturation_masks) :
            if np.min(lgsm)<1 :
                layers_string = f'{self._layer_groups[lgi][0]}-{self._layer_groups[lgi][1]}'
                enumerated_sat_mask = getEnumeratedMask(lgsm,start_i)
                self._compressed_mask[:,:,lgi+1][enumerated_sat_mask!=0] = enumerated_sat_mask[enumerated_sat_mask!=0]
                start_i = np.max(enumerated_sat_mask)+1
                region_indices = list(range(np.min(enumerated_sat_mask[enumerated_sat_mask!=0]),np.max(enumerated_sat_mask)+1))
                for ri in region_indices :
                    r_size = np.sum(enumerated_sat_mask==ri)
                    self._labelled_mask_regions.append(LabelledMaskRegion(self._image_key,ri,layers_string,r_size,CONST.SATURATION_FLAG_STRING))
        #finally add in the tissue mask
        for li in range(self._compressed_mask.shape[-1]) :
            self._compressed_mask[:,:,li][self._compressed_mask[:,:,li]==1] = tissue_mask[self._compressed_mask[:,:,li]==1]

#################### FILE-SCOPE HELPER FUNCTIONS ####################

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

#helper function to unpack, reshape, and return a tissue mask from the packed mask file
def unpackTissueMask(filepath,dimensions) :
    if not os.path.isfile(filepath) :
        raise ValueError(f'ERROR: tissue mask file {filepath} does not exist!')
    packed_mask = np.memmap(filepath,dtype=np.uint8,mode='r')
    return (np.unpackbits(packed_mask)).reshape(dimensions)

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

#return the fully-determined single tissue mask for a multilayer image
def getImageTissueMask(image_arr,bkg_thresholds) :
    #figure out where the layer groups are
    if image_arr.shape[-1]==35 :
        mask_layer_groups=UNIV_CONST.LAYER_GROUPS_35
        dapi_layer_group_index=UNIV_CONST.DAPI_LAYER_GROUP_INDEX_35
    elif image_arr.shape[-1]==43 :
        mask_layer_groups=UNIV_CONST.LAYER_GROUPS_43
        dapi_layer_group_index=UNIV_CONST.DAPI_LAYER_GROUP_INDEX_43
    else :
        raise RuntimeError(f'ERROR: no defined list of broadband filter breaks for images with {image_arr.shape[-1]} layers!')
    #smooth the entire image
    sm_image_array = smoothImageWorker(image_arr,CONST.TISSUE_MASK_SMOOTHING_SIGMA)
    #mask each layer individually first
    layer_masks = []
    for li in range(image_arr.shape[-1]) :
        #threshold
        layer_mask = (np.where(sm_image_array[:,:,li]>bkg_thresholds[li],1,0)).astype(np.uint8)
        #convert to UMat to use on the GPU
        layer_mask = cv2.UMat(layer_mask)
        #small close/open
        cv2.morphologyEx(layer_mask,cv2.MORPH_CLOSE,CONST.SMALL_CO_EL,layer_mask,borderType=cv2.BORDER_REPLICATE)
        cv2.morphologyEx(layer_mask,cv2.MORPH_OPEN,CONST.SMALL_CO_EL,layer_mask,borderType=cv2.BORDER_REPLICATE)
        layer_masks.append(layer_mask.get())
    #find the well-defined tissue and background in each layer group
    overall_tissue_mask = np.zeros_like(layer_masks[0])
    overall_background_mask = np.zeros_like(layer_masks[0])
    total_stacked_masks = np.zeros_like(layer_masks[0])
    #for each layer group
    for lgi,lgb in enumerate(mask_layer_groups) :
        stacked_masks = np.zeros_like(layer_masks[0])
        for ln in range(lgb[0],lgb[1]+1) :
            stacked_masks+=layer_masks[ln-1]
        total_stacked_masks+=stacked_masks
        #well-defined tissue is anything called tissue in at least all but two layers
        overall_tissue_mask[stacked_masks>(lgb[1]-lgb[0]-1)]+= 10 if lgi==dapi_layer_group_index else 1
        #well-defined background is anything called background in at least half the layers
        overall_background_mask[stacked_masks<(lgb[1]-lgb[0]+1)/2.]+= 10 if lgi==dapi_layer_group_index else 1
    #threshold tissue/background masks to include only those from the DAPI and at least one other layer group
    overall_tissue_mask = (np.where(overall_tissue_mask>10,1,0)).astype(np.uint8)
    overall_background_mask = (np.where(overall_background_mask>10,1,0)).astype(np.uint8)
    #final mask has tissue=1, background=0
    final_mask = np.zeros_like(layer_masks[0])+2
    final_mask[overall_tissue_mask==1] = 1
    final_mask[overall_background_mask==1] = 0
    #anything left over is signal if it's stacked in at least 60% of the total number of layers
    thresholded_stacked_masks = np.where(total_stacked_masks>(0.6*image_arr.shape[-1]),1,0)
    final_mask[final_mask==2] = thresholded_stacked_masks[final_mask==2]
    if np.min(final_mask) != np.max(final_mask) :
        #filter the tissue and background portions to get rid of the small islands
        final_mask = getSizeFilteredMask(final_mask,min_size=CONST.TISSUE_MIN_SIZE)
        #convert to UMat
        final_mask = cv2.UMat(final_mask)
        #medium size close/open to smooth out edges
        cv2.morphologyEx(final_mask,cv2.MORPH_CLOSE,CONST.MEDIUM_CO_EL,final_mask,borderType=cv2.BORDER_REPLICATE)
        cv2.morphologyEx(final_mask,cv2.MORPH_OPEN,CONST.MEDIUM_CO_EL,final_mask,borderType=cv2.BORDER_REPLICATE)
        final_mask = final_mask.get()
    return final_mask

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
                 {'image':overlay_c,'title':f'layer {layer_group_bounds[0]}-{layer_group_bounds[1]} blur mask overlay (clipped)'},
                 {'bar':ethistandbins[0],'bins':ethistandbins[1],'xlabel':f'layer {layer_group_bounds[0]}-{layer_group_bounds[1]} exposure times (ms)','line_at':exp_times[layer_group_bounds[0]-1]},
                 {'hist':brightest_layer_nlv.flatten(),'xlabel':'variance of normalized laplacian','line_at':nlv_cut},
                 {'image':stacked_masks,'title':f'stacked layer masks (cut at {n_layers_flag_cut})','cmap':'gist_ncar','vmin':0,'vmax':layer_group_bounds[1]-layer_group_bounds[0]+1},
                ]
    else :
        plots = None
    #return the blur mask for the layer group and the list of plot dictionaries
    return group_blur_mask, plots

#return the tissue fold mask for an image combining information from all layer groups
def getImageTissueFoldMask(sm_img_array,exp_times,tissue_mask,exp_t_hists,mask_layer_groups,brightest_layers,dapi_lgi,rbc_lgi,return_plots=False) :
    #make the list of flag cuts (how many layers can miss being flagged to include the region in the layer group mask)
    fold_flag_cuts = []
    for lgi,lgb in enumerate(mask_layer_groups) :
        if lgi==dapi_lgi or lgi==rbc_lgi :
            fold_flag_cuts.append(3)
        elif lgb[1]-lgb[0]<3 :
            fold_flag_cuts.append(0)
        else :
            fold_flag_cuts.append(1)
    #get the fold masks for each layer group
    fold_masks_by_layer_group = []; fold_mask_plots_by_layer_group = []
    for lgi,lgb in enumerate(mask_layer_groups) :
        lgtfm, lgtfmps = getImageLayerGroupBlurMask(sm_img_array,
                                                    exp_times,
                                                    lgb,
                                                    CONST.FOLD_NLV_CUT,
                                                    fold_flag_cuts[lgi],
                                                    CONST.FOLD_MAX_MEAN,
                                                    brightest_layers[lgi],
                                                    exp_t_hists[lgi],
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

#return the single blur mask for an image
def getImageBlurMask(img_array,exp_times,tissue_mask,exp_time_hists,return_plots=False) :
    #figure out where the layer groups are, etc.
    if img_array.shape[-1]==35 :
        mask_layer_groups=UNIV_CONST.LAYER_GROUPS_35
        brightest_layers=UNIV_CONST.BRIGHTEST_LAYERS_35
        dapi_layer_group_index=UNIV_CONST.DAPI_LAYER_GROUP_INDEX_35
        rbc_layer_group_index=UNIV_CONST.RBC_LAYER_GROUP_INDEX_35
    elif img_array.shape[-1]==43 :
        mask_layer_groups=UNIV_CONST.LAYER_GROUPS_43
        brightest_layers=UNIV_CONST.BRIGHTEST_LAYERS_43
        dapi_layer_group_index=UNIV_CONST.DAPI_LAYER_GROUP_INDEX_43
        rbc_layer_group_index=UNIV_CONST.RBC_LAYER_GROUP_INDEX_43
    else :
        raise RuntimeError(f'ERROR: no defined list of broadband filter breaks for images with {img_array.shape[-1]} layers!')
    #smooth the image array for both the tissue fold and DAPI layer blur detection
    sm_img_array = smoothImageWorker(img_array,CONST.BLUR_MASK_SMOOTHING_SIGMA)
    #get the tissue fold mask and its associated plots
    tissue_fold_mask,tissue_fold_plots_by_layer_group = getImageTissueFoldMask(sm_img_array,exp_times,tissue_mask,exp_time_hists,mask_layer_groups,
                                                                               brightest_layers,dapi_layer_group_index,rbc_layer_group_index,return_plots)
    #get masks for the blurriest areas of the DAPI layer group
    dapi_dust_mask,dapi_dust_plots = getImageLayerGroupBlurMask(sm_img_array,
                                                                exp_times,
                                                                mask_layer_groups[dapi_layer_group_index],
                                                                CONST.DUST_NLV_CUT,
                                                                0.5*(mask_layer_groups[dapi_layer_group_index][1]-mask_layer_groups[dapi_layer_group_index][0]+1),
                                                                CONST.DUST_MAX_MEAN,
                                                                brightest_layers[dapi_layer_group_index],
                                                                exp_time_hists[dapi_layer_group_index],
                                                                False)
    #same morphology transformations as for the multilayer blur masks
    dapi_dust_mask = getMorphedAndFilteredMask(dapi_dust_mask,tissue_mask,CONST.DUST_MIN_PIXELS,CONST.DUST_MIN_SIZE)
    #make sure any regions in that mask are sufficiently exclusive w.r.t. what's already flagged
    dapi_dust_mask = getExclusiveMask(dapi_dust_mask,tissue_fold_mask,0.25)
    #combine the multilayer and single layer blur masks into one by multiplying them together
    final_blur_mask = tissue_fold_mask*dapi_dust_mask
    #return the blur mask and the list of plot dicts 
    return final_blur_mask,tissue_fold_plots_by_layer_group

#return the list of saturation masks by layer group for a given image
def getImageSaturationMasks(image_arr,norm_ets) :
    #figure out where the layer groups are, etc.
    if image_arr.shape[-1]==35 :
        mask_layer_groups=UNIV_CONST.LAYER_GROUPS_35
        saturation_intensity_cuts=CONST.SATURATION_INTENSITY_CUTS_35
    elif image_arr.shape[-1]==43 :
        mask_layer_groups=UNIV_CONST.LAYER_GROUPS_43
        saturation_intensity_cuts=CONST.SATURATION_INTENSITY_CUTS_43
    else :
        raise RuntimeError(f'ERROR: no defined list of broadband filter breaks for images with {image_arr.shape[-1]} layers!')
    #normalize the image by its exposure time
    normalized_image_arr = np.zeros(image_arr.shape,dtype=np.float32)
    for li in range(image_arr.shape[-1]) :
        normalized_image_arr[:,:,li] = image_arr[:,:,li]/norm_ets[li]
    #smooth the exposure time-normalized image
    sm_n_image_arr = smoothImageWorker(normalized_image_arr,CONST.TISSUE_MASK_SMOOTHING_SIGMA)
    #make masks for each layer group
    layer_group_saturation_masks = []
    for lgi,lgb in enumerate(mask_layer_groups) :
        stacked_masks = np.zeros(sm_n_image_arr.shape[:-1],dtype=np.uint8)
        for ln in range(lgb[0],lgb[1]+1) :
            #threshold the image layer to make a binary mask
            layer_mask = (np.where(sm_n_image_arr[:,:,ln-1]>saturation_intensity_cuts[lgi],0,1)).astype(np.uint8)
            stacked_masks+=layer_mask
        #the final mask is anything flagged in ANY layer
        group_mask = (np.where(stacked_masks>lgb[1]-lgb[0],1,0)).astype(np.uint8)    
        if np.min(group_mask)!=np.max(group_mask) :
            #convert to UMat
            group_mask = cv2.UMat(group_mask)
            #medium sized open/close to refine it
            cv2.morphologyEx(group_mask,cv2.MORPH_OPEN,CONST.MEDIUM_CO_EL,group_mask,borderType=cv2.BORDER_REPLICATE)
            cv2.morphologyEx(group_mask,cv2.MORPH_CLOSE,CONST.MEDIUM_CO_EL,group_mask,borderType=cv2.BORDER_REPLICATE)
            group_mask = group_mask.get()
            #filter the mask for the total number of pixels and regions by the minimum size
            group_mask = getSizeFilteredMask(group_mask,CONST.SATURATION_MIN_SIZE)
        if np.sum(group_mask==0)<CONST.SATURATION_MIN_PIXELS :
            group_mask = np.ones_like(group_mask)
        layer_group_saturation_masks.append(group_mask)
    return layer_group_saturation_masks

#helper function to create a layered binary image mask for a given image array
#this can be run in parallel with a given index and return dict
def getImageMaskWorker(im_array,rfp,rawfile_top_dir,bg_thresholds,exp_time_hists,norm_ets,make_plots=False,plotdir_path=None,i=None,return_dict=None) :
    #need the exposure times for this image
    exp_times = getExposureTimesByLayer(rfp,rawfile_top_dir)
    #start by creating the tissue mask
    tissue_mask = getImageTissueMask(im_array,bg_thresholds)
    #next create the blur mask
    blur_mask,blur_mask_plots = getImageBlurMask(im_array,exp_times,tissue_mask,exp_time_hists,make_plots)
    #finally create masks for the saturated regions in each layer group
    layer_group_saturation_masks = getImageSaturationMasks(im_array,norm_ets if norm_ets is not None else exp_times)
    #make the image_mask object 
    key = (os.path.basename(rfp)).rstrip(UNIV_CONST.RAW_EXT)
    image_mask = ImageMask(key)
    image_mask.addCreatedMasks(tissue_mask,blur_mask,layer_group_saturation_masks)
    #make the plots for this image if requested
    if make_plots :
        flatfield_logger.info(f'Saving masking plots for image {i}')
        doMaskingPlotsForImage(key,tissue_mask,blur_mask_plots,image_mask.compressed_mask,plotdir_path)
    #if there is anything flagged in the final blur and saturation masks, write out the compressed mask
    is_masked = np.min(blur_mask)<1
    if not is_masked :
        for lgsm in layer_group_saturation_masks :
            if np.min(lgsm)<1 :
                is_masked=True
                break
    if plotdir_path is not None :
        with cd(plotdir_path) :
            im3writeraw(f'{key}_tissue_mask.bin',image_mask.packed_tissue_mask)
            if is_masked :
                writeImageToFile(image_mask.compressed_mask,f'{key}_full_mask.bin',dtype=np.uint8)
    #return the mask (either in the shared dict or just on its own)
    if i is not None and return_dict is not None :
        return_dict[i] = image_mask
    else :
        return image_mask

