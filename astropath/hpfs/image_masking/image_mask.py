#imports
from .utilities import LabelledMaskRegion, getEnumeratedMask, getSizeFilteredMask, get_image_tissue_fold_mask, get_image_layer_group_blur_mask, get_morphed_and_filtered_mask, get_exclusive_mask
from .plotting import doMaskingPlotsForImage
from .config import CONST
from ...utilities.img_file_io import smooth_image_worker, getExposureTimesByLayer, im3writeraw, writeImageToFile
from ...utilities.misc import cd
from ...utilities.config import CONST as UNIV_CONST
import numpy as np
import pathlib, cv2

class ImageMask() :
    """
    Class to store and work with a mask for an image
    """

    #################### PROPERTIES ####################

    @property
    def packed_tissue_mask(self) : #the binary tissue mask packed using np.packbits
        if self._tissue_mask is None :
            raise RuntimeError('ERROR: pasked_tissue_mask called before setting a tissue mask')
        return np.packbits(self._tissue_mask)
    @property
    def compressed_mask(self): #the compressed mask with (# layers groups)+1 layers
        if self._compressed_mask is None :
            raise RuntimeError('ERROR: compressed_mask called without first creating a mask!')
        return self._compressed_mask
    @property
    def onehot_mask(self) : #the mask with good tissue=1, everything else=0
        uncompressed_full_mask = self.uncompressed_full_mask
        return (np.where(uncompressed_full_mask==1,1,0)).astype(np.uint8)
    @property
    def uncompressed_full_mask(self): #the uncompressed mask with the real number of image layers
        if self._compressed_mask is None or self._layer_groups==[] :
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

    def __init__(self,im_array,im_key,bg_thresholds,norm_ets,savedir=None) :
        """
        im_array       = the multilayer image array whose mask should be created (may already be corrected to a set of exposure times)
        im_key         = the string representing the key of the image filename (used as a prepend to the masking file name)
        bg_thresholds  = a list of the background intensity thresholds in counts in each image layer
        norm_ets       = a list of the exposure times to which the image layers have been normalized 
        savedir        = path to the directory in which the mask file(s) should be saved (if None the mask files will not be written to disk)
        """
        #set the layer groups for the image
        if im_array.shape[-1]==35 :
            self.__layer_groups=UNIV_CONST.LAYER_GROUPS_35
        elif im_array.shape[-1]==43 :
            self.__layer_groups=UNIV_CONST.LAYER_GROUPS_43
        else :
            raise ValueError(f'ERROR: no defined list of broadband filter breaks for images with {im_array.shape[-1]} layers!')
        #create the tissue mask
        self.__tissue_mask = self.get_image_tissue_mask(im_array,bg_thresholds)
        #create the blur mask
        self.__blur_mask = self.get_image_blur_mask(im_array,self.__tissue_mask)
        #create the saturation masks (one for each layer group)
        self.__saturation_masks = get_image_saturation_masks(im_array,norm_ets)
        #make the compressed mask and the list of labelled mask regions
        self._make_compressed_mask_and_list_of_mask_regions()
        #make the plots for this image if requested
        if make_plots :
            doMaskingPlotsForImage(self._image_key,self._tissue_mask,blur_mask_plots,self._compressed_mask,savedir)
        #if there is anything flagged in the final blur and saturation masks, write out the compressed mask
        is_masked = np.min(self._blur_mask)<1
        if not is_masked :
            for lgsm in self._saturation_masks :
                if np.min(lgsm)<1 :
                    is_masked=True
                    break
        if savedir is not None :
            with cd(savedir) :
                im3writeraw(f'{self._image_key}_tissue_mask.bin',self.packed_tissue_mask)
                if is_masked :
                    writeImageToFile(self._compressed_mask,f'{self._image_key}_full_mask.bin',dtype=np.uint8)

    @classmethod
    def create_write_and_return_labelled_regions(cls,im_array,im_key,bg_thresholds,norm_ets,savedir=None) :
        """
        Create an ImageMask, write out the files it creates, and return its list of labelled mask regions
        This function writes out the ImageMask it creates and doesn't return it in order to have the lowest possible memory footprint
        which is useful for running many iterations of it in parallel processes
        
        arguments are the same as __init__ above
        """
        pass

    @classmethod
    def save_plots_for_image(cls,im_array,im_key,bg_thresholds,norm_ets,sample_exp_time_hists_and_bins,savedir=None) :
        """
        Create the masks for a given image and write out plots of the process
        Useful if all you care about is getting the plots; this function also has the lowest possible memory footprint to be run in parallel processes

        arguments are the same as __init__ with the addition of:
        sample_exp_time_hists_and_bins = a list of tuples of histograms and their bins of all layer group exposure times in the sample that im_array is coming from
        """
        pass

    @staticmethod
    def get_image_tissue_mask(image_arr,bkg_thresholds) :
        """
        return the fully-determined overall tissue mask as a 2d array of ones and zeroes for a given multilayer image

        image_arr      = the multilayer image array whose mask should be created
        bkg_thresholds = the list of background intensity thresholds to use in each image layer 
        """
        #figure out where the layer groups are
        if image_arr.shape[-1]==35 :
            mask_layer_groups=UNIV_CONST.LAYER_GROUPS_35
            dapi_layer_group_index=UNIV_CONST.DAPI_LAYER_GROUP_INDEX_35
        elif image_arr.shape[-1]==43 :
            mask_layer_groups=UNIV_CONST.LAYER_GROUPS_43
            dapi_layer_group_index=UNIV_CONST.DAPI_LAYER_GROUP_INDEX_43
        else :
            raise ValueError(f'ERROR: no defined list of broadband filter breaks for images with {image_arr.shape[-1]} layers!')
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

    @staticmethod
    def get_image_blur_mask(img_array,tissue_mask) :
        """
        return the single blur mask for an image (multilayer "tissue fold" blur and single-layer "dust" blur combined)

        img_array      = the multilayer image array whose mask should be created
        tissue_mask    = the 2d binary tissue mask for this image 
        """
        #figure out where the layer groups are, etc.
        if img_array.shape[-1]==35 :
            mask_layer_groups=UNIV_CONST.LAYER_GROUPS_35
            brightest_layers=UNIV_CONST.BRIGHTEST_LAYERS_35
            dapi_layer_group_index=UNIV_CONST.DAPI_LAYER_GROUP_INDEX_35
            rbc_layer_group_index=UNIV_CONST.RBC_LAYER_GROUP_INDEX_35
            nlv_cuts_by_layer_group=CONST.FOLD_NLV_CUTS_BY_LAYER_GROUP_35
            nlv_max_means_by_layer_group=CONST.FOLD_MAX_MEANS_BY_LAYER_GROUP_35
            dust_nlv_cut=CONST.DUST_NLV_CUT_35
            dust_max_mean=CONST.DUST_MAX_MEAN_35
            #smooth the image array for both the tissue fold and DAPI layer blur detection
            sm_img_array = smooth_image_worker(img_array,CONST.BLUR_MASK_SMOOTHING_SIGMA)
        elif img_array.shape[-1]==43 :
            mask_layer_groups=UNIV_CONST.LAYER_GROUPS_43
            brightest_layers=UNIV_CONST.BRIGHTEST_LAYERS_43
            dapi_layer_group_index=UNIV_CONST.DAPI_LAYER_GROUP_INDEX_43
            rbc_layer_group_index=UNIV_CONST.RBC_LAYER_GROUP_INDEX_43
            nlv_cuts_by_layer_group=CONST.FOLD_NLV_CUTS_BY_LAYER_GROUP_43
            nlv_max_means_by_layer_group=CONST.FOLD_MAX_MEANS_BY_LAYER_GROUP_43
            dust_nlv_cut=CONST.DUST_NLV_CUT_43
            dust_max_mean=CONST.DUST_MAX_MEAN_43
            #don't smooth the image array before blur detection
            sm_img_array = img_array
        else :
            raise ValueError(f'ERROR: no defined list of broadband filter breaks for images with {img_array.shape[-1]} layers!')
        #get the tissue fold mask
        tissue_fold_mask = get_image_tissue_fold_mask(sm_img_array,tissue_mask,mask_layer_groups,brightest_layers,dapi_layer_group_index,rbc_layer_group_index,
                                                  nlv_cuts_by_layer_group,nlv_max_means_by_layer_group)
        #get masks for the blurriest areas of the DAPI layer group
        dapi_dust_mask = get_image_layer_group_blur_mask(sm_img_array,
                                                         mask_layer_groups[dapi_layer_group_index],
                                                         dust_nlv_cut,
                                                         0.5*(mask_layer_groups[dapi_layer_group_index][1]-mask_layer_groups[dapi_layer_group_index][0]+1),
                                                         dust_max_mean,
                                                         brightest_layers[dapi_layer_group_index])
        #same morphology transformations as for the multilayer blur masks
        dapi_dust_mask = get_morphed_and_filtered_mask(dapi_dust_mask,tissue_mask,CONST.DUST_MIN_PIXELS,CONST.DUST_MIN_SIZE)
        #make sure any regions in that mask are sufficiently exclusive w.r.t. what's already flagged as blurry
        dapi_dust_mask = get_exclusive_mask(dapi_dust_mask,tissue_fold_mask,0.25)
        #combine the multilayer and single layer blur masks into one by multiplying them together
        final_blur_mask = tissue_fold_mask*dapi_dust_mask
        #return the blur mask
        return final_blur_mask

    #################### PRIVATE HELPER FUNCTIONS ####################

    #helper function to combine all the created masks into a compressed mask and also make the list of labelled mask region objects
    def _make_compressed_mask_and_list_of_mask_regions(self) :
        #start the list of labelled mask regions
        self._labelled_mask_regions = []
        #make the compressed mask, which has (# of layer groups)+1 layers total
        #the first layer holds just the tissue and blur masks; the other layers have the tissue and blur masks plus the saturation mask for each layer group
        self._compressed_mask = np.ones((*self._tissue_mask.shape,len(self._layer_groups)+1),dtype=np.uint8)
        #add in the blur mask, starting with index 2
        start_i = 2
        if np.min(self._blur_mask)<1 :
            layers_string = 'all'
            enumerated_blur_mask = getEnumeratedMask(self._blur_mask,start_i)
            for li in range(self._compressed_mask.shape[-1]) :
                self._compressed_mask[:,:,li][enumerated_blur_mask!=0] = enumerated_blur_mask[enumerated_blur_mask!=0]
            start_i = np.max(enumerated_blur_mask)+1
            region_indices = list(range(np.min(enumerated_blur_mask[enumerated_blur_mask!=0]),np.max(enumerated_blur_mask)+1))
            for ri in region_indices :
                r_size = np.sum(enumerated_blur_mask==ri)
                self._labelled_mask_regions.append(LabelledMaskRegion(self._image_key,ri,layers_string,r_size,CONST.BLUR_FLAG_STRING))
        #add in the saturation masks 
        for lgi,lgsm in enumerate(self._saturation_masks) :
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
            self._compressed_mask[:,:,li][self._compressed_mask[:,:,li]==1] = self._tissue_mask[self._compressed_mask[:,:,li]==1]

#################### FILE-SCOPE HELPER FUNCTIONS ####################

def get_image_saturation_masks(image_arr,norm_ets) :
    """
    Return the list of saturation masks by layer group for a given image

    image_arr = the multilayer image array of uint16s whose mask should be created
    norm_ets  = the list of exposure times in each layer that should be used to normalize the given image 
    """
    #figure out where the layer groups are, etc.
    if image_arr.shape[-1]==35 :
        mask_layer_groups=UNIV_CONST.LAYER_GROUPS_35
        saturation_intensity_cuts=CONST.SATURATION_INTENSITY_CUTS_35
    elif image_arr.shape[-1]==43 :
        mask_layer_groups=UNIV_CONST.LAYER_GROUPS_43
        saturation_intensity_cuts=CONST.SATURATION_INTENSITY_CUTS_43
    else :
        raise ValueError(f'ERROR: no defined list of broadband filter breaks for images with {image_arr.shape[-1]} layers!')
    #normalize the image by its exposure time and smooth it
    normalized_image_arr = image_arr / norm_ets[np.newaxis,np.newaxis,:]
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
