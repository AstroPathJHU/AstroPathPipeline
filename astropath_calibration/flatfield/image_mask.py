#imports
from .utilities import LabelledMaskRegion
from ..utilities.config import CONST as UNIV_CONST
import numpy as np
import cv2

class ImageMask() :
    """
    Class to store and work with a mask for an image
    """

    #################### PROPERTIES ####################

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

    #################### CLASS CONSTANTS ####################

    BLUR_FLAG_STRING = 'blurred likely folded tissue or dust' #descriptive string to use for blurred areas in the labelled mask regions file
    SATURATION_FLAG_STRING = 'saturated likely skin or red blood cells or stain' #descriptive string to use for saturated areas in the labelled mask regions file

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

    def addCreatedMasks(self,tissue_mask,blur_mask,saturation_masks) :
        """
        tissue_mask      = the tissue (1) vs. background (0) mask that should be added to the file
        blur_mask        = the blurred region mask (1=not blurred, 0=blurred) that should be added to the file
        saturation_masks = the list of saturation masks (1=not saturated, 0=saturated) to add to the file, one per broadband filter layer group
        """
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
                self._labelled_mask_regions.append(LabelledMaskRegion(self._image_key,ri,layers_string,r_size,self.BLUR_FLAG_STRING))
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
                    self._labelled_mask_regions.append(LabelledMaskRegion(self._image_key,ri,layers_string,r_size,self.SATURATION_FLAG_STRING))
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

