#imports
from abc import abstractmethod
import numpy as np
from sklearn.decomposition import IncrementalPCA
from ...utilities.config import CONST as UNIV_CONST
from ...utilities.img_file_io import smooth_image_worker
from ...shared.overlap import Overlap
from ...shared.image_masking.image_mask import ImageMask
from ...shared.sample import ReadRectanglesBase, ReadCorrectedRectanglesOverlapsIm3MultiLayerFromXML, MaskSampleBase

class PCASampleBase(ReadRectanglesBase,MaskSampleBase) :
    """
    General class to work with a PCA across a set of rectangle images

    The PCA is calculated neglecting any pixels that aren't marked as "good tissue" in the image masks
    """

    overlaptype = Overlap
    nclip = UNIV_CONST.N_CLIP

    def __init__(self,*args,n_components=None,batch_size=10,**kwargs) :
        super().__init__(*args,**kwargs)
        self.pca = IncrementalPCA(n_components=n_components,batch_size=batch_size)

    def run(self) :
        """
        Calculate the PCA using all of the rectangle images
        """
        #loop over the rectangles
        for ir,r in enumerate(self.rectangles,start=1) :
            self.logger.debug(f'Adding {r.file.replace(".im3","")} to PCA ({ir}/{len(self.rectangles)})...')
            dims = r.im3shape
            #find the mask to use for this image
            fmfp = self.maskfolder/r.file.replace('.im3','_full_mask.bin')
            if fmfp.is_file() :
                mask_as_read = ImageMask.onehot_mask_from_full_mask_file(fmfp,dims)
                sum_mask = np.sum(mask_as_read,axis=2)
                mask = np.where(sum_mask==dims[2],1,0)
            else :
                tmfp = self.maskfolder/r.file.replace('.im3','_tissue_mask.bin')
                if not tmfp.is_file() :
                    raise ValueError(f'ERROR: tissue mask file {tmfp} not found!')
                mask = ImageMask.unpack_tissue_mask(tmfp,dims[:-1])
            #add the image to the PCA
            with self.get_image_for_rectangle(r) as im :
                #smooth the image VERY gently
                im = smooth_image_worker(im,1)
                #mask out any pixels other than the good tissue in every layer
                im = im.reshape(dims[0]*dims[1],dims[2])
                mask = mask.flatten()
                masked_im = np.delete(im,np.where(mask==0),axis=0)
                if masked_im.shape[0]>0 :
                    self.pca.partial_fit(masked_im)

    @abstractmethod
    def get_image_for_rectangle(self,rect) :
        raise NotImplementedError('ERROR: get_image_for_rectangle is not implemented in the base class!')

    @classmethod
    def logmodule(cls) : 
        return "pcasample"

class PCASampleCorrectedIm3(ReadCorrectedRectanglesOverlapsIm3MultiLayerFromXML,PCASampleBase) :
    """
    Class to work with a PCA across all of a slide's images
    The PCA is calculated using images after correction for 
    exposure time and flatfielding effects
    """

    def get_image_for_rectangle(self, rect):
        return rect.using_corrected_im3()
