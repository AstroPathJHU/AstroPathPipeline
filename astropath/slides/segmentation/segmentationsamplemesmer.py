#imports
import numpy as np
from ...utilities.optionalimports import deepcell
from ...utilities.config import CONST as UNIV_CONST
from .config import SEG_CONST
from .utilities import initialize_app, run_mesmer_segmentation
from .segmentationsample import SegmentationSampleBase

class SegmentationSampleMesmer(SegmentationSampleBase) :
    
    def __init__(self,*args,**kwargs) :
        super().__init__(*args,**kwargs)

    @classmethod
    def segmentationalgorithm(cls):
      return "mesmer"

    @classmethod
    def logmodule(cls) : 
        return "segmentationmesmer"

    def runsegmentation(self) :
        """
        Run whole-cell segmentation using the Mesmer segmentation algorithm
        """
        Mesmer = deepcell.applications.Mesmer
        pca_vec_to_dot = np.expand_dims(SEG_CONST.IHC_PCA_BLACK_COMPONENT,0).T
        self.logger.debug('Running whole-cell and nuclear segmentation with Mesmer....')
        if self.njobs is not None and self.njobs>1 :
            self.logger.warning(f'WARNING: njobs is {self.njobs} but Mesmer segmentation cannot be run in parallel.')
        app = initialize_app(Mesmer, logger=self.logger)
        rects_to_run = []
        for ir,rect in enumerate(self.rectangles,start=1) :
            #skip any rectangles that already have segmentation output
            if self.__get_rect_segmented_fp(rect).is_file() :
                msg = f'Skipping {rect.ihctifffile.name} ({ir} of {len(self.rectangles)}) '
                msg+= '(segmentation output already exists)'
                self.logger.debug(msg)
                continue
            rects_to_run.append((ir,rect,self.__get_rect_segmented_fp(rect)))
        completed_files = 0
        try :
            mesmer_batch_images = []
            mesmer_batch_segmented_filepaths = []
            for realir,(ir,rect,segmented_file_path) in enumerate(rects_to_run,start=1) :
                #add to the batch
                msg = f'Adding {rect.ihctifffile.name} ({ir} of {len(self.rectangles)}) to the next group of images....'
                self.logger.debug(msg)
                with rect.using_component_tiff() as im :
                    dapi_layer = im
                with rect.using_ihc_tiff() as im :
                    im_membrane = -1.0*(np.dot(im,pca_vec_to_dot))[:,:,0]
                    membrane_layer = (im_membrane-np.min(im_membrane))/SEG_CONST.IHC_MEMBRANE_LAYER_NORM
                im_for_mesmer = np.array([dapi_layer,membrane_layer]).transpose(1,2,0)
                mesmer_batch_images.append(im_for_mesmer)
                mesmer_batch_segmented_filepaths.append(segmented_file_path)
                #run segmentations for a whole batch
                if (len(mesmer_batch_images)>=SEG_CONST.GROUP_SIZE) or (realir==len(rects_to_run)) :
                    msg = f'Running Mesmer segmentation for the current group of {len(mesmer_batch_images)} images'
                    self.logger.debug(msg)
                    run_mesmer_segmentation(np.array(mesmer_batch_images),
                                            app,
                                            self.pscale,
                                            mesmer_batch_segmented_filepaths)
                    mesmer_batch_images = []
                    mesmer_batch_segmented_filepaths = []
            for rect in self.rectangles :
                if self.__get_rect_segmented_fp(rect).is_file() :
                    completed_files+=1
        except Exception as e :
            raise e
        finally :
            if completed_files==len(self.rectangles) :
                self.logger.info(f'All files segmented using Mesmer with output in {self.segmentationfolder}')
            else :
                msg = f'{completed_files} of {len(self.rectangles)} files segmented using Mesmer. '
                msg+= 'Rerun the same command to retry.'
                self.logger.info(msg)
        pass

    def __get_rect_segmented_fp(self,rect) :
        seg_fn = f'{rect.file.rstrip(UNIV_CONST.IM3_EXT)}_{SEG_CONST.MESMER_SEGMENT_FILE_APPEND}'
        return self.segmentationfolder/seg_fn

def main(args=None) :
    SegmentationSampleMesmer.runfromargumentparser(args)

if __name__=='__main__' :
    main()