#imports
import numpy as np
from abc import abstractmethod
from ...utilities.optionalimports import deepcell
from .config import SEG_CONST
from .utilities import initialize_app, run_mesmer_segmentation
from .segmentationsample import SegmentationSampleDAPIMembraneComponentTiff, SegmentationSampleUsingComponentTiff, SegmentationSampleDAPIComponentMembraneIHCTiff

class SegmentationSampleMesmer(SegmentationSampleUsingComponentTiff) :

    def runsegmentation(self) :
        """
        Run whole-cell segmentation using the Mesmer segmentation algorithm
        """
        Mesmer = deepcell.applications.Mesmer
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
                mesmer_batch_images.append(self.get_rect_im_for_mesmer(rect))
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

    @abstractmethod
    def get_rect_im_for_mesmer(self,rect) :
        """
        Return the two-layer (DAPI/membrane) image that should be sent to Mesmer given a rectangle
        Not implemented in this class
        """
        pass
    
    @classmethod
    def segmentationalgorithm(cls):
      return "mesmer"

    @classmethod
    def logmodule(cls) : 
        return "segmentationmesmer"

    def __get_rect_segmented_fp(self,rect) :
        seg_fn = f'{rect.file.stem}_{SEG_CONST.MESMER_SEGMENT_FILE_APPEND}'
        return self.segmentationfolder/seg_fn

class SegmentationSampleMesmerWithIHC(SegmentationSampleDAPIComponentMembraneIHCTiff,SegmentationSampleMesmer) :

    def get_rect_im_for_mesmer(self, rect):
        with rect.using_component_tiff() as im :
            dapi_layer = im
        with rect.using_ihc_tiff() as im :
            membrane_layer = self.get_membrane_layer_from_ihc_image(im)
        return np.array([dapi_layer,membrane_layer]).transpose(1,2,0)

class SegmentationSampleMesmerComponentTiff(SegmentationSampleDAPIMembraneComponentTiff,SegmentationSampleMesmer) :

    def get_rect_im_for_mesmer(self, rect):
        with rect.using_component_tiff() as im :
            dapi_layer = im[:,:,0]
            membrane_layer = im[:,:,1]
        return np.array([dapi_layer,membrane_layer]).transpose(1,2,0)

def segmentationsamplemesmerwithihc(args=None) :
    SegmentationSampleMesmerWithIHC.runfromargumentparser(args)

def segmentationsamplemesmercomponenttiff(args=None) :
    SegmentationSampleMesmerComponentTiff.runfromargumentparser(args)
