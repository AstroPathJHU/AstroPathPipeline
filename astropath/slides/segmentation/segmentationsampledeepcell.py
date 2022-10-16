#imports
import numpy as np
from ...utilities.optionalimports import deepcell
from .config import SEG_CONST
from .utilities import initialize_app, run_deepcell_nuclear_segmentation
from .segmentationsample import SegmentationSampleDAPIComponentTiff

class SegmentationSampleDeepCell(SegmentationSampleDAPIComponentTiff) :
    
    def __init__(self,*args,**kwargs) :
        super().__init__(*args,**kwargs)

    @classmethod
    def segmentationalgorithm(cls):
      return "deepcell"

    @classmethod
    def logmodule(cls) : 
        return "segmentationdeepcell"

    def runsegmentation(self) :
        """
        Run nuclear segmentation using DeepCell's nuclear segmentation algorithm
        """
        NuclearSegmentation = deepcell.applications.NuclearSegmentation
        self.logger.debug('Running nuclear segmentation with DeepCell....')
        if self.njobs is not None and self.njobs>1 :
            self.logger.warning(f'WARNING: njobs is {self.njobs} but DeepCell segmentation cannot be run in parallel.')
        app = initialize_app(NuclearSegmentation, logger=self.logger)
        rects_to_run = []
        for ir,rect in enumerate(self.rectangles,start=1) :
            #skip any rectangles that already have segmentation output
            if self.__get_rect_segmented_fp(rect).is_file() :
                msg = f'Skipping {rect.componenttifffile.name} ({ir} of {len(self.rectangles)}) '
                msg+= '(segmentation output already exists)'
                self.logger.debug(msg)
                continue
            rects_to_run.append((ir,rect,self.__get_rect_segmented_fp(rect)))
        completed_files = 0
        try :
            deepcell_batch_images = []
            deepcell_batch_segmented_filepaths = []
            for realir,(ir,rect,segmented_file_path) in enumerate(rects_to_run,start=1) :
                #add to the batch
                msg = f'Adding {rect.componenttifffile.name} ({ir} of {len(self.rectangles)}) '
                msg+= 'to the next group of images....'
                self.logger.debug(msg)
                with rect.using_component_tiff() as im :
                    dapi_layer = im
                im_for_deepcell = np.expand_dims(dapi_layer,axis=-1)
                deepcell_batch_images.append(im_for_deepcell)
                deepcell_batch_segmented_filepaths.append(segmented_file_path)
                #run segmentations for a whole batch
                if (len(deepcell_batch_images)>=SEG_CONST.GROUP_SIZE) or (realir==len(rects_to_run)) :
                    msg = f'Running DeepCell segmentation for the current group of {len(deepcell_batch_images)} images'
                    self.logger.debug(msg)
                    run_deepcell_nuclear_segmentation(np.array(deepcell_batch_images),
                                                      app,
                                                      self.pscale,
                                                      deepcell_batch_segmented_filepaths)
                    deepcell_batch_images = []
                    deepcell_batch_segmented_filepaths = []
            for rect in self.rectangles :
                if self.__get_rect_segmented_fp(rect).is_file() :
                    completed_files+=1
        except Exception as e :
            raise e
        finally :
            if completed_files==len(self.rectangles) :
                self.logger.info(f'All files segmented using DeepCell with output in {self.segmentationfolder}')
            else :
                msg = f'{completed_files} of {len(self.rectangles)} files segmented using DeepCell. '
                msg+= 'Rerun the same command to retry.'
                self.logger.info(msg)

    def __get_rect_segmented_fp(self,rect) :
        seg_fn = f'{rect.file.stem}_{SEG_CONST.DEEPCELL_SEGMENT_FILE_APPEND}'
        return self.segmentationfolder/seg_fn

def main(args=None) :
    SegmentationSampleDeepCell.runfromargumentparser(args)

if __name__=='__main__' :
    main()
