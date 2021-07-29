#imports
import sys, numpy as np
from reikna.fft import FFT
from ...utilities.config import CONST as UNIV_CONST
from ...utilities.misc import get_GPU_thread
from ...utilities.tableio import readtable
from ...shared.argumentparser import FileTypeArgumentParser, WorkingDirArgumentParser, GPUArgumentParser
from ...shared.sample import ReadCorrectedRectanglesOverlapsIm3SingleLayerFromXML, WorkflowSample
from ...shared.rectangle import RectangleCorrectedIm3SingleLayer
from ...slides.align.rectangle import AlignmentRectangleBase
from ...slides.align.overlap import AlignmentOverlap
from ..flatfield.config import CONST as FF_CONST
from ..flatfield.utilities import ThresholdTableEntry
from ..flatfield.meanimagesample import MeanImageSample
from .config import CONST
from .utilities import OverlapOctet

class AlignmentRectangleForWarping(RectangleCorrectedIm3SingleLayer,AlignmentRectangleBase) :
    """
    Rectangles that are used to fit warping
    """
    def __post_init__(self,*args,**kwargs) :
        super().__post_init__(*args,use_mean_image=False,**kwargs)

class WarpingSample(ReadCorrectedRectanglesOverlapsIm3SingleLayerFromXML, WorkflowSample,
                    FileTypeArgumentParser, WorkingDirArgumentParser, GPUArgumentParser) :
    """
    Class to find octets to use in determining warping patterns and 
    help collect results of fits to find those patterns
    """

    #################### PUBLIC FUNCTIONS ####################

    def __init__(self,*args,workingdir=None,useGPU=True,**kwargs) :
        super().__init__(*args,**kwargs)
        if self.et_offset_file is None :
            raise RuntimeError('ERROR: must supply an exposure time offset file to fit for warping!')
        if self.flatfield_file is None :
            raise RuntimeError('ERROR: must supply a flatfield file to fit for warping!')
        if self.warping_file is not None :
            raise RuntimeError('ERROR: cannot apply warping corrections before fitting for warping!')
        self.__workingdir = workingdir
        #if the working directory wasn't given, set it to the "Warping" directory inside the root directory
        if self.__workingdir is None :
            self.__workingdir = self.root / UNIV_CONST.WARPING_DIRNAME
        if not self.__workingdir.is_dir() :
            self.__workingdir.mkdir(parents=True)
        #set things up to use the GPU to perform alignment
        self.gputhread = get_GPU_thread(sys.platform=='darwin') if useGPU else None
        self.gpufftdict = None if self.gputhread is None else {}

    def run(self) :
        """
        Main function that will be looped when entire cohorts are run
        For my purposes here all it does is set the variable holding the octets to use
        """
        self.get_octets()

    def get_octets(self) :
        """
        Gets the list of OverlapOctet objects to use for fitting warping in this sample. 
        If there's already a file with the information in it this returns that information, 
        otherwise it will run routines to create that file.
        """
        workingdir_octet_filepath = self.__workingdir / CONST.OCTET_SUBDIR_NAME
        workingdir_octet_filepath = workingdir_octet_filepath / f'{self.SlideID}-{CONST.OCTET_FILENAME_STEM}'
        cohort_octet_filepath = self.root / UNIV_CONST.WARPING_DIRNAME / CONST.OCTET_SUBDIR_NAME
        cohort_octet_filepath = cohort_octet_filepath / f'{self.SlideID}-{CONST.OCTET_FILENAME_STEM}'
        self.__octets = None
        for octet_filepath in (workingdir_octet_filepath,cohort_octet_filepath) :
            if octet_filepath.is_file() :
                try :
                    self.__octets = readtable(octet_filepath,OverlapOctet)
                except TypeError :
                    msg = f'Found an empty octet file for {self.SlideID} at {octet_filepath} and so it will be assumed '
                    msg+= f'this sample has no octets to use in fitting for warping'
                    self.logger.info(msg)
                    self.__octets = []
                    return
                self.logger.info(f'Will use {len(self.__octets)} octets found in {octet_filepath} for {self.SlideID}')
        if self.__octets is None :
            self.logger.info(f'Will find octets to use for {self.SlideID}')
            self.__find_octets(workingdir_octet_filepath)

    def inputfiles(self,**kwargs) :
        return [*super().inputfiles(**kwargs),
                *(r.imagefile for r in self.rectangles),
                self.bg_threshold_filepath,
               ]
        
    #################### CLASS VARIABLES + PROPERTIES ####################

    rectangletype = AlignmentRectangleForWarping
    overlaptype = AlignmentOverlap
    nclip = UNIV_CONST.N_CLIP

    @property
    def workingdir(self) :
        return self.__workingdir
    @property
    def bg_threshold_filepath(self) :
        bgtfp = self.root / f'{self.SlideID}' / f'{UNIV_CONST.IM3_DIR_NAME}' / f'{UNIV_CONST.MEANIMAGE_DIRNAME}'
        bgtfp = bgtfp / f'{self.SlideID}-{FF_CONST.BACKGROUND_THRESHOLD_CSV_FILE_NAME_STEM}'
        return bgtfp

    #################### CLASS METHODS ####################

    @classmethod
    def getoutputfiles(cls,SlideID,root,**otherworkflowkwargs) :
        #the octet file is the only required output for the sample
        return root / UNIV_CONST.WARPING_DIRNAME / CONST.OCTET_SUBDIR_NAME / f'{SlideID}-{CONST.OCTET_FILENAME_STEM}'
    @classmethod
    def logmodule(cls) : 
        return "warping"
    @classmethod
    def defaultunits(cls) :
        return "fast"
    @classmethod
    def workflowdependencyclasses(cls):
        return [MeanImageSample]+super().workflowdependencyclasses()

    #################### PRIVATE HELPER FUNCTIONS ####################

    def __find_octets(self,octet_filepath) :
        """
        Find octets of overlaps that have a sufficient amount of tissue to use for fitting for warping
        """
        #first find all of the rectangles that even have 8 overlaps
        octet_candidate_rects = [r for r in self.rectangles if len(self.overlapsforrectangle(r.n))==8]
        if len(octet_candidate_rects)<1 :
            msg = f'{self.SlideID} does not have any rectangles with a full octet of overlaps and cannot be used '
            msg+=  'for fitting warping'
            self.logger.info(msg)
            octet_filepath.touch()
            self.__octets = []
            return
        #get the background thresholds for this sample and make sure they're valid
        bg_thresholds = readtable(self.bg_threshold_filepath,ThresholdTableEntry)
        for li in range(self.nlayers) :
            this_layer_bgts = [bgt for bgt in bg_thresholds if bgt.layer_n==li+1]
            if len(this_layer_bgts)!=1 :
                errmsg = f'ERROR: invalid set of background thresholds for {self.SlideID} found '
                errmsg+= f'in {self.bg_threshold_filepath}!'
                raise ValueError(errmsg)
        #check each possible octet of overlaps to make sure all its images have at least some 
        #base fraction of pixels with intensities above the background threshold
        for r in octet_candidate_rects :
            overlap_n_tuples = self.overlapsforrectangle(r.n)
            overlaps = [o for o in self.overlaps if (o.p1,o.p2) in overlap_n_tuples]
            for o in overlaps :
                result = self.__align_overlap(o)
                print(result)




    def __align_overlap(self,overlap) :
        """
        Return the result of aligning an overlap, after making some replacements etc. 
        to possibly run the alignment on the GPU
        """
        if self.gputhread is None or self.gpufftdict is None :
            return overlap.align()
        cutimages_shapes = tuple(im.shape for im in overlap.cutimages)
        assert cutimages_shapes[0] == cutimages_shapes[1]
        if cutimages_shapes[0] not in self.gpufftdict.keys() :
            gpu_im = np.ndarray(cutimages_shapes[0],dtype=np.csingle)
            new_fft = FFT(gpu_im)
            new_fftc = new_fft.compile(self.gputhread)
            self.gpufftdict[cutimages_shapes[0]] = new_fftc
        return overlap.align(gputhread=self.gputhread,gpufftdict=self.gpufftdict)
        



#################### FILE-SCOPE FUNCTIONS ####################

def main(args=None) :
    WarpingSample.runfromargumentparser(args)

if __name__=='__main__' :
    main()
