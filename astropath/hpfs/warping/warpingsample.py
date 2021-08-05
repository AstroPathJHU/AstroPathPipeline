#imports
import sys, numpy as np
from reikna.fft import FFT
from ...utilities.config import CONST as UNIV_CONST
from ...utilities.misc import cd, get_GPU_thread
from ...utilities.tableio import readtable, writetable
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

    def __init__(self,*args,workingdir=None,useGPU=True,gputhread=None,gpufftdict=None,**kwargs) :
        super().__init__(*args,**kwargs)
        #make sure the user is only specifying a single layer
        if len(self.layers)!=1 :
            errmsg = f'ERROR: a WarpingSample can only be run for one layer at a time but layers = {self.layers}'
            raise RuntimeError(errmsg)
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
        if gputhread is not None and gpufftdict is not None :
            if not useGPU :
                raise RuntimeError(f'ERROR: passed GPU parameters to a WarpingSample but useGPU = {useGPU}')
            self.gputhread = gputhread
            self.gpufftdict = gpufftdict
        else :
            self.gputhread = get_GPU_thread(sys.platform=='darwin') if useGPU else None
            self.gpufftdict = None if self.gputhread is None else {}
        #give the sample a placeholder octets variable
        self.__octets = None

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
        for octet_filepath in (workingdir_octet_filepath,cohort_octet_filepath) :
            if octet_filepath.is_file() :
                try :
                    self.__octets = readtable(octet_filepath,OverlapOctet)
                    self.logger.info(f'Will use {len(self.__octets)} octets in {octet_filepath} for {self.SlideID}')
                    return
                except TypeError :
                    msg = f'Found an empty octet file for {self.SlideID} at {octet_filepath} and so it will be assumed '
                    msg+= f'this sample has no octets to use in fitting for warping'
                    self.logger.info(msg)
                    self.__octets = []
                    return
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
    def octets(self) :
        return self.__octets
    @property
    def bg_threshold_filepath(self) :
        bgtfp = self.root / f'{self.SlideID}' / f'{UNIV_CONST.IM3_DIR_NAME}' / f'{UNIV_CONST.MEANIMAGE_DIRNAME}'
        bgtfp = bgtfp / f'{self.SlideID}-{FF_CONST.BACKGROUND_THRESHOLD_CSV_FILE_NAME_STEM}'
        return bgtfp

    #################### CLASS METHODS ####################

    @classmethod
    def getoutputfiles(cls,SlideID,root,workingdir=None,**otherworkflowkwargs) :
        #the octet file is the only required output for the sample
        if workingdir is not None :
            octet_filepath = workingdir
        else :
            octet_filepath = root / UNIV_CONST.WARPING_DIRNAME
        octet_filepath = octet_filepath / CONST.OCTET_SUBDIR_NAME / f'{SlideID}-{CONST.OCTET_FILENAME_STEM}'
        return [octet_filepath]
    @classmethod
    def logmodule(cls) : 
        return "warping"
    @classmethod
    def defaultunits(cls) :
        return "fast"
    @property
    def workflowkwargs(self) :
        return{**super().workflowkwargs,'skip_masking':False,'workingdir':self.__workingdir}
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
        msg = f'Finding octets to use for warp fitting in {self.SlideID} from a set of {len(octet_candidate_rects)} '
        msg+= f'possible central HPFs'
        self.logger.debug(msg)
        #get the background threshold for this layer of this sample
        bg_thresholds = readtable(self.bg_threshold_filepath,ThresholdTableEntry)
        this_layer_bgts = [bgt for bgt in bg_thresholds if bgt.layer_n==self.layers[0]]
        if len(this_layer_bgts)!=1 :
            errmsg = f'ERROR: invalid set of background thresholds for {self.SlideID} found '
            errmsg+= f'in {self.bg_threshold_filepath}!'
            raise ValueError(errmsg)
        bg_threshold = this_layer_bgts[0]
        #check each possible octet of overlaps to make sure all its overlaps can be aligned and its images have 
        #at least some base fraction of pixels with intensities above the background threshold
        self.__octets = []
        for ir,r in enumerate(octet_candidate_rects,start=1) :
            n_good_overlaps = 0
            overlap_n_tuples = self.overlapsforrectangle(r.n)
            overlaps = [o for o in self.overlaps if (o.p1,o.p2) in overlap_n_tuples]
            overlaps_by_tag = {}
            p1_pixel_fracs_by_tag = {}
            p2_pixel_fracs_by_tag = {}
            for o in overlaps :
                result = self.__align_overlap(o)
                if result is not None and result.exit==0 :
                    ip1,ip2 = o.cutimages
                    p1frac = (np.sum(np.where(ip1>bg_threshold.counts_threshold,1,0)))/(ip1.shape[0]*ip1.shape[1])
                    p2frac = (np.sum(np.where(ip2>bg_threshold.counts_threshold,1,0)))/(ip2.shape[0]*ip2.shape[1])
                    if p1frac>CONST.REQ_OVERLAP_PIXEL_FRAC and p2frac>CONST.REQ_OVERLAP_PIXEL_FRAC :
                        overlaps_by_tag[o.tag] = o
                        p1_pixel_fracs_by_tag[o.tag] = p1frac
                        p2_pixel_fracs_by_tag[o.tag] = p2frac
                        n_good_overlaps+=1
                    else :
                        msg = f'Overlap {o.n} for rectangle {r.n} ({ir}/{len(octet_candidate_rects)}) rejected '
                        msg+= f'(pixel fractions: {p1frac:.02f},{p2frac:.02f})'
                        self.logger.debug(msg)
                        break
                else :
                    msg = f'Overlap {o.n} for rectangle {r.n} ({ir}/{len(octet_candidate_rects)}) rejected '
                    msg+= f'(alignment result: {result})'
                    self.logger.debug(msg)
                    break
            if n_good_overlaps==8 :
                new_octet = OverlapOctet(self.layers[0],
                                         bg_threshold.counts_threshold,bg_threshold.counts_per_ms_threshold,
                                         r.n,
                                         *([overlaps_by_tag[ot].n for ot in range(1,10) if ot!=5]),
                                         *([p1_pixel_fracs_by_tag[ot] for ot in range(1,10) if ot!=5]),
                                         *([p2_pixel_fracs_by_tag[ot] for ot in range(1,10) if ot!=5]))
                self.__octets.append(new_octet)
                self.logger.debug(f'Octet found surrounding rectangle {r.n} ({ir}/{len(octet_candidate_rects)})')
        #write out the octet file
        workingdir_octet_filepath = self.__workingdir / CONST.OCTET_SUBDIR_NAME
        workingdir_octet_filepath = workingdir_octet_filepath / f'{self.SlideID}-{CONST.OCTET_FILENAME_STEM}'
        if not workingdir_octet_filepath.parent.is_dir() :
            workingdir_octet_filepath.parent.mkdir(parents=True)
        msg = f'Found {len(self.__octets)} octet{"s" if len(self.__octets)!=1 else ""} for {self.SlideID}, '
        msg+= f'writing out octet table to {workingdir_octet_filepath.resolve()}'
        self.logger.info(msg)
        if len(self.__octets)==0 :
            workingdir_octet_filepath.touch()
        else :
            with cd(workingdir_octet_filepath.parent) :
                writetable(workingdir_octet_filepath.name,self.__octets)

    def __align_overlap(self,overlap) :
        """
        Return the result of aligning an overlap, after making some replacements
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
