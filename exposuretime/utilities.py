#imports
from .alignmentset import AlignmentSetForExposureTime
from .config import CONST
from typing import List
import os, logging, dataclasses

#set up the logger
et_fit_logger = logging.getLogger("exposure_time_fitter")
et_fit_logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s  [%(funcName)s]","%Y-%m-%d %H:%M:%S"))
et_fit_logger.addHandler(handler)

#helper function to make sure necessary directories exist and that other arguments are valid
def checkArgs(args) :
    #rawfile_top_dir/[sample] must exist
    rawfile_dir = os.path.join(args.rawfile_top_dir,args.sample)
    if not os.path.isdir(rawfile_dir) :
        raise ValueError(f'ERROR: rawfile directory {rawfile_dir} does not exist!')
    #metadata top dir must exist
    if not os.path.isdir(args.metadata_top_dir) :
        raise ValueError(f'ERROR: metadata_top_dir argument ({args.metadata_top_dir}) does not point to a valid directory!')
    #metadata top dir dir must be usable to find a metafile directory
    metafile_dir = os.path.join(args.metadata_top_dir,args.sample)
    if not os.path.isdir(metafile_dir) :
        raise ValueError(f'ERROR: metadata_top_dir ({args.metadata_top_dir}) does not contain "[sample name]" subdirectory!')
    #make sure the flatfield file exists
    if not os.path.isfile(args.flatfield_file) :
        raise ValueError(f'ERROR: flatfield_file ({args.flatfield_file}) does not exist!')
    #create the working directory if it doesn't already exist
    if not os.path.isdir(args.workingdir_name) :
        os.mkdir(args.workingdir_name)
    #make sure the layers argument makes sense
    if len(args.layers)<1 :
    	raise ValueError(f'ERROR: layers argument {args.layers} must have at least one layer number (or -1)!')
    #make sure the overlaps argument makes sense
    if len(args.overlaps)<1 :
        raise ValueError(f'ERROR: overlaps argument {args.overlaps} must have at least one overlap number (or -1)!')

#helper function to return a list of overlap ns for overlaps where the p1 and p2 image exposure times are different
#can be run in parallel if given a return_dict (will be keyed by layer)
def getOverlapsWithExposureTimeDifferences(rtd,mtd,sn,exp_times,layer,overlaps=None,return_dict=None) :
    et_fit_logger.info(f'Finding overlaps with exposure time differences in {sn} layer {layer}....')
    if overlaps is None or overlaps==[-1] :
        a = AlignmentSetForExposureTime(mtd,rtd,sn,nclip=CONST.N_CLIP,readlayerfile=False,layer=layer,smoothsigma=None,flatfield=None)
    else :
        a = AlignmentSetForExposureTime(mtd,rtd,sn,nclip=CONST.N_CLIP,readlayerfile=False,layer=layer,
                                        selectoverlaps=overlaps,onlyrectanglesinoverlaps=True,smoothsigma=None,flatfield=None)
    rect_rfkey_by_n = {}
    for r in a.rectangles :
        rect_rfkey_by_n[r.n] = r.file.rstrip('.im3')
    olaps_with_et_diffs = []
    for olap in a.overlaps :
        p1key = rect_rfkey_by_n[olap.p1]
        p2key = rect_rfkey_by_n[olap.p2]
        if p1key in exp_times.keys() and p2key in exp_times.keys() :
            p1et = exp_times[p1key]
            p2et = exp_times[p2key]
            if p2et!=p1et :
                olaps_with_et_diffs.append(olap.n)
    if return_dict is not None :
        return_dict[layer] = olaps_with_et_diffs
    else :
        return olaps_with_et_diffs

#helper function to get the number of the first layer in the group of a given layer
def getFirstLayerInGroup(layer_n,nlayers) :
    if nlayers==35 :
        if layer_n in range(1,10) :
            return 1
        elif layer_n in range(10,19) :
            return 10
        elif layer_n in range(19,26) :
            return 19
        elif layer_n in range(26,33) :
            return 26
        elif layer_n in range(33,36) :
            return 33
        else :
            raise ValueError(f'ERROR: getFirstLayerInGroup called with nlayers={nlayers} but layer_n={layer_n} is outside that range!')
    elif nlayers==43 :
        if layer_n in range(1,10) :
            return 1
        elif layer_n in range(10,12) :
            return 10
        elif layer_n in range(12,18) :
            return 12
        elif layer_n in range(18,21) :
            return 18
        elif layer_n in range(21,30) :
            return 21
        elif layer_n in range(30,37) :
            return 30
        elif layer_n in range(37,44) :
            return 37
        else :
            raise ValueError(f'ERROR: getFirstLayerInGroup called with nlayers={nlayers} but layer_n={layer_n} is outside that range!')
    else :
        raise ValueError(f'ERROR: number of image layers ({nlayers}) passed to getFirstLayerInGroup is not a recognized option!')

#helper class to hold the pre- and post-fit details of overlaps
@dataclasses.dataclass
class ExposureTimeOverlapFitResult :
    n            : int
    p1           : int
    p2           : int
    tag          : int
    p1_et        : float
    p2_et        : float
    med_et       : float
    et_diff      : float
    npix         : int
    prefit_cost  : float
    postfit_cost : float

#helper class to log fields used in making the measurement
@dataclasses.dataclass
class FieldLog :
    file : str
    rect_n : int
    in_overlaps : List[int]
