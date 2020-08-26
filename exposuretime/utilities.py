#imports
import numpy as np
import os, logging, dataclasses

#set up the logger
et_fit_logger = logging.getLogger("exposure_time_fitter")
et_fit_logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(message)s    [%(funcName)s, %(asctime)s]"))
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
    metafile_dir = os.path.join(args.metadata_top_dir,args.sample,'im3','xml')
    if not os.path.isdir(metafile_dir) :
        raise ValueError(f'ERROR: metadata_top_dir ({args.metadata_top_dir}) does not contain "[sample name]/im3/xml" subdirectories!')
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

#helper class to hold a rectangle's rawfile key, raw image, and index in a list of Rectangles 
@dataclasses.dataclass(eq=False, repr=False)
class UpdateImage :
    rawfile_key          : str
    raw_image            : np.array
    rectangle_list_index : int

#helper class to hold the pre- and post-fit details of overlaps
@dataclasses.dataclass
class ExposureTimeOverlapFitResult :
    n            : int
    p1           : int
    p2           : int
    tag          : int
    p1_et        : float
    p2_et        : float
    et_diff      : float
    npix         : int
    prefit_cost  : float
    postfit_cost : float

#helper class to store offset factors by layer with some extra info
@dataclasses.dataclass
class LayerOffset :
    layer_n    : int
    n_overlaps : int
    offset     : float
    final_cost : float

#helper class to log fields used in making the measurement
@dataclasses.dataclass
class FieldLog :
    file : str
    rect_n : int
    in_overlaps : List[int]
