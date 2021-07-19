#imports
from .config import CONST
from ...utilities.dataclasses import MyDataClass
from ...utilities.img_correction import correctImageLayerForExposureTime, correctImageLayerWithFlatfield
from ...utilities.img_file_io import get_raw_as_hwl, getExposureTimesByLayer
from ...utilities.misc import split_csv_to_list, addCommonArgumentsToParser
import cv2, pathlib, logging, copy

#set up the logger
warp_logger = logging.getLogger("warpfitter")
warp_logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s  [%(funcName)s]","%Y-%m-%d %H:%M:%S"))
warp_logger.addHandler(handler)

#Class for errors encountered during warping
class WarpingError(Exception) :
    pass

#helper class to hold a rectangle's rawfile key, raw image, warped image, and tag for whether it's only relevant for overlaps that are corners
class WarpImage(MyDataClass) :
    rawfile_key          : str
    raw_image_umat       : cv2.UMat
    warped_image_umat    : cv2.UMat
    is_corner_only       : bool
    rectangle_list_index : int
    @property
    def raw_image(self):
        return self.raw_image_umat.get()
    @property
    def warped_image(self):
        return self.warped_image_umat.get()

#little utility class to represent a warp fit result
class WarpFitResult(MyDataClass) :
    dirname         : str = None
    n               : int = 0
    m               : int = 0
    cx              : float = 0.0
    cy              : float = 0.0
    fx              : float = 0.0
    fy              : float = 0.0
    k1              : float = 0.0
    k2              : float = 0.0
    k3              : float = 0.0
    p1              : float = 0.0
    p2              : float = 0.0
    max_r_x_coord   : float = 0.0
    max_r_y_coord   : float = 0.0
    max_r           : float = 0.0
    max_rad_warp    : float = 0.0
    max_tan_warp    : float = 0.0
    global_fit_its  : int = 0
    polish_fit_its  : int = 0
    global_fit_time : float = 0.0
    polish_fit_time : float = 0.0
    raw_cost        : float = 0.0
    best_cost       : float = 0.0
    cost_reduction  : float = 0.0

#little utility class to log the fields used
class FieldLog(MyDataClass) :
    slide_ID : str
    file   : str
    rect_n : int

#little utility class for logging x and y principal point shifts
class WarpShift(MyDataClass) :
    layer_n  : int
    cx_shift : float
    cy_shift : float

#utility class for logging warping parameters and the slide they come from
class WarpingSummary(MyDataClass) :
    slide_ID     : str
    project         : int
    cohort          : int
    microscope_name : str
    mindate         : str
    maxdate         : str
    first_layer_n   : int
    last_layer_n    : int
    n               : int
    m               : int
    cx              : float
    cy              : float
    fx              : float
    fy              : float
    k1              : float
    k2              : float
    k3              : float
    p1              : float
    p2              : float

#helper function to mutate an argument parser for some generic warping options
def addCommonWarpingArgumentsToParser(parser,fit=True,fitpars=True,job_organization=True) :
    #add the common options, except not for warping
    addCommonArgumentsToParser(parser,warping=False)
    #group for basic options of how the fit(s) should be done
    if fit :
        fit_option_group = parser.add_argument_group('general fit options', 'how should the fit(s) be done in general?')
        fit_option_group.add_argument('--max_iter',                 default=1000,                    type=int,
                                      help='Maximum number of iterations for differential_evolution and for minimize.trust-constr')
        fit_option_group.add_argument('--normalize',                default=CONST.DEFAULT_NORMALIZE,
                                      help='Comma-separated list of parameters to normalize between their default bounds (default is everything).')
        fit_option_group.add_argument('--max_radial_warp',          default=8.,                      type=float,
                                      help='Maximum amount of radial warp to use for constraint')
        fit_option_group.add_argument('--max_tangential_warp',      default=4.,                      type=float,
                                      help='Maximum amount of radial warp to use for constraint')
        fit_option_group.add_argument('--print_every',              default=100,                     type=int,
                                      help='How many iterations to wait between printing minimization progress')
    #group for fit parameter options (which to use, what their bounds are, etc.)
    if fitpars:
        fitpar_option_group = parser.add_argument_group('fit parameter options', 'How should the fit parameters be treated?')
        fitpar_option_group.add_argument('--fixed',                    default=CONST.DEFAULT_FIXED,
                                         help='Comma-separated list of parameters to keep fixed during fitting')
        fitpar_option_group.add_argument('--init_pars',
                                         help='Comma-separated list of initial parameter name=value pairs to use in lieu of defaults.')
        fitpar_option_group.add_argument('--init_bounds',
                                         help='Comma-separated list of parameter name=low_bound:high_bound pairs to use in lieu of defaults.')
        fitpar_option_group.add_argument('--float_p1p2_to_polish',     action='store_true',
                                         help="""Add this flag to float p1 and p2 in the polishing minimization 
                                              (regardless of whether they are in the list of fixed parameters)""")
        fitpar_option_group.add_argument('--p1p2_polish_lasso_lambda', default=0.0,                 type=float,
                                         help="""Lambda magnitude parameter for the LASSO constraint on p1 and p2 in the polishing minimization
                                              (if those parameters will float then)""")
    #group for how to find overlap octets to use
    octet_finding_group = parser.add_argument_group('octet finding', 'information to find octets for the slide')
    octet_finding_group.add_argument('--octet_run_dir', 
                                     help=f'Path to a previously-created workingdir that contains a [slideID]_{CONST.OCTET_OVERLAP_CSV_FILE_NAMESTEM} file')
    octet_finding_group.add_argument('--octet_file',
                                     help='Path to a previously-created file of octets to use')
    octet_finding_group.add_argument('--threshold_file_dir',
                                     help='Path to the directory holding the background threshold file created for the slide in question')
    octet_finding_group.add_argument('--req_pixel_frac', default=0.85, type=float,
                                     help="What fraction of an overlap image's pixels must be above the threshold to accept it in a valid octet")
    #group for organizing and splitting into jobs
    if job_organization :
        job_organization_group = parser.add_argument_group('job organization', 'how should the group of jobs be organized?')
        job_organization_group.add_argument('--octet_selection', default='random_2',
                                            help='String for how to select octets for each job: "first_n" or "random_n".')
        job_organization_group.add_argument('--workers',         default=None,       type=int,
                                            help='Number of CPUs to use in the multiprocessing pool (defaults to all available)')
    #group for other run options
    run_option_group = parser.add_argument_group('run options', 'other options for this run')
    run_option_group.add_argument('--layer', default=1, type=int,         
                                  help='Image layer to use (indexed from 1)')

#helper function to check directory command-line arguments
def checkDirArgs(args) :
    #rawfile_top_dir/[slideID] must exist for every requested slide
    if hasattr(args,'slideID') :
        ids_to_check=[args.slideID]
    elif hasattr(args,'slideIDs') :
        ids_to_check=args.slideIDs
    else :
        raise ValueError('ERROR: neither slideID nor slideIDs are in arguments!')
    for sid in ids_to_check :
        rawfile_dir = pathlib.Path(args.rawfile_top_dir).absolute() / sid
        if not rawfile_dir.is_dir() :
            raise ValueError(f'ERROR: rawfile directory {rawfile_dir} does not exist!')
    #root dir must exist, with subdirectories for each slide
    if not (pathlib.Path(args.root_dir)).is_dir() :
        raise ValueError(f'ERROR: root_dir argument ({args.root_dir}) does not point to a valid directory!')
    for sid in ids_to_check :
        if not (pathlib.Path(args.root_dir)/sid).is_dir() :
            raise ValueError(f'ERROR: root directory has no subdirectory for slide {sid}!')
    #if images are to be corrected for exposure time, exposure time correction file must exist
    if (not args.skip_exposure_time_correction) :   
        if not (pathlib.Path(args.exposure_time_offset_file)).is_file() :
            raise ValueError(f'ERROR: exposure_time_offset_file {args.exposure_time_offset_file} does not exist!')
    #make sure the flatfield file exists
    if (not args.skip_flatfielding) :
        if not (pathlib.Path(args.flatfield_file)).is_file() :
            raise ValueError(f'ERROR: flatfield_file ({args.flatfield_file}) does not exist!')
    #the octet file must exist if it's to be used
    if args.octet_file is not None and not (pathlib.Path(args.octet_file)).is_file() :
        raise FileNotFoundError(f'ERROR: octet file {args.octet_file} does not exist!')
    #the octet run directory must exist if it's to be used
    if args.octet_run_dir is not None and not (pathlib.Path(args.octet_run_dir)).is_dir() :
        raise ValueError(f'ERROR: octet_run_dir ({args.octet_run_dir}) does not exist!')
    #the threshold file dir must exist if it's to be used
    if args.threshold_file_dir is not None and not (pathlib.Path(args.threshold_file_dir)).is_dir() :
        raise ValueError(f'ERROR: threshold_file_dir ({args.threshold_file_dir}) does not exist!')
    #create the working directory if it doesn't already exist
    if not (pathlib.Path(args.workingdir)).is_dir() :
        pathlib.Path.mkdir(pathlib.Path(args.workingdir))

#helper function to check the ``fixed'' command line argument
def checkFixedArg(args,parse=False) :
    #the parameter fixing string must correspond to some combination of options
    fixed_arg_parsed = split_csv_to_list(args.fixed)
    fix_cxcy   = 'cx' in fixed_arg_parsed and 'cy' in fixed_arg_parsed
    fix_fxfy   = 'fx' in fixed_arg_parsed and 'fy' in fixed_arg_parsed
    fix_k1k2k3 = 'k1' in fixed_arg_parsed and 'k2' in fixed_arg_parsed and 'k3' in fixed_arg_parsed
    fix_p1p2   = 'p1' in fixed_arg_parsed and 'p2' in fixed_arg_parsed
    if fixed_arg_parsed!=[''] and len(fixed_arg_parsed)!=2*sum([fix_cxcy,fix_fxfy,fix_p1p2])+(3*int(fix_k1k2k3)) :
        raise ValueError(f'ERROR: Fixed parameters argument ({args.fixed}) does not result in a valid fixed parameter condition!')
    if parse :
        args.fixed = fixed_arg_parsed

#helper function to make sure necessary directories exist and that the input choice of fixed parameters is valid
def checkDirAndFixedArgs(args,parse=False) :
    checkDirArgs(args)
    checkFixedArg(args,parse)

#Helper function to load a single raw file, correct its illumination with a flatfield layer, smooth it, 
#and return information needed to create a new WarpImage
#meant to be run in parallel
def loadRawImageWorker(rfp,m,n,nlayers,layer,flatfield,med_et,offset,overlaps,rectangles,root_dir,smoothsigma,return_dict=None,return_dict_key=None) :
    #get the raw image
    rawimage = (get_raw_as_hwl(rfp,m,n,nlayers))[:,:,layer-1]   
    #correct the raw image for exposure time if requested
    if med_et is not None and offset is not None :
        exp_time = (getExposureTimesByLayer(rfp,root_dir))[layer-1]
        rawimage = correctImageLayerForExposureTime(rawimage,exp_time,med_et,offset)
    #correct the raw image with the flatfield   
    if flatfield is not None :
        rawimage = correctImageLayerWithFlatfield(rawimage,flatfield)
    rfkey = (((pathlib.Path.resolve(pathlib.Path(rfp)))).name).split('.')[0]
    #find out if this image should be masked when skipping the corner overlaps
    if overlaps is not None and rectangles is not None :
        is_corner_only=True
        this_rect = [r for r in rectangles if r.file.split('.')[0]==rfkey]
        assert len(this_rect)==1; this_rect=this_rect[0]
        this_rect_number = this_rect.n
        this_rect_index = rectangles.index(this_rect)
        for tag in [o.tag for o in overlaps if o.p1==this_rect_number or o.p2==this_rect_number] :
            if tag not in CONST.CORNER_OVERLAP_TAGS :
                is_corner_only=False
                break
    else :
        is_corner_only=False #default is to consider every image
        this_rect_index = -1
    #if requested, smooth the image and add it to the list, otherwise just add it to the list
    image_to_add = rawimage
    if smoothsigma is not None :
        image_to_add = cv2.GaussianBlur(image_to_add,(0,0),smoothsigma,borderType=cv2.BORDER_REPLICATE)
    return_item = {'rfkey':rfkey,'image':image_to_add,'is_corner_only':is_corner_only,'list_index':this_rect_index}
    if return_dict is not None and return_dict_key is not None:
        return_dict[return_dict_key]=return_item
    else :
        return return_item

#helper function to find the limit on a parameter that produces the maximum warp
def findDefaultParameterLimit(parindex,parincrement,warplimit,warpamtfunc,testpars) :
    warpamt=0.; testparval=0.
    while warpamt<warplimit :
        testparval+=parincrement
        testpars[parindex]=testparval
        warpamt=warpamtfunc(tuple(testpars))
    return testparval

#helper function to make the default list of parameter constraints
def buildDefaultParameterBoundsDict(warp,max_rad_warp,max_tan_warp) :
    bounds = {}
    # cx/cy bounds are +/- 25% of the center point
    bounds['cx']=(0.5*(warp.n/2.),1.5*(warp.n/2.))
    bounds['cy']=(0.5*(warp.m/2.),1.5*(warp.m/2.))
    # fx/fy bounds are +/- 2% of the nominal values 
    bounds['fx']=(0.98*CONST.MICROSCOPE_OBJECTIVE_FOCAL_LENGTH,1.02*CONST.MICROSCOPE_OBJECTIVE_FOCAL_LENGTH)
    bounds['fy']=(0.98*CONST.MICROSCOPE_OBJECTIVE_FOCAL_LENGTH,1.02*CONST.MICROSCOPE_OBJECTIVE_FOCAL_LENGTH)
    # k1/k2/k3 and p1/p2 bounds are 2x those that would produce the max radial and tangential warp, respectively, with all others zero
    # (except k1 can't be negative)
    testpars=[warp.cx,warp.cy,warp.fx,warp.fy,0.,0.,0.,0.,0.]
    maxk1 = findDefaultParameterLimit(4,1,max_rad_warp,warp.maxRadialDistortAmount,copy.deepcopy(testpars))
    bounds['k1']=(0.,2.0*maxk1)
    maxk2 = findDefaultParameterLimit(5,1000,max_rad_warp,warp.maxRadialDistortAmount,copy.deepcopy(testpars))
    bounds['k2']=(-2.0*maxk2,2.0*maxk2)
    maxk3 = findDefaultParameterLimit(6,10000000,max_rad_warp,warp.maxRadialDistortAmount,copy.deepcopy(testpars))
    bounds['k3']=(-2.0*maxk3,2.0*maxk3)
    maxp1 = findDefaultParameterLimit(7,0.01,max_tan_warp,warp.maxTangentialDistortAmount,copy.deepcopy(testpars))
    bounds['p1']=(-2.0*maxp1,2.0*maxp1)
    maxp2 = findDefaultParameterLimit(8,0.01,max_tan_warp,warp.maxTangentialDistortAmount,copy.deepcopy(testpars))
    bounds['p2']=(-2.0*maxp2,2.0*maxp2)
    return bounds
