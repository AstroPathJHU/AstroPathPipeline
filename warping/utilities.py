#imports
from .config import CONST
from .alignmentset import AlignmentSetForWarping
from ..utilities.img_file_io import getRawAsHWL, getImageHWLFromXMLFile, getExposureTimesByLayer, getMedianExposureTimeAndCorrectionOffsetForSampleLayer
from ..utilities.img_correction import correctImageLayerForExposureTime, correctImageLayerWithFlatfield
from ..utilities.tableio import readtable, writetable
from ..utilities.misc import cd, split_csv_to_list, addCommonArgumentsToParser
import numpy as np
import cv2, os, logging, dataclasses, copy, platform

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
@dataclasses.dataclass(eq=False, repr=False)
class WarpImage :
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

#helper classes to represent octets of overlaps
@dataclasses.dataclass(eq=False, repr=False)
class OverlapOctet :
    metadata_top_dir     : str
    rawfile_top_dir      : str
    sample_name          : str
    nclip                : int
    layer                : int
    threshold            : float
    p1_rect_n            : int
    olap_1_n             : int
    olap_2_n             : int
    olap_3_n             : int
    olap_4_n             : int
    olap_6_n             : int
    olap_7_n             : int
    olap_9_n             : int
    olap_8_n             : int
    opposite_olap_1_n    : int
    opposite_olap_2_n    : int
    opposite_olap_3_n    : int
    opposite_olap_4_n    : int
    opposite_olap_6_n    : int
    opposite_olap_7_n    : int
    opposite_olap_9_n    : int
    opposite_olap_8_n    : int
    olap_1_p1_pixel_frac : float
    olap_2_p1_pixel_frac : float
    olap_3_p1_pixel_frac : float
    olap_4_p1_pixel_frac : float
    olap_6_p1_pixel_frac : float
    olap_7_p1_pixel_frac : float
    olap_8_p1_pixel_frac : float
    olap_9_p1_pixel_frac : float
    olap_1_p2_pixel_frac : float
    olap_2_p2_pixel_frac : float
    olap_3_p2_pixel_frac : float
    olap_4_p2_pixel_frac : float
    olap_6_p2_pixel_frac : float
    olap_7_p2_pixel_frac : float
    olap_8_p2_pixel_frac : float
    olap_9_p2_pixel_frac : float
    @property
    def overlap_ns(self) :
        return [self.olap_1_n,self.olap_2_n,self.olap_3_n,self.olap_4_n,self.olap_6_n,self.olap_7_n,self.olap_8_n,self.olap_9_n]
    @property
    def opposite_overlap_ns(self):
        return [self.opposite_olap_1_n,self.opposite_olap_2_n,self.opposite_olap_3_n,self.opposite_olap_4_n,
                self.opposite_olap_6_n,self.opposite_olap_7_n,self.opposite_olap_8_n,self.opposite_olap_9_n]
    

#little utility class to represent a warp fit result
@dataclasses.dataclass
class WarpFitResult :
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
@dataclasses.dataclass
class FieldLog :
    sample : str
    file   : str
    rect_n : int

#little utilitiy class for logging x and y principal point shifts
@dataclasses.dataclass
class WarpShift :
    layer_n  : int
    cx_shift : float
    cy_shift : float

#utility class for logging warping parameters and the sample they come from
@dataclasses.dataclass
class WarpingSummary :
    sample_name     : str
    project         : int
    cohort          : int
    microscope_name : str
    mindate         : str
    maxdate         : str
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
    octet_finding_group = parser.add_argument_group('octet finding', 'information to find octets for the sample')
    octet_finding_group.add_argument('--octet_run_dir', 
                                     help=f'Path to a previously-created workingdir that contains a [sample]_{CONST.OCTET_OVERLAP_CSV_FILE_NAMESTEM} file')
    octet_finding_group.add_argument('--threshold_file_dir',
                                     help='Path to the directory holding the background threshold file created for the sample in question')
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
     #rawfile_top_dir/[sample] must exist
    rawfile_dir = os.path.join(args.rawfile_top_dir,args.sample)
    if not os.path.isdir(rawfile_dir) :
        raise ValueError(f'ERROR: rawfile directory {rawfile_dir} does not exist!')
    #metadata top dir must exist
    if not os.path.isdir(args.metadata_top_dir) :
        raise ValueError(f'ERROR: metadata_top_dir argument ({args.metadata_top_dir}) does not point to a valid directory!')
    #metadata top dir dir must be usable to find a metafile directory
    if not os.path.isdir(args.metadata_top_dir) :
        raise ValueError(f'ERROR: metadata_top_dir ({args.metadata_top_dir}) does not exist!')
    #if images are to be corrected for exposure time, exposure time correction file must exist and must contain the necessary file
    if (not args.skip_exposure_time_correction) :
        if not os.path.isfile(args.exposure_time_offset_file) :
            raise ValueError(f'ERROR: exposure_time_offset_file {args.exposure_time_offset_file} does not exist!')
    #make sure the flatfield file exists
    if (not args.skip_flatfielding) :
        if not os.path.isfile(args.flatfield_file) :
            raise ValueError(f'ERROR: flatfield_file ({args.flatfield_file}) does not exist!')
    #the octet run workingdir must exist if it's to be used
    if args.octet_run_dir is not None and not os.path.isdir(args.octet_run_dir) :
        raise ValueError(f'ERROR: octet_run_dir ({args.octet_run_dir}) does not exist!')
    #create the working directory if it doesn't already exist
    if not os.path.isdir(args.workingdir) :
        os.mkdir(args.workingdir)

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
def loadRawImageWorker(rfp,m,n,nlayers,layer,flatfield,med_et,offset,overlaps,rectangles,metadata_top_dir,smoothsigma,return_dict=None,return_dict_key=None) :
    #get the raw image
    rawimage = (getRawAsHWL(rfp,m,n,nlayers))[:,:,layer-1]
    #correct the raw image for exposure time if requested
    if med_et is not None and offset is not None :
        exp_time = (getExposureTimesByLayer(rfp,nlayers,metadata_top_dir))[layer-1]
        rawimage = correctImageLayerForExposureTime(rawimage,exp_time,med_et,offset)
    #correct the raw image with the flatfield
    if flatfield is not None :
        rawimage = correctImageLayerWithFlatfield(rawimage,flatfield)
    rfkey = os.path.basename(os.path.normpath(rfp)).split('.')[0]
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

# Helper function to read previously-saved octet definitions from a file
def readOctetsFromFile(octet_run_dir,rawfile_top_dir,metadata_top_dir,sample_name,layer) :
    #get the .csv file holding the octet p1s and overlaps ns
    octet_filepath = os.path.join(octet_run_dir,f'{sample_name}{CONST.OCTET_OVERLAP_CSV_FILE_NAMESTEM}')
    warp_logger.info(f'Reading octet overlaps numbers from file {octet_filepath}...')
    #read the overlap ns from the file
    octets = readtable(octet_filepath,OverlapOctet)
    for octet_olap_n in octets :
        if octet_olap_n.metadata_top_dir!=metadata_top_dir :
            msg = f'ERROR: metadata_top_dir {metadata_top_dir} passed to readOctetsFromFile does not match '
            msg+= f'{octet_olap_n.metadata_top_dir} in octet file {octet_filepath}!'
            raise(WarpingError(msg))
        if octet_olap_n.rawfile_top_dir!=rawfile_top_dir :
            msg = f'ERROR: rawfile_top_dir {rawfile_top_dir} passed to readOctetsFromFile does not match '
            msg+= f'{octet_olap_n.rawfile_top_dir} in octet file {octet_filepath}!'
            raise(WarpingError(msg))
        if octet_olap_n.sample_name!=sample_name :
            msg = f'ERROR: sample_name {sample_name} passed to readOctetsFromFile does not match '
            msg+= f'{octet_olap_n.sample_name} in octet file {octet_filepath}!'
            raise(WarpingError(msg))
        if octet_olap_n.nclip!=CONST.N_CLIP :
            msg = f'ERROR: constant nclip {CONST.N_CLIP} in readOctetsFromFile does not match '
            msg+= f'{octet_olap_n.nclip} in octet file {octet_filepath}!'
            raise(WarpingError(msg))
        if octet_olap_n.layer!=layer :
            msg = f'ERROR: layer {layer} passed to readOctetsFromFile does not match '
            msg+= f'{octet_olap_n.layer} in octet file {octet_filepath}!'
            raise(WarpingError(msg))
    octets.sort(key=lambda x:x.p1_rect_n)
    return octets

# Helper function to get the list of octets
def findSampleOctets(rtd,mtd,threshold_file_path,req_pixel_frac,samp,working_dir,layer,flatfield_file,et_offset_file) :
    #start by getting the threshold of this sample layer from the the inputted file
    with open(threshold_file_path) as tfp :
        vals = [int(l.rstrip()) for l in tfp.readlines() if l.rstrip()!='']
    threshold_value = vals[layer-1]
    #create the alignment set, correct its files, and run its alignment
    warp_logger.info("Performing an initial alignment to find this sample's valid octets...")
    img_dims = getImageHWLFromXMLFile(mtd,samp)
    flatfield = (getRawAsHWL(flatfield_file,*(img_dims),CONST.FLATFIELD_DTYPE))[:,:,layer-1] if flatfield_file is not None else None
    med_et, offset = getMedianExposureTimeAndCorrectionOffsetForSampleLayer(mtd,samp,et_offset_file,layer) if et_offset_file is not None else None
    use_GPU = platform.system()!='Darwin'
    a = AlignmentSetForWarping(mtd,rtd,samp,med_et=med_et,offset=offset,flatfield=flatfield,nclip=CONST.N_CLIP,readlayerfile=False,layer=layer,filetype='raw',useGPU=use_GPU)
    a.getDAPI()
    a.align()
    #get the list of overlaps
    overlaps = a.overlaps
    #filter out any that could not be aligned or that don't show enough bright pixels
    good_overlaps = []; rejected_overlaps = []
    for overlap in overlaps :
        if overlap.result.exit!=0 :
            warp_logger.info(f'overlap number {overlap.n} rejected: alignment status {overlap.result.exit}.')
            rejected_overlaps.append(overlap)
            continue
        ip1,ip2 = overlap.cutimages
        p1frac = (np.sum(np.where(ip1>threshold_value,1,0)))/(ip1.shape[0]*ip1.shape[1])
        p2frac = (np.sum(np.where(ip2>threshold_value,1,0)))/(ip2.shape[0]*ip2.shape[1])
        if p1frac<req_pixel_frac :
            warp_logger.info(f'overlap number {overlap.n} rejected: p1 image ({overlap.p1}) only has {100.*p1frac:.2f}% above threshold at flux = {threshold_value}.')
            rejected_overlaps.append((overlap,p1frac,p2frac))
            continue
        if p2frac<req_pixel_frac :
            warp_logger.info(f'overlap number {overlap.n} rejected: p2 image ({overlap.p2}) only has {100.*p2frac:.2f}% above threshold at flux = {threshold_value}.')
            rejected_overlaps.append((overlap,p1frac,p2frac))
            continue
        good_overlaps.append((overlap,p1frac,p2frac))
    warp_logger.info(f'Found a total of {len(good_overlaps)} good overlaps from an original set of {len(overlaps)}')
    #find the overlaps that form full octets
    octets = []
    #begin by getting the set of all p1s
    p1s = set([o[0].p1 for o in good_overlaps])
    #for each p1, if there are eight good overlaps it forms an octet
    for p1 in p1s :
        overlapswiththisp1 = [o for o in good_overlaps if o[0].p1==p1]
        if len(overlapswiththisp1)==8 :
            overlapswiththisp1.sort(key=lambda x: x[0].tag)
            ons = [o[0].n for o in overlapswiththisp1]
            op1pfs = [o[1] for o in overlapswiththisp1]
            op2pfs = [o[2] for o in overlapswiththisp1]
            opposite_ons = [[opp_o[0].n for opp_o in good_overlaps if opp_o[0].p1==o[0].p2 and opp_o[0].p2==o[0].p1][0] for o in overlapswiththisp1]
            warp_logger.info(f'octet found with p1={p1} (overlaps #{min(ons)}-{max(ons)}).')
            octets.append(OverlapOctet(mtd,rtd,samp,CONST.N_CLIP,layer,threshold_value,p1,*(ons),*(opposite_ons),*(op1pfs),*(op2pfs)))
    octets.sort(key=lambda x: x.p1_rect_n)
    #save the file of which overlaps are in each valid octet
    with cd(working_dir) :
        writetable(f'{samp}{CONST.OCTET_OVERLAP_CSV_FILE_NAMESTEM}',octets)
    #print how many octets there are 
    warp_logger.info(f'{len(octets)} total octets found.')
    #return the list of octets
    return octets

#helper function to return the octets for a sample given just the command line arguments
def getOctetsFromArguments(args) :
    octet_run_dir = os.path.abspath(args.octet_run_dir) if args.octet_run_dir is not None else args.workingdir
    if os.path.isfile(os.path.join(octet_run_dir,f'{args.sample}{CONST.OCTET_OVERLAP_CSV_FILE_NAMESTEM}')) :
        if args.threshold_file_dir is not None :
            msg = 'ERROR: an octet file exists in the working directory, but a threshold file directory was also given!'
            msg+= ' Get rid of the threshold file dir argument to use the octet file that already exists, or remove the octet file to recreate it.'
            raise WarpingError(msg)
        all_octets = readOctetsFromFile(octet_run_dir,args.rawfile_top_dir,args.metadata_top_dir,args.sample,args.layer)
    elif args.threshold_file_dir is not None :
        threshold_file_path=os.path.join(args.threshold_file_dir,f'{args.sample}{CONST.THRESHOLD_FILE_EXT}')
        all_octets = findSampleOctets(args.rawfile_top_dir,args.metadata_top_dir,threshold_file_path,args.req_pixel_frac,args.sample,
                                      args.workingdir,args.layer,args.flatfield_file,args.exposure_time_offset_file)
    else :
        raise WarpingError('ERROR: either an octet_run_dir or a threshold_file_dir must be supplied to define octets to run on!')
    warp_logger.info(f'Found a total set of {len(all_octets)} valid octets for {args.sample}')
    return all_octets

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
