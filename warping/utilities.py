#imports
from .config import CONST
from ..alignment.alignmentset import AlignmentSet
from ..baseclasses.overlap import rectangleoverlaplist_fromcsvs
from ..utilities.img_file_io import getImageHWLFromXMLFile, getRawAsHWL, writeImageToFile
from ..utilities.misc import cd
import numpy as np, multiprocessing as mp
import cv2, os, logging, glob, shutil, dataclasses

#set up the logger
warp_logger = logging.getLogger("warpfitter")
warp_logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(message)s    [%(funcName)s, %(asctime)s]"))
warp_logger.addHandler(handler)

#Class for errors encountered during warping
class WarpingError(Exception) :
    pass

#helper class to hold a rectangle's rawfile key, raw image, warped image, and tag for whether it's only relevant for overlaps that are corners
@dataclasses.dataclass(eq=False, repr=False)
class WarpImage :
    rawfile_key    : str
    raw_image      : cv2.UMat
    warped_image   : cv2.UMat
    is_corner_only : bool

#helper function to make sure necessary directories exist and that the input choice of fixed parameters is valid
def checkDirAndFixedArgs(args) :
    #rawfile_top_dir/[sample] must exist
    rawfile_dir = os.path.join(args.rawfile_top_dir,args.sample)
    if not os.path.isdir(rawfile_dir) :
        raise ValueError(f'ERROR: rawfile directory {rawfile_dir} does not exist!')
    #dbload top dir must exist
    if not os.path.isdir(args.dbload_top_dir) :
        raise ValueError(f'ERROR: dbload_top_dir argument ({args.dbload_top_dir}) does not point to a valid directory!')
    #dbload top dir dir must be usable to find a metafile directory
    metafile_dir = os.path.join(args.dbload_top_dir,args.sample,'dbload')
    if not os.path.isdir(metafile_dir) :
        raise ValueError(f'ERROR: dbload_top_dir ({args.dbload_top_dir}) does not contain "[sample name]/dbload" subdirectories!')
    #make sure the flatfield file exists
    if not os.path.isfile(args.flatfield_file) :
        raise ValueError(f'ERROR: flatfield_file ({args.flatfield_file}) does not exist!')
    #the octet run workingdir must exist if it's to be used
    if args.octet_run_dir is not None and not os.path.isdir(args.octet_run_dir) :
        raise ValueError(f'ERROR: octet_run_dir ({args.octet_run_dir}) does not exist!')
    #the threshold file must exist if it's to be used
    if args.threshold_file_dir is not None :
        if not os.path.isdir(args.threshold_file_dir) :
            raise ValueError(f'ERROR: threshold_file_dir ({args.threshold_file_dir}) does not exist!')
        tfp = os.path.join(args.threshold_file_dir,f'{args.sample}{CONST.THRESHOLD_FILE_EXT}')
        if not os.path.isfile(tfp) :
            raise ValueError(f'ERROR: threshold_file_dir does not contain a threshold file for this sample ({tfp})!')
    #if the thresholding file dir and the octet dir are both provided the user needs to disambiguate
    if args.threshold_file_dir is not None and args.octet_run_dir is not None :
        raise ValueError('ERROR: cannot specify both an octet_run_dir and a threshold_file_dir!')
    #create the working directory if it doesn't already exist
    if not os.path.isdir(args.workingdir_name) :
        os.mkdir(args.workingdir_name)
    #the parameter fixing string must correspond to some combination of options
    fix_cxcy   = 'cx' in args.fixed and 'cy' in args.fixed
    fix_fxfy   = 'fx' in args.fixed and 'fy' in args.fixed
    fix_k1k2k3 = 'k1' in args.fixed and 'k2' in args.fixed and 'k3' in args.fixed
    fix_p1p2   = 'p1' in args.fixed and 'p2' in args.fixed
    if args.fixed!=[''] and len(args.fixed)!=2*sum([fix_cxcy,fix_fxfy,fix_p1p2])+(3*int(fix_k1k2k3)) :
        raise ValueError(f'ERROR: Fixed parameters argument ({args.fixed}) does not result in a valid fixed parameter condition!')

# Helper function to read previously-saved octet definitions from a file
def readOctetsFromFile(octet_run_dir,dbload_top_dir,sample_name,layer) :
    #get the .csv file holding the octet p1s and overlaps ns
    octet_filepath = os.path.join(octet_run_dir,f'{sample_name}{CONST.OCTET_OVERLAP_CSV_FILE_NAMESTEM}')
    warp_logger.info(f'Reading octet overlaps numbers from file {octet_filepath}...')
    #make a list of overlaps for the whole sample from the dbload dir
    rol = rectangleoverlaplist_fromcsvs(os.path.join(dbload_top_dir,sample_name,'dbload'),layer_override=layer)
    octets = {}
    with open(octet_filepath,'r') as ofp :
        for line in [l.rstrip() for l in ofp.readlines()] :
            linesplit = line.split(',')
            this_olap_list = [olap for olap in rol.overlaps if olap.n in [int(val) for val in linesplit[1:]]]
            octets[int(linesplit[0])] = this_olap_list
    return octets

#Helper function to load a single raw file, correct its illumination with a flatfield layer, smooth it, 
#and return information needed to create a new WarpImage
#meant to be run in parallel
def loadRawImageWorker(rfp,m,n,nlayers,layer,flatfield_layer,overlaps=None,rectangles=None,smoothsigma=None,return_dict=None,return_dict_key=None) :
    rawimage = (((getRawAsHWL(rfp,m,n,nlayers))[:,:,layer-1])/flatfield_layer).astype(np.uint16)
    rfkey = os.path.basename(os.path.normpath(rfp)).split('.')[0]
    #find out if this image should be masked when skipping the corner overlaps
    if overlaps is not None and rectangles is not None :
        is_corner_only=True
        this_rect_number = [r.n for r in rectangles if r.file.split('.')[0]==rfkey]
        assert len(this_rect_number)==1; this_rect_number=this_rect_number[0]
        for tag in [o.tag for o in overlaps if o.p1==this_rect_number or o.p2==this_rect_number] :
            if tag not in CONST.CORNER_OVERLAP_TAGS :
                is_corner_only=False
                break
    else :
        is_corner_only=False #default is to consider every image
    #if requested, smooth the image and add it to the list, otherwise just add it to the list
    image_to_add = rawimage
    if smoothsigma is not None :
        image_to_add = cv2.GaussianBlur(image_to_add,(0,0),smoothsigma,borderType=cv2.BORDER_REPLICATE)
    return_item = {'rfkey':rfkey,'image':image_to_add,'is_corner_only':is_corner_only}
    if return_dict is not None and return_dict_key is not None:
        return_dict[return_dict_key]=return_item
    else :
        return return_item

# Helper function to extract a layer of a single raw file to a .fw## file
#meant to be run in parallel
def extractRawFileLayerWorker(fp,flatfield_layer,img_height,img_width,img_nlayers,layer=1,dtype=np.uint16) :
    rawImageDict = loadRawImageWorker(fp,img_height,img_width,img_nlayers,layer,flatfield_layer,smoothsigma=CONST.smoothsigma)
    new_fn = f'{rawImageDict["rfkey"]}{CONST.FW_EXT}{layer:02d}'
    writeImageToFile(rawImageDict['image'],new_fn)

# Helper function to extract a single layer of a sample's raw files into the workingdir/sample_name directory
def extractRawFileLayers(rawfile_top_dir,sample_name,workingdir,flatfield_file,n_procs,layer=1) :
    warp_logger.info(f'Extracting layer {layer} from all raw files in sample {sample_name} (correcting with flatfield file {flatfield_file})....')
    #get a list of all the filenames in the sample
    with cd(os.path.join(rawfile_top_dir,sample_name)) :
        all_raw_filepaths = [os.path.join(rawfile_top_dir,sample_name,fn) for fn in glob.glob(f'*{CONST.RAW_EXT}')]
    #get the image dimensions
    img_h,img_w,img_nlayers=getImageHWLFromXMLFile(rawfile_top_dir,sample_name)
    #get the corresponding layer of the flatfield file
    flatfield_layer = (getRawAsHWL(flatfield_file,img_h,img_w,img_nlayers,np.float64))[:,:,layer-1]
    #extract the given layer from each of the files into the working directory
    with cd(workingdir) :
        if not os.path.isdir(sample_name) :
            os.mkdir(sample_name)
        with cd(sample_name) :
            procs = []
            for ifp,rfp in enumerate(all_raw_filepaths,start=1) :
                warp_logger.info(f'  extracting layer {layer} from rawfile {rfp} ({ifp} of {len(all_raw_filepaths)})....')
                p = mp.Process(target=extractRawFileLayerWorker, 
                               args=(rfp,flatfield_layer,img_h,img_w,img_nlayers,layer))
                procs.append(p)
                p.start()
                if len(procs)>=n_procs :
                    for proc in procs :
                        proc.join()
            for proc in procs:
                proc.join()
    warp_logger.info('Done!')

# Helper function to get the dictionary of octets
def findSampleOctets(rawfile_top_dir,dbload_top_dir,threshold_file_path,req_pixel_frac,samp,working_dir,flatfield_file,n_procs,layer) :
    #start by getting the threshold of this sample layer from the the inputted file
    with open(threshold_file_path) as tfp :
        vals = [int(l.rstrip()) for l in tfp.readlines() if l.rstrip()!='']
    threshold_value = vals[layer-1]
    #extract the raw file layers to the working directory to run a test alignment
    extractRawFileLayers(rawfile_top_dir,samp,working_dir,flatfield_file,n_procs,layer)
    #create the alignment set and run its alignment
    warp_logger.info("Performing an initial alignment to find this sample's valid octets...")
    a = AlignmentSet(dbload_top_dir,working_dir,samp)
    a.getDAPI(writeimstat=False)
    a.align(write_result=False)
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
            rejected_overlaps.append(overlap)
            continue
        if p2frac<req_pixel_frac :
            warp_logger.info(f'overlap number {overlap.n} rejected: p2 image ({overlap.p2}) only has {100.*p2frac:.2f}% above threshold at flux = {threshold_value}.')
            rejected_overlaps.append(overlap)
            continue
        good_overlaps.append(overlap)
    warp_logger.info(f'Found a total of {len(good_overlaps)} good overlaps from an original set of {len(overlaps)}')
    #find the overlaps that form full octets (indexed by p1 number)
    octets = {}
    #begin by getting the set of all p1s
    p1s = set([o.p1 for o in good_overlaps])
    #for each p1, if there are eight good overlaps it forms an octet
    for p1 in p1s :
        overlapswiththisp1 = [o for o in good_overlaps if o.p1==p1]
        if len(overlapswiththisp1)==8 :
            ons = [o.n for o in overlapswiththisp1]
            warp_logger.info(f'octet found with p1={p1} (overlaps #{min(ons)}-{max(ons)}).')
            octets[p1]=overlapswiththisp1
    #save the file of which overlaps are in each valid octet
    with cd(working_dir) :
        with open(f'{samp}{CONST.OCTET_OVERLAP_CSV_FILE_NAMESTEM}','w') as ofp :
            for octet_p1,octet_olaps in octets.items() :
                new_line=f'{octet_p1},'
                for octet_olap_n in [oo.n for oo in octet_olaps] :
                    new_line+=f'{octet_olap_n},'
                new_line=new_line[:-1]+'\n'
                ofp.write(new_line)
    #print how many octets there are 
    warp_logger.info(f'{len(octets)} total octets found.')
    #remove the extracted layers 
    with cd(working_dir) :
        shutil.rmtree(samp)
    #return the dictionary of octets
    return octets

#helper function to find the limit on a parameter that produces the maximum warp
def findDefaultParameterLimit(parindex,parincrement,warplimit,warpamtfunc,testpars) :
    warpamt=0.; testparval=0.
    while warpamt<warplimit :
        testparval+=parincrement
        testpars[parindex]=testparval
        warpamt=warpamtfunc(testpars)
    return testparval

#helper function to make the default list of parameter constraints
def buildDefaultParameterBoundsDict(warp,max_rad_warp,max_tan_warp) :
    bounds = {}
    # cx/cy bounds are +/- 10% of the center point
    bounds['cx']=(0.8*(warp.n/2.),1.2*(warp.n/2.))
    bounds['cy']=(0.8*(warp.m/2.),1.2*(warp.m/2.))
    # fx/fy bounds are +/- 2% of the nominal values 
    bounds['fx']=(0.98*CONST.MICROSCOPE_OBJECTIVE_FOCAL_LENGTH,1.02*CONST.MICROSCOPE_OBJECTIVE_FOCAL_LENGTH)
    bounds['fy']=(0.98*CONST.MICROSCOPE_OBJECTIVE_FOCAL_LENGTH,1.02*CONST.MICROSCOPE_OBJECTIVE_FOCAL_LENGTH)
    # k1/k2/k3 and p1/p2 bounds are 1.5x those that would produce the max radial and tangential warp, respectively, with all others zero
    # (except k1 can't be negative)
    testpars=[warp.n/2,warp.m/2,CONST.MICROSCOPE_OBJECTIVE_FOCAL_LENGTH,CONST.MICROSCOPE_OBJECTIVE_FOCAL_LENGTH,0.,0.,0.,0.,0.]
    maxk1 = findDefaultParameterLimit(4,0.1,max_rad_warp,warp.maxRadialDistortAmount,testpars)
    bounds['k1']=(0.,1.5*maxk1)
    maxk2 = findDefaultParameterLimit(5,100,max_rad_warp,warp.maxRadialDistortAmount,testpars)
    bounds['k2']=(-1.5*maxk2,1.5*maxk2)
    maxk3 = findDefaultParameterLimit(8,10000,max_rad_warp,warp.maxRadialDistortAmount,testpars)
    bounds['k3']=(-1.5*maxk3,1.5*maxk3)
    maxp1 = findDefaultParameterLimit(6,0.001,max_tan_warp,warp.maxTangentialDistortAmount,testpars)
    bounds['p1']=(-1.5*maxp1,1.5*maxp1)
    maxp2 = findDefaultParameterLimit(7,0.001,max_tan_warp,warp.maxTangentialDistortAmount,testpars)
    bounds['p2']=(-1.5*maxp2,1.5*maxp2)
    return bounds
