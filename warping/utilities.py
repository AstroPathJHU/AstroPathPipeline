#imports
from .config import CONST
from ..alignment.alignmentset import AlignmentSetFromXML
from ..baseclasses.overlap import rectangleoverlaplist_fromcsvs
from ..utilities.img_file_io import getImageHWLFromXMLFile, getRawAsHWL, writeImageToFile
from ..utilities.misc import cd
import numpy as np, multiprocessing as mp, matplotlib.pyplot as plt
import cv2, os, logging, glob, dataclasses, copy

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

# Helper function to get the dictionary of octets
def findSampleOctets(rawfile_top_dir,dbload_top_dir,threshold_file_path,req_pixel_frac,samp,working_dir,flatfield_file,n_procs,layer) :
    #start by getting the threshold of this sample layer from the the inputted file
    with open(threshold_file_path) as tfp :
        vals = [int(l.rstrip()) for l in tfp.readlines() if l.rstrip()!='']
    threshold_value = vals[layer-1]
    #create the alignment set and run its alignment
    warp_logger.info("Performing an initial alignment to find this sample's valid octets...")
    a = AlignmentSetFromXML(dbload_top_dir,rawfile_top_dir,samp,nclip=CONST.N_CLIP,readlayerfile=False,layer=layer)
    a.getDAPI(filetype='raw')
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
    # cx/cy bounds are +/- 25% of the center point
    bounds['cx']=(0.5*(warp.n/2.),1.5*(warp.n/2.))
    bounds['cy']=(0.5*(warp.m/2.),1.5*(warp.m/2.))
    # fx/fy bounds are +/- 2% of the nominal values 
    bounds['fx']=(0.98*CONST.MICROSCOPE_OBJECTIVE_FOCAL_LENGTH,1.02*CONST.MICROSCOPE_OBJECTIVE_FOCAL_LENGTH)
    bounds['fy']=(0.98*CONST.MICROSCOPE_OBJECTIVE_FOCAL_LENGTH,1.02*CONST.MICROSCOPE_OBJECTIVE_FOCAL_LENGTH)
    # k1/k2/k3 and p1/p2 bounds are 1.5x those that would produce the max radial and tangential warp, respectively, with all others zero
    # (except k1 can't be negative)
    testpars=[warp.n/2,warp.m/2,CONST.MICROSCOPE_OBJECTIVE_FOCAL_LENGTH,CONST.MICROSCOPE_OBJECTIVE_FOCAL_LENGTH,0.,0.,0.,0.,0.]
    maxk1 = findDefaultParameterLimit(4,1,max_rad_warp,warp.maxRadialDistortAmount,copy.deepcopy(testpars))
    bounds['k1']=(0.,1.5*maxk1)
    maxk2 = findDefaultParameterLimit(5,1000,max_rad_warp,warp.maxRadialDistortAmount,copy.deepcopy(testpars))
    bounds['k2']=(-1.5*maxk2,1.5*maxk2)
    maxk3 = findDefaultParameterLimit(6,10000000,max_rad_warp,warp.maxRadialDistortAmount,copy.deepcopy(testpars))
    bounds['k3']=(-1.5*maxk3,1.5*maxk3)
    maxp1 = findDefaultParameterLimit(7,0.01,max_tan_warp,warp.maxTangentialDistortAmount,copy.deepcopy(testpars))
    bounds['p1']=(-1.5*maxp1,1.5*maxp1)
    maxp2 = findDefaultParameterLimit(8,0.01,max_tan_warp,warp.maxTangentialDistortAmount,copy.deepcopy(testpars))
    bounds['p2']=(-1.5*maxp2,1.5*maxp2)
    return bounds

#little utility class to help with making the octet overlap comparison images
class OctetComparisonVisualization :

    def __init__(self,overlaps,shifted,name_stem) :
        """
        overlaps  = list of 8 AlignmentOverlap objects to use in building the figure
        shifted   = whether the figure should be built using the shifted overlap images
        name_stem = name to use for the title and filename of the figure
        """
        self.overlaps = overlaps
        self.shifted = shifted
        self.name_stem = name_stem
        self.outer_clip = self.overlaps[0].nclip
        self.shift_clip = self.outer_clip+2
        self.normalize = CONST.OVERLAY_NORMALIZE
        self.p1_im = self.overlaps[0].images[0]/self.normalize
        self.whole_image = np.zeros((self.p1_im.shape[0],self.p1_im.shape[1],3),dtype=self.p1_im.dtype)
        self.images_stacked_mask = np.zeros(self.whole_image.shape,dtype=np.uint8)
        self.overlay_dicts = {}
        for olap in self.overlaps :
            self.overlay_dicts[olap.tag] = {'image':olap.getimage(self.normalize,self.shifted),'dx':-olap.result.dx/2.,'dy':-olap.result.dy/2.}

    def stackOverlays(self) :
        """
        Stack the overlay images into the whole image
        returns a list of tuples of (p1, code) for any overlaps that couldn't be stacked into the whole image
        """
        failed_p1s_codes = []
        #add each overlay to the total image
        for code in self.overlay_dicts.keys() :
            ret = self.__addSingleOverlap(code)
            if ret is not True :
                failed_p1s_codes.append(ret)
        #divide the total image by how many overlays are contributing at each point
        self.whole_image[self.images_stacked_mask!=0]/=self.images_stacked_mask[self.images_stacked_mask!=0]
        #fill in the holes with the p1 image in magenta
        magenta_p1 = np.array([self.p1_im,np.zeros_like(self.p1_im),0.5*self.p1_im]).transpose(1,2,0)
        self.whole_image=np.where(self.whole_image==0,magenta_p1,self.whole_image)
        return failed_p1s_codes

    def writeOutFigure(self,dirpath) :
        """
        Write out a .png of the total octet overlay image
        dirpath = directory to write the figure out to
        """ 
        f,ax = plt.subplots(figsize=(CONST.OCTET_OVERLAP_COMPARISON_FIGURE_WIDTH,
                                     np.rint((self.whole_image.shape[0]/self.whole_image.shape[1])*CONST.OCTET_OVERLAP_COMPARISON_FIGURE_WIDTH)))
        ax.imshow(self.whole_image)
        ax.set_title(self.name_stem.replace('_',' '))
        with cd(dirpath) :
            plt.savefig(f'{self.name_stem}.png')
            plt.close()

    #helper function to add a single overlap's set of overlays to the total image
    def __addSingleOverlap(self,code) :
        #figure out the total image x and y start and end points
        tix_1 = 0; tix_2 = 0; tiy_1 = 0; tiy_2 = 0
        #x positions
        if code in [3,6,9] : #left column
            tix_1 = self.outer_clip
            if self.shifted :
                tix_1+=self.shift_clip
            tix_2 = tix_1+self.overlay_dicts[code]['image'].shape[1]
        elif code in [2,8] : #center column
            tix_1 = self.outer_clip
            tix_2 = self.p1_im.shape[1]-self.outer_clip
            if self.shifted :
                tix_1+=self.shift_clip
                tix_2-=self.shift_clip
        elif code in [1,4,7] : #right column
            tix_2 = self.p1_im.shape[1]-self.outer_clip
            if self.shifted :
                tix_2-=self.shift_clip
            tix_1 = tix_2-self.overlay_dicts[code]['image'].shape[1]
        #y positions
        if code in [7,8,9] : #top row
            tiy_1 = self.outer_clip
            if self.shifted :
                tiy_1+=self.shift_clip
            tiy_2 = tiy_1+self.overlay_dicts[code]['image'].shape[0]
        elif code in [4,6] : #center row
            tiy_1 = self.outer_clip
            tiy_2 = self.p1_im.shape[0]-self.outer_clip
            if self.shifted :
                tiy_1+=self.shift_clip
                tiy_2-=self.shift_clip
        elif code in [1,2,3] : #bottom column
            tiy_2 = self.p1_im.shape[0]-self.outer_clip
            if self.shifted :
                tiy_2-=self.shift_clip
            tiy_1 = tiy_2-self.overlay_dicts[code]['image'].shape[0]
        #figure out the alignment adjustment if necessary
        dx = self.overlay_dicts[code]['dx'] if self.shifted else 0
        dy = self.overlay_dicts[code]['dy'] if self.shifted else 0
        tix_1+=dx; tix_2+=dx
        tiy_1+=dy; tiy_2+=dy
        tix_1=int(np.rint(tix_1)); tix_2=int(np.rint(tix_2)); tiy_1=int(np.rint(tiy_1)); tiy_2=int(np.rint(tiy_2))
        #add the overlay to the total image and increment the mask
        try :
            self.whole_image[tiy_1:tiy_2,tix_1:tix_2,:]+=self.overlay_dicts[code]['image']
            self.images_stacked_mask[tiy_1:tiy_2,tix_1:tix_2,:]+=1
            return True
        except Exception as e :
            fp1 = self.overlaps[0].p1
            msg=f'WARNING: overlap with p1={fp1} and code {code} could not be stacked into octet overlay comparison'
            msg+=f' and will be plotted separately. Exception: {e}'
            warp_logger.warn(msg)
            return tuple((fp1,code))

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
