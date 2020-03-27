#imports 
from .warpfitter import WarpFitter
from .alignmentset import AlignmentSet
from argparse import ArgumentParser
from scipy import stats
import os, copy, gc, logging, matplotlib.pyplot as plt, seaborn as sns
import cProfile

#constants
logger = logging.getLogger("warpfitter")
DEFAULT_OVERLAPS = '-999'
DEFAULT_OCTETS   = '-999'
DEFAULT_CHUNKS   = '-999'
REJECTED_OVERLAP_IMAGE_DIR_NAME = 'rejected_overlap_images'

#################### PARSE ARGUMENTS ####################

#parser callback function to split a string of comma-separated values into a list
def split_csv_to_list(value) :
    return value.split(',')

#parser callback function to split a string of comma-separated values into a list of integers
def split_csv_to_list_of_ints(value) :
    try :
        return [int(v) for v in value.split(',')]
    except ValueError :
        raise ValueError(f'Option value {value} is expected to be a comma-separated list of integers!')

#define and get the command-line arguments
parser = ArgumentParser()
#positional arguments
parser.add_argument('mode',        help='Operation to perform', choices=['fit','show_octets','show_chunks'])
parser.add_argument('sample',      help='Name of the data sample to use')
parser.add_argument('rawfile_dir', help='Path to the directory containing the "[sample_name]/*.raw" files')
parser.add_argument('root1_dir',   help='Path to the directory containing "[sample name]/dbload" subdirectories')
parser.add_argument('root2_dir',   help='Path to the directory containing the "[sample_name]/*.fw01" files to use for initial alignment')
#optional arguments
parser.add_argument('--working_dir',         default='warpfit_test',
    help='Path to (or name of) the working directory that will be created')
parser.add_argument('--overlaps',            default=DEFAULT_OVERLAPS, type=split_csv_to_list_of_ints,         
    help='Comma-separated list of numbers (n) of the overlaps to use (two-element defines a range)')
parser.add_argument('--octets',              default=DEFAULT_OCTETS,   type=split_csv_to_list_of_ints,         
    help='Comma-separated list of overlap octet indices (ordered by n of octet central rectangle) to use')
parser.add_argument('--chunks',              default=DEFAULT_CHUNKS,   type=split_csv_to_list_of_ints,         
    help='Comma-separated list of overlap octet chunk indices (ordered by n of first octet central rectangle) to use')
parser.add_argument('--layer',               default='1',         type=int,         
    help='Image layer to use (indexed from 1)')
parser.add_argument('--fixed',               default='',          type=split_csv_to_list,         
    help='Comma-separated list of parameters to keep fixed during fitting')
parser.add_argument('--max_radial_warp',     default=10.,         type=float,
    help='Maximum amount of radial warp to use for constraint')
parser.add_argument('--max_tangential_warp', default=10.,         type=float,
    help='Maximum amount of radial warp to use for constraint')
parser.add_argument('--print_every',         default=100,          type=int,
    help='Maximum amount of radial warp to use for constraint')
parser.add_argument('--max_iter',            default=1000,        type=int,
    help='Maximum number of iterations for differential_evolution and for minimize.trust-constr')
args = parser.parse_args()

#################### HELPER FUNCTIONS ####################

# Helper function that produces the visualizations of the rejected overlaps
def makeImagePixelPlots(overlaps) :
    for overlap in overlaps :
        f,(ax1,ax2) = plt.subplots(1,2)
        f.set_size_inches(10.,5.)
        im = overlap.getimage(normalize=1000.,shifted=False)
        hist = im.ravel()
        ax1.imshow(im)
        sns.distplot(hist,fit=stats.norm,kde=False,ax=ax2)
        (mu,sigma) = stats.norm.fit(hist)
        ax2.set_xlabel('normalized pixel value')
        ax2.set_xlabel('frequency')
        txt1=f'{len(hist)} pixels: mu={mu:.03f}, sigma={sigma:.03f}'
        txt2=f'exit code = {overlap.result.exit}'
        ax2.text(0.5,0.8,txt1,horizontalalignment='center',verticalalignment='center',transform=ax2.transAxes)
        ax2.text(0.5,0.7,txt2,horizontalalignment='center',verticalalignment='center',transform=ax2.transAxes)
        fn = f'overlap_{overlap.n}.png'
        plt.savefig(fn)
        plt.close()

# Helper function to get the dictionary of octets
def getSampleOctets(root1,root2,samp,working_dir,save_plots=False) :
    #create the alignment set and run its alignment
    logger.info("Performing an initial alignment to find this sample's valid octets/chunks...")
    a = AlignmentSet(args.root1_dir,args.root2_dir,args.sample)
    a.getDAPI(writeimstat=False)
    whole_sample_meanimage = a.meanimage
    result = a.align(write_result=False)
    #get the list of overlaps
    overlaps = a.overlaps
    #filter out any that could not be aligned
    good_overlaps = []
    rejected_overlaps = []
    for overlap in overlaps :
        if overlap.result.exit==0 :
            good_overlaps.append(overlap)
        else :
            logger.info(f'overlap number {overlap.n} rejected: alignment status {overlap.result.exit}.')
            rejected_overlaps.append(overlap)
    logger.info(f'Found a total of {len(good_overlaps)} good overlaps from an original set of {len(overlaps)}')
    if save_plots :
        init_dir = os.getcwd()
        if not os.path.isdir(working_dir) :
            os.mkdir(working_dir)
        os.chdir(working_dir)
        if not os.path.isdir(REJECTED_OVERLAP_IMAGE_DIR_NAME) :
            os.mkdir(REJECTED_OVERLAP_IMAGE_DIR_NAME)
        os.chdir(REJECTED_OVERLAP_IMAGE_DIR_NAME)
        try :
            makeImagePixelPlots(rejected_overlaps)
        except Exception as e :
            raise e
        finally :
            os.chdir(init_dir)
    #find the overlaps that form full octets (indexed by p1 number)
    octets = {}
    #begin by getting the set of all p1s
    p1s = set([o.p1 for o in good_overlaps])
    #for each p1, if there are eight good overlaps it forms an octet
    for p1 in p1s :
        overlapswiththisp1 = [o for o in good_overlaps if o.p1==p1]
        if len(overlapswiththisp1)==8 :
            ons = [o.n for o in overlapswiththisp1]
            logger.info(f'octet found with p1={p1} (overlaps #{min(ons)}-{max(ons)}).')
            octets[p1]=overlapswiththisp1
    msg=f'{len(octets)} total octets found'
    if save_plots :
        msg+=f'; rejected overlap images in {os.path.join(working_dir,REJECTED_OVERLAP_IMAGE_DIR_NAME)}'
    logger.info(msg+'.')
    return octets, whole_sample_meanimage

# Recursive function to get groups of interconnected octets
def getRectangleConnectedOctets(all_octets,rects_done,rect) :
    if rect in rects_done :
        return set(rects_done), None
    rects_done.add(rect)
    if rect not in all_octets.keys() :
        return set(rects_done), None
    this_rect_octet = all_octets[rect]
    octets = [this_rect_octet]
    for overlap in this_rect_octet :
        rects_done, toadd = getRectangleConnectedOctets(all_octets,set(rects_done),overlap.p2)
        if toadd is not None :
            for otoadd in toadd :
                octets.append(otoadd)
    return set(rects_done),octets

# Helper function to return the list of octet chunks
def getSampleChunks(octets) :
    octet_chunks = []
    #use each octet as a starting point for a call to the recursive function
    rects_checked = set([])
    for op1,octet in octets.items() :
        if op1 in rects_checked :
            continue
        rects_searched,chunk = getRectangleConnectedOctets(octets,rects_checked,op1)
        rects_checked = set(list(rects_checked)+list(rects_searched))
        octet_chunks.append(chunk)
    logger.info(f'{len(octet_chunks)} total octet chunks found:')
    for i,chunk in enumerate(octet_chunks,start=1) :
        chunkp1s = f'  chunk {i} is made of octets surrounding rectangles '
        for centralp1 in [octet[0].p1 for octet in chunk] :
            chunkp1s+=f'{centralp1}, '
        logger.info(chunkp1s[:-2]+'.')
        msg = f'  chunk {i} will contain '
        chunkolaps=''
        nolaps = 0
        for octet in chunk :
            thisoctetns = []
            for overlap in octet :
                nolaps+=1
                thisoctetns.append(overlap.n)
            chunkolaps+=f'{min(thisoctetns)}-{max(thisoctetns)}, '
        msg+=f'{nolaps} total overlaps: numbers {chunkolaps[:-2]}.'
        logger.info(msg)
    return octet_chunks

# Helper function to determine the list of overlaps
def getOverlaps(args) :
    #set the overlaps variable based on which of the options was used to specify
    overlaps=[]; whole_sample_meanimage=None
    #if the overlaps are being specified then they have to be either -1 (to use all), a tuple (to use a range), or a list
    if args.overlaps!=split_csv_to_list_of_ints(DEFAULT_OVERLAPS) :
        overlaps = args.overlaps
        if overlaps==[-1] :
            overlaps = overlaps[0]
        elif len(overlaps)==2 :
            overlaps=tuple(overlaps)
    #otherwise overlaps will have to be set after finding the octets and/or chunks
    else :
        if args.root2_dir is None :
            raise ValueError(f'A root2_dir must be specified if you want the code to determine the valid octets and/or chunks for this sample!')
        if not os.path.isdir(args.root2_dir) :
            raise ValueError(f'root2_dir {args.root2_dir} is not a valid directory!')
        #get the dictionary of overlap octets
        octets, whole_sample_meanimage = getSampleOctets(args.root1_dir,args.root2_dir,args.sample,args.working_dir,(args.mode=='show_octets' or args.mode=='show_chunks'))
        if args.mode=='fit' and args.octets!=split_csv_to_list_of_ints(DEFAULT_OCTETS):
            for i,octet in enumerate([octets[key] for key in sorted(octets.keys())],start=1) :
                if i in args.octets or args.octets==[-1]:
                    logger.info(f'Adding overlaps in octet #{i}...')
                    for overlap in octet :
                        overlaps.append(overlap.n)
            if len(overlaps)!=8*len(args.octets) :
                msg =f'specified octets {args.octets} did not result in the desired set of overlaps! '
                msg+=f'(asked for {len(args.octets)} octets but found {len(overlaps)} corresponding overlaps)'
                msg+=f' there are {len(octets)} octets to choose from.'
                raise ValueError(msg)
        if args.mode=='show_chunks' or (args.mode=='fit' and overlaps==[]) :
            #find the chunks of interconnected octets
            octet_chunks = getSampleChunks(octets)
            if args.mode=='fit' and args.chunks!=split_csv_to_list_of_ints(DEFAULT_CHUNKS) :
                for i,chunk in enumerate(octet_chunks,start=1) :
                    if i in args.chunks or args.chunks==[-1] :
                        logger.info(f'Adding overlaps in chunk #{i}...')
                        for octet in chunk :
                            for overlap in octet :
                                overlaps.append(overlap.n)
                if len(overlaps)==0 :
                    msg =f'specified chunks {args.chunks} did not result in the desired set of overlaps! '
                    msg+=f'(asked for {len(args.chunks)} chunks but found {len(overlaps)} corresponding overlaps)'
                    msg+=f' there are {len(octet_chunks)} chunks to choose from.'
                    raise ValueError(msg)
    if whole_sample_meanimage is None :
        logger.info("Loading an AlignmentSet to find the meanimage over the whole sample...")
        a = AlignmentSet(args.root1_dir,args.root2_dir,args.sample)
        a.getDAPI()
        whole_sample_meanimage = a.meanimage
    return overlaps, copy.deepcopy(whole_sample_meanimage)

#################### MAIN SCRIPT ####################

#apply some checks to the arguments to make sure they're valid
#only one of "overlaps" "octets" and/or "chunks" can be specified
nspec = sum([args.overlaps!=split_csv_to_list_of_ints(DEFAULT_OVERLAPS),
             args.octets!=split_csv_to_list_of_ints(DEFAULT_OCTETS),
             args.chunks!=split_csv_to_list_of_ints(DEFAULT_CHUNKS)])
if nspec==0 :
    logger.info('No overlaps, octets, or chunks specified; will use the default of octets=[-1] to run all octets.')
    args.octets=[-1]
nspec = sum([args.overlaps!=split_csv_to_list_of_ints(DEFAULT_OVERLAPS),
             args.octets!=split_csv_to_list_of_ints(DEFAULT_OCTETS),
             args.chunks!=split_csv_to_list_of_ints(DEFAULT_CHUNKS)])
if nspec!=1 :
    raise ValueError(f'Must specify exactly ONE of overlaps, octets, or chunks! (overlaps={args.overlaps}, octets={args.octets}, chunks={args.chunks})')
#rawfile_dir/[sample] must exist
rawfile_dir = args.rawfile_dir if args.rawfile_dir.endswith(args.sample) else os.path.join(args.rawfile_dir,args.sample)
if not os.path.isdir(rawfile_dir) :
    raise ValueError(f'rawfile_dir argument ({rawfile_dir}) does not point to a valid directory!')
#root1 dir must exist
if not os.path.isdir(args.root1_dir) :
    raise ValueError(f'root1_dir argument ({args.root1_dir}) does not point to a valid directory!')
#root1 dir must be usable to find a metafile directory
metafile_dir = os.path.join(args.root1_dir,args.sample,'dbload')
if not os.path.isdir(metafile_dir) :
    raise ValueError(f'root1_dir ({args.root1_dir}) does not contain "[sample name]/dbload" subdirectories!')
#the parameter fixing string must correspond to some combination of options
fix_cxcy = 'cx' in args.fixed and 'cy' in args.fixed
fix_fxfy = 'fx' in args.fixed and 'fy' in args.fixed
fix_k1k2 = 'k1' in args.fixed and 'k2' in args.fixed
fix_p1p2 = 'p1' in args.fixed and 'p2' in args.fixed
if args.fixed!=[''] and len(args.fixed)!=2*sum([fix_cxcy,fix_fxfy,fix_k1k2,fix_p1p2]) :
    raise ValueError(f'Fixed parameters argument ({args.fixed}) does not result in a valid fixed parameter condition!')
#choice of overlaps must be valid
overlaps,whole_sample_meanimage=getOverlaps(args)
gc.collect()
if args.mode=='fit' :
    logger.info(f'Will run fit on a sample of {len(overlaps)} total overlaps.')

if args.mode=='fit' :
    #make the WarpFitter Objects
    logger.info('Initializing WarpFitter')
    fitter = WarpFitter(args.sample,rawfile_dir,metafile_dir,args.working_dir,overlaps,args.layer,whole_sample_meanimage)
    #load the raw files
    logger.info('Loading raw files')
    fitter.loadRawFiles()
    #fit the model to the data
    logger.info('Running doFit')
    result = fitter.doFit(fix_cxcy=fix_cxcy,fix_fxfy=fix_fxfy,fix_k1k2=fix_k1k2,fix_p1p2=fix_p1p2,
                          max_radial_warp=args.max_radial_warp,max_tangential_warp=args.max_tangential_warp,
                          polish=True,print_every=args.print_every,maxiter=args.max_iter)
    #cProfile.run('fitter.doFit(fix_cxcy=fix_cxcy,fix_fxfy=fix_fxfy,fix_k1k2=fix_k1k2,fix_p1p2=fix_p1p2,max_radial_warp=args.max_radial_warp,max_tangential_warp=args.max_tangential_warp,polish=True,print_every=args.print_every,maxiter=args.max_iter)')

logger.info('All done : )')

