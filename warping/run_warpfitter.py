#imports 
from .warpfitter import WarpFitter
from ..alignment.alignmentset import AlignmentSet
from ..utilities.misc import cd, split_csv_to_list, split_csv_to_list_of_ints
from argparse import ArgumentParser
from scipy import stats
import os, copy, gc, logging, matplotlib.pyplot as plt, seaborn as sns
import cProfile
import multiprocessing as mp

#constants
warp_logger = logging.getLogger("warpfitter")
DEFAULT_OVERLAPS = '-999'
DEFAULT_OCTETS   = '-999'
DEFAULT_CHUNKS   = '-999'
REJECTED_OVERLAP_IMAGE_DIR_NAME = 'rejected_overlap_images'

#################### HELPER FUNCTIONS ####################

#helper function to make sure necessary directories exist and that the input choice of fixed parameters is valid
def checkDirAndFixedArgs(args) :
    #rawfile_top_dir/[sample] must exist
    rawfile_dir = os.path.join(args.rawfile_top_dir,args.sample)
    if not os.path.isdir(rawfile_dir) :
        raise ValueError(f'rawfile directory {rawfile_dir} does not exist!')
    #dbload top dir must exist
    if not os.path.isdir(args.dbload_top_dir) :
        raise ValueError(f'dbload_top_dir argument ({args.dbload_top_dir}) does not point to a valid directory!')
    #dbload top dir dir must be usable to find a metafile directory
    metafile_dir = os.path.join(args.dbload_top_dir,args.sample,'dbload')
    if not os.path.isdir(metafile_dir) :
        raise ValueError(f'dbload_top_dir ({args.dbload_top_dir}) does not contain "[sample name]/dbload" subdirectories!')
    #the parameter fixing string must correspond to some combination of options
    fix_cxcy   = 'cx' in args.fixed and 'cy' in args.fixed
    fix_fxfy   = 'fx' in args.fixed and 'fy' in args.fixed
    fix_k1k2k3 = 'k1' in args.fixed and 'k2' in args.fixed and 'k3' in args.fixed
    fix_p1p2   = 'p1' in args.fixed and 'p2' in args.fixed
    if args.fixed!=[''] and len(args.fixed)!=2*sum([fix_cxcy,fix_fxfy,fix_p1p2])+(3*int(fix_k1k2k3)) :
        raise ValueError(f'Fixed parameters argument ({args.fixed}) does not result in a valid fixed parameter condition!')

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
    warp_logger.info("Performing an initial alignment to find this sample's valid octets/chunks...")
    a = AlignmentSet(root1,root2,samp)
    a.getDAPI(writeimstat=False)
    whole_sample_meanimage = a.meanimage
    a.align(write_result=False,warpwarnings=True)
    #get the list of overlaps
    overlaps = a.overlaps
    #filter out any that could not be aligned
    good_overlaps = []
    rejected_overlaps = []
    for overlap in overlaps :
        if overlap.result.exit==0 :
            good_overlaps.append(overlap)
        else :
            warp_logger.info(f'overlap number {overlap.n} rejected: alignment status {overlap.result.exit}.')
            rejected_overlaps.append(overlap)
    warp_logger.info(f'Found a total of {len(good_overlaps)} good overlaps from an original set of {len(overlaps)}')
    if save_plots :
        if not os.path.isdir(working_dir) :
            os.mkdir(working_dir)
        with cd(working_dir):
            if not os.path.isdir(REJECTED_OVERLAP_IMAGE_DIR_NAME) :
                os.mkdir(REJECTED_OVERLAP_IMAGE_DIR_NAME)
            with cd(REJECTED_OVERLAP_IMAGE_DIR_NAME):
                makeImagePixelPlots(rejected_overlaps)
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
    msg=f'{len(octets)} total octets found'
    if save_plots :
        msg+=f'; rejected overlap images in {os.path.join(working_dir,REJECTED_OVERLAP_IMAGE_DIR_NAME)}'
    warp_logger.info(msg+'.')
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
    warp_logger.info(f'{len(octet_chunks)} total octet chunks found:')
    for i,chunk in enumerate(octet_chunks,start=1) :
        chunkp1s = f'  chunk {i} is made of octets surrounding rectangles '
        for centralp1 in [octet[0].p1 for octet in chunk] :
            chunkp1s+=f'{centralp1}, '
        warp_logger.info(chunkp1s[:-2]+'.')
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
        warp_logger.info(msg)
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
        if args.fw01_top_dir is None :
            raise ValueError('A fw01_top_dir must be specified if you want the code to determine the valid octets and/or chunks for this sample!')
        if not os.path.isdir(args.fw01_top_dir) :
            raise ValueError(f'fw01_top_dir {args.fw01_top_dir} is not a valid directory!')
        #get the dictionary of overlap octets
        octets, whole_sample_meanimage = getSampleOctets(args.dbload_top_dir,args.fw01_top_dir,args.sample,args.working_dir,(args.mode=='show_octets' or args.mode=='show_chunks'))
        if args.mode in ('fit', 'cProfile') and args.octets!=split_csv_to_list_of_ints(DEFAULT_OCTETS):
            for i,octet in enumerate([octets[key] for key in sorted(octets.keys())],start=1) :
                if i in args.octets or args.octets==[-1]:
                    warp_logger.info(f'Adding overlaps in octet #{i}...')
                    for overlap in octet :
                        overlaps.append(overlap.n)
            if len(overlaps)!=8*len(args.octets) :
                msg =f'specified octets {args.octets} did not result in the desired set of overlaps! '
                msg+=f'(asked for {len(args.octets)} octets but found {len(overlaps)} corresponding overlaps)'
                msg+=f' there are {len(octets)} octets to choose from.'
                raise ValueError(msg)
        if args.mode=='show_chunks' or (args.mode in ('fit', 'cProfile') and overlaps==[]) :
            #find the chunks of interconnected octets
            octet_chunks = getSampleChunks(octets)
            if args.mode in ('fit', 'cProfile') and args.chunks!=split_csv_to_list_of_ints(DEFAULT_CHUNKS) :
                for i,chunk in enumerate(octet_chunks,start=1) :
                    if i in args.chunks or args.chunks==[-1] :
                        warp_logger.info(f'Adding overlaps in chunk #{i}...')
                        for octet in chunk :
                            for overlap in octet :
                                overlaps.append(overlap.n)
                if len(overlaps)==0 :
                    msg =f'specified chunks {args.chunks} did not result in the desired set of overlaps! '
                    msg+=f'(asked for {len(args.chunks)} chunks but found {len(overlaps)} corresponding overlaps)'
                    msg+=f' there are {len(octet_chunks)} chunks to choose from.'
                    raise ValueError(msg)
    if whole_sample_meanimage is None :
        warp_logger.info("Loading an AlignmentSet to find the meanimage over the whole sample...")
        a = AlignmentSet(args.dbload_top_dir,args.fw01_top_dir,args.sample)
        a.getDAPI()
        whole_sample_meanimage = a.meanimage
    return overlaps, copy.deepcopy(whole_sample_meanimage)

#################### MAIN SCRIPT ####################

if __name__=='__main__' :
    mp.freeze_support()
    #define and get the command-line arguments
    parser = ArgumentParser()
    #positional arguments
    parser.add_argument('mode',        help='Operation to perform', choices=['fit','show_octets','show_chunks','cProfile'])
    parser.add_argument('sample',      help='Name of the data sample to use')
    parser.add_argument('rawfile_top_dir', help='Path to the directory containing the "[sample_name]/*.Data.dat" files')
    parser.add_argument('dbload_top_dir',   help='Path to the directory containing "[sample name]/dbload" subdirectories')
    parser.add_argument('fw01_top_dir',   help='Path to the directory containing the "[sample_name]/*.fw01" files to use for initial alignment')
    #optional arguments
    parser.add_argument('--working_dir',          default='warpfit_test',
        help='Path to (or name of) the working directory that will be created')
    parser.add_argument('--overlaps',             default=DEFAULT_OVERLAPS, type=split_csv_to_list_of_ints,         
        help='Comma-separated list of numbers (n) of the overlaps to use (two-element defines a range)')
    parser.add_argument('--octets',               default=DEFAULT_OCTETS,   type=split_csv_to_list_of_ints,         
        help='Comma-separated list of overlap octet indices (ordered by n of octet central rectangle) to use')
    parser.add_argument('--chunks',               default=DEFAULT_CHUNKS,   type=split_csv_to_list_of_ints,         
        help='Comma-separated list of overlap octet chunk indices (ordered by n of first octet central rectangle) to use')
    parser.add_argument('--layer',                default='1',         type=int,         
        help='Image layer to use (indexed from 1)')
    parser.add_argument('--fixed',                default='',          type=split_csv_to_list,         
        help='Comma-separated list of parameters to keep fixed during fitting')
    parser.add_argument('--float_p1p2_to_polish', action='store_true',
        help='Add this flag to float p1 and p2 in the polishing minimization regardless of whether they are in the list of fixed parameters')
    parser.add_argument('--max_radial_warp',      default=8.,         type=float,
        help='Maximum amount of radial warp to use for constraint')
    parser.add_argument('--max_tangential_warp',  default=4.,         type=float,
        help='Maximum amount of radial warp to use for constraint')
    parser.add_argument('--lasso_lambda',         default=0.0,         type=float,
        help='Lambda magnitude parameter for the LASSO constraint on p1 and p2 if those parameters are to float in the polishing minimization')
    parser.add_argument('--print_every',          default=100,          type=int,
        help='How many iterations to wait between printing minimization progress')
    parser.add_argument('--max_iter',             default=1000,        type=int,
        help='Maximum number of iterations for differential_evolution and for minimize.trust-constr')
    args = parser.parse_args()

    #apply some checks to the arguments to make sure they're valid
    #make sure directories exist and fixed parameter choice is valid
    checkDirAndFixedArgs(args)
    #only one of "overlaps" "octets" and/or "chunks" can be specified
    nspec = sum([args.overlaps!=split_csv_to_list_of_ints(DEFAULT_OVERLAPS),
                 args.octets!=split_csv_to_list_of_ints(DEFAULT_OCTETS),
                 args.chunks!=split_csv_to_list_of_ints(DEFAULT_CHUNKS)])
    if nspec==0 :
        warp_logger.info('No overlaps, octets, or chunks specified; will use the default of octets=[-1] to run all octets.')
        args.octets=[-1]
    nspec = sum([args.overlaps!=split_csv_to_list_of_ints(DEFAULT_OVERLAPS),
                 args.octets!=split_csv_to_list_of_ints(DEFAULT_OCTETS),
                 args.chunks!=split_csv_to_list_of_ints(DEFAULT_CHUNKS)])
    if nspec!=1 :
        raise ValueError(f'Must specify exactly ONE of overlaps, octets, or chunks! (overlaps={args.overlaps}, octets={args.octets}, chunks={args.chunks})')
    #choice of overlaps must be valid
    overlaps,whole_sample_meanimage=getOverlaps(args)
    gc.collect()
    if args.mode in ('fit', 'cProfile') :
        warp_logger.info(f'Will run fit on a sample of {len(overlaps)} total overlaps.')

    if args.mode in ('fit', 'cProfile') :
        #make the WarpFitter Objects
        warp_logger.info('Initializing WarpFitter')
        rawfile_dir = os.path.join(args.rawfile_top_dir,args.sample)
        metafile_dir = os.path.join(args.dbload_top_dir,args.sample,'dbload')
        fitter = WarpFitter(args.sample,rawfile_dir,metafile_dir,args.working_dir,overlaps,args.layer,whole_sample_meanimage)
        #load the raw files
        warp_logger.info('Loading raw files')
        fitter.loadRawFiles()
        #fit the model to the data
        warp_logger.info('Running doFit')
        fix_cxcy   = 'cx' in args.fixed and 'cy' in args.fixed
        fix_fxfy   = 'fx' in args.fixed and 'fy' in args.fixed
        fix_k1k2k3 = 'k1' in args.fixed and 'k2' in args.fixed and 'k3' in args.fixed
        fix_p1p2   = 'p1' in args.fixed and 'p2' in args.fixed
        if args.mode == 'fit' :
            result = fitter.doFit(fix_cxcy=fix_cxcy,fix_fxfy=fix_fxfy,fix_k1k2k3=fix_k1k2k3,fix_p1p2_in_global_fit=fix_p1p2,fix_p1p2_in_polish_fit=(not args.float_p1p2_to_polish),
                                  max_radial_warp=args.max_radial_warp,max_tangential_warp=args.max_tangential_warp,p1p2_polish_lasso_lambda=args.lasso_lambda,
                                  polish=True,print_every=args.print_every,maxiter=args.max_iter)
        elif args.mode == 'cProfile' :
            cProfile.run("""fitter.doFit(fix_cxcy=fix_cxcy,fix_fxfy=fix_fxfy,fix_k1k2k3=fix_k1k2k3,fix_p1p2_in_global_fit=fix_p1p2,
                            fix_p1p2_in_polish_fit=(not args.float_p1p2_to_polish),max_radial_warp=args.max_radial_warp,
                            max_tangential_warp=args.max_tangential_warp,p1p2_polish_lasso_lambda=args.lasso_lambda,polish=True,
                            print_every=args.print_every,maxiter=args.max_iter)""")

    warp_logger.info('All done : )')

