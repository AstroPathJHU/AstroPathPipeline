#imports
from astropath.flatfield.utilities import chunkListOfFilepaths, readImagesMT
from astropath.utilities.img_file_io import getImageHWLFromXMLFile, getSlideMedianExposureTimesByLayer, LayerOffset
from astropath.utilities.tableio import readtable, writetable
from astropath.utilities.miscpath import cd
from astropath.scripts.untested_scripts.utilities import addCommonArgumentsToParser
from argparse import ArgumentParser
from scipy.ndimage.filters import convolve
import numpy as np, multiprocessing as mp
import logging, pathlib, glob, cv2, dataclasses

#constants
RAWFILE_EXT        = '.Data.dat'
LAYERS             = [5,11,21,29,34]
LOCAL_MEAN_KERNEL  = np.array([[0.0,0.2,0.0],
                               [0.2,0.2,0.2],
                               [0.0,0.2,0.0]])
OPEN_EL            = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
SUBIMAGE_GRID_SIZE = 12

#logger
logger = logging.getLogger("subimage_laplacian_info")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s  [%(funcName)s]","%Y-%m-%d %H:%M:%S"))
logger.addHandler(handler)

#subimage information helper class
@dataclasses.dataclass(eq=False)
class SubImageInfo :
    image_key          : str
    x0                 : int
    width              : int
    y0                 : int
    height             : int
    laplacian_var      : float
    norm_laplacian_var : float
    n_pixels           : int

#################### HELPER FUNCTIONS ####################

#helper function to make sure all the necessary information is available from the command line arguments
def checkArgs(args) :
    #rawfile_top_dir/[slideID] must exist
    rawfile_dir = pathlib.Path(f'{args.rawfile_top_dir}/{args.slideID}')
    if not pathlib.Path.is_dir(rawfile_dir) :
        raise ValueError(f'ERROR: rawfile directory {rawfile_dir} does not exist!')
    #root dir must exist
    if not pathlib.Path.is_dir(pathlib.Path(args.root_dir)) :
        raise ValueError(f'ERROR: root_dir argument ({args.root_dir}) does not point to a valid directory!')
    #images must be corrected for exposure time, and exposure time correction file must exist
    if (args.skip_exposure_time_correction) :   
        raise ValueError('ERROR: exposure time offset file must be provided.')
    if not pathlib.Path.is_file(pathlib.Path(args.exposure_time_offset_file)) :
        raise ValueError(f'ERROR: exposure_time_offset_file {args.exposure_time_offset_file} does not exist!')
    #need the threshold file
    if args.threshold_file_dir is None :
        raise ValueError('ERROR: must provide a threshold file dir.')
    tfp = pathlib.Path(f'{args.threshold_file_dir}/{args.slideID}_background_thresholds.txt')
    if not pathlib.Path.is_file(tfp) :
        raise ValueError(f'ERROR: threshold file path {tfp} does not exist!')
    #create the working directory if it doesn't already exist
    if not pathlib.Path.is_dir(pathlib.Path(args.workingdir)) :
        pathlib.Path.mkdir(pathlib.Path(args.workingdir))

#helper function to calculate and add the subimage infos for a single image to a shared dictionary (run in parallel)
def getSubImageInfosWorker(img_layers,dims,key,thresholds,return_lists) :
    sub_image_height = dims[0]/SUBIMAGE_GRID_SIZE
    sub_image_width  = dims[1]/SUBIMAGE_GRID_SIZE
    #for each layer in question
    for i,ln in enumerate(LAYERS) :
        img_layer = img_layers[i]
        return_list = return_lists[i]
        #build the signal mask and the (normalized) laplacian images
        img_mask = cv2.morphologyEx((np.where(img_layer>thresholds[i],1,0)).astype(np.uint8),cv2.MORPH_OPEN,OPEN_EL,borderType=cv2.BORDER_REPLICATE)
        img_laplacian = cv2.Laplacian(img_layer,cv2.CV_32F,borderType=cv2.BORDER_REPLICATE)
        img_lap_norm = convolve(img_layer,LOCAL_MEAN_KERNEL)
        img_norm_lap = img_laplacian/img_lap_norm
        #for each subimage in the grid
        for i_subimage_x in range(SUBIMAGE_GRID_SIZE) :
            sixpmin = 0 if i_subimage_x==0 else round(i_subimage_x*sub_image_width)
            sixpmax = dims[1] if i_subimage_x==SUBIMAGE_GRID_SIZE-1 else round((i_subimage_x+1)*sub_image_width)
            for i_subimage_y in range(SUBIMAGE_GRID_SIZE) :
                siypmin = 0 if i_subimage_y==0 else round(i_subimage_y*sub_image_height)
                siypmax = dims[0] if i_subimage_y==SUBIMAGE_GRID_SIZE-1 else round((i_subimage_y+1)*sub_image_height)
                simask = img_mask[siypmin:siypmax,sixpmin:sixpmax]
                sinp = np.sum(simask)
                if sinp>0 :
                    silv = ((img_laplacian[siypmin:siypmax,sixpmin:sixpmax])[simask==1]).var()
                    sinlv = ((img_norm_lap[siypmin:siypmax,sixpmin:sixpmax])[simask==1]).var()
                    return_list.append(SubImageInfo(key,sixpmin,sixpmax-sixpmin,siypmin,siypmax-siypmin,silv,sinlv,sinp))

#helper function to get a dictionary keyed by layer number of the subimage infos for a chunk of files
def getSubImageInfosForChunk(fris,dims,metsbl,etcobl,thresholds) :
    #get the image arrays
    img_arrays = readImagesMT(fris,smoothed=False,med_exposure_times_by_layer=metsbl,et_corr_offsets_by_layer=etcobl)
    #get all of the subimage info objects
    manager = mp.Manager()
    return_lists = []
    for ln in LAYERS :
        return_lists.append(manager.list())
    procs = []
    for i,im_array in enumerate(img_arrays) :
        msg = f'Getting subimage information for {fris[i].rawfile_path}'
        logger.info(msg)
        key = ((fris[i].rawfile_path).name).rstrip(RAWFILE_EXT)
        img_layers = [im_array[:,:,ln-1] for ln in LAYERS]
        p = mp.Process(target=getSubImageInfosWorker,args=(img_layers,dims,key,thresholds,return_lists))
        procs.append(p)
        p.start()
    for proc in procs:
        proc.join()
    #return just a regular dictionary of all the subimage laplacian info objects keyed by layer number
    ret = {}
    for i,ln in enumerate(LAYERS) :
        ret[ln]=list(return_lists[i])
    return ret

#################### MAIN SCRIPT ####################

def main(args=None) :
    parser = ArgumentParser()
    #add the common options to the parser
    addCommonArgumentsToParser(parser,flatfielding=False,warping=False)
    #threshold file directory
    parser.add_argument('--threshold_file_dir',
                        help='Path to the directory holding the slide [slideID]_background_thresholds.txt file')
    #group for other run options
    run_option_group = parser.add_argument_group('run options', 'other options for this run')
    run_option_group.add_argument('--n_threads', default=10, type=int,
                                  help='Maximum number of threads to use in reading files and processing into subimages')
    run_option_group.add_argument('--max_files', default=-1, type=int,
                                  help='Maximum number of files to use (default = -1 runs all files for the requested slide)')
    args = parser.parse_args(args=args)
    #check the arguments
    checkArgs(args)
    #get all the rawfile paths
    with cd(pathlib.Path(f'{args.rawfile_top_dir}/{args.slideID}')) :
        all_rfps = [pathlib.Path(f'{args.rawfile_top_dir}/{args.slideID}/{fn}') for fn in glob.glob(f'*{RAWFILE_EXT}')]
    if args.max_files>0 :
        all_rfps=all_rfps[:args.max_files]
    #get the correction information stuff
    dims   = getImageHWLFromXMLFile(args.root_dir,args.slideID)
    for ln in LAYERS :
        if ln not in range(1,dims[-1]+1) :
            raise RuntimeError(f'ERROR: images have dimensions {dims} but layers {LAYERS} are needed.')
    metsbl = getSlideMedianExposureTimesByLayer(args.root_dir,args.slideID)
    etcobl = [lo.offset for lo in readtable(args.exposure_time_offset_file,LayerOffset)]
    with open(pathlib.Path(f'{args.threshold_file_dir}/{args.slideID}_background_thresholds.txt')) as fp :
        bgtbl = [int(v) for v in fp.readlines() if v!='']
    if len(bgtbl)!=dims[-1] :
        raise RuntimeError(f'ERROR: found {len(bgtbl)} background thresholds but images have {dims[-1]} layers!')
    thresholds = [bgtbl[ln-1] for ln in LAYERS]
    #chunk up the rawfile read information 
    fri_chunks = chunkListOfFilepaths(all_rfps,dims,args.root_dir,args.n_threads)
    #get the subimage infos for each chunk
    all_siis = {}
    for ln in LAYERS :
        all_siis[ln] = []
    for fri_chunk in fri_chunks :
        new_sii_dict = getSubImageInfosForChunk(fri_chunk,dims,metsbl,etcobl,thresholds)
        for ln,siis in new_sii_dict.items() :
            all_siis[ln]+=siis
    #write out the final tables
    for ln in LAYERS :
        fn = f'{args.slideID}_layer_{ln}_subimage_laplacians.csv'
        with cd(args.workingdir) :
            writetable(fn,all_siis[ln])

if __name__=='__main__' :
    main()
