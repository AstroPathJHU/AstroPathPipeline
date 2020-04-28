#imports
from .mean_image import MeanImage
from ..utilities.img_file_io import getRawAsHWL
from ..utilities.misc import cd, split_csv_to_list_of_ints
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
import numpy as np, matplotlib.pyplot as plt
import os, logging, glob, csv, random

#################### GLOBALS ####################

#image dimensions and datatype
IMG_X = None
IMG_Y = None
IMG_NLAYERS = None
IMG_DTYPE = np.uint16
FILEPATH_TEXT_FILE_NAME='filepath_log.txt'

#set up a logger
flatfield_logger = logging.getLogger("flatfield")
flatfield_logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(message)s    [%(funcName)s, %(asctime)s]"))
flatfield_logger.addHandler(handler)

#################### HELPER FUNCTIONS ####################

#helper function to make sure arguments are valid
def checkArgs(a) :
    #make sure that the samplename CSV file exists
    if not os.path.isfile(a.samplename_csv_file) :
        raise ValueError(f'ERROR: sample name CSV file {a.samplename_csv_file} does not exist!')
    #make sure that the raw file directory exist
    if not os.path.isdir(a.rawfile_top_dir) :
        raise ValueError(f'ERROR: Raw file directory {a.rawfile_top_dir} does not exist!')
    #make sure raw image dimensions is the right size
    try:
        assert len(a.raw_image_dims)==3
    except AssertionError :
        raise ValueError(f'ERROR: Raw image dimensions ({a.raw_image_dims}) must be a list of three integers in x,y,#layers!')
    #reset the global image dimensions
    global IMG_X, IMG_Y, IMG_NLAYERS
    IMG_X = a.raw_image_dims[0]
    IMG_Y = a.raw_image_dims[1]
    IMG_NLAYERS = a.raw_image_dims[2]
    #create the working directory if it doesn't already exist
    if not os.path.isdir(a.workingdir_name) :
        os.mkdir(a.workingdir_name)

#helper function to get the list of filepaths to run on based on the selection method and number of images requested
def getFilepathsToRun(a) :
    #Get the list of sample names from the file
    with open(a.samplename_csv_file,) as f:
        reader = csv.reader(f)
        all_sample_names = (list(reader))[0]
    #make sure the directories all exist
    for sn in all_sample_names :
        if not os.path.isdir(os.path.join(a.rawfile_top_dir,sn)) :
            raise ValueError(f'ERROR: sample directory {os.path.join(a.rawfile_top_dir,sn)} does not exist!')
    #get the (sorted) full list of file names in the sample to choose from
    all_image_filepaths = []
    #iterate over the samples
    for sn in all_sample_names :
        #get the list of all the filenames for this sample
        with cd(os.path.join(a.rawfile_top_dir,sn)) :
            this_sample_image_filepaths = [os.path.join(a.rawfile_top_dir,sn,fn) 
                                            for fn in glob.glob(f'{sn}_[[]*,*[]]{a.rawfile_ext}')]
            all_image_filepaths+=this_sample_image_filepaths
    all_image_filepaths.sort()
    #select the total filepaths to run on
    filepaths_to_run = None
    logstring='Will run on a sample of'
    if min(a.max_images,len(all_image_filepaths)) == len(all_image_filepaths) :
        logstring+=f' all {len(all_image_filepaths)} images (not enough images in the sample to deliver all {a.max_images} requested)'
        flatfield_logger.info(logstring)
        return all_image_filepaths
    if a.selection=='first' :
        filepaths_to_run=all_image_filepaths[:a.max_images]
        logstring+=f' the first {a.max_images} images'
    elif a.selection=='last' :
        filepaths_to_run=all_image_filepaths[-(a.max_images):]
        logstring+=f' the last {a.max_images} images'
    elif a.selection=='random' :
        random.shuffle(all_image_filepaths)
        filepaths_to_run=all_image_filepaths[:a.max_images]
        logstring+=f' {a.max_images} randomly-chosen images'
    flatfield_logger.info(logstring)
    return filepaths_to_run

#helper function to parallelize calls to getRawAsHWL
def getRawImageArray(fpt) :
    flatfield_logger.info(f'  reading file {fpt[0]} {fpt[1]}')
    raw_img_arr = getRawAsHWL(fpt[0],IMG_Y,IMG_X,IMG_NLAYERS)
    return raw_img_arr

#helper function to read and return a group of raw images with multithreading
def readImagesMT(sample_image_filepath_tuples,layerlist) :
    e = ThreadPoolExecutor(len(sample_image_filepath_tuples))
    new_img_arrays = list(e.map(getRawImageArray,[fp for fp in sample_image_filepath_tuples]))
    e.shutdown()
    if layerlist==[-1] :
        return new_img_arrays
    else :
        to_return = []
        for new_img_array in new_img_arrays :
            to_add = np.ndarray((IMG_Y,IMG_X,len(layerlist)),dtype=IMG_DTYPE)
            for i,layer in enumerate(layerlist) :
                to_add[:,:,i] = new_img_array[:,:,layer-1]
            to_return.append(to_add)
        return to_return


#################### MAIN SCRIPT ####################
def main() :
    #define and get the command-line arguments
    parser = ArgumentParser()
    #positional arguments
    parser.add_argument('samplename_csv_file', help='Path to the csv file that lists the sample names to include')
    parser.add_argument('rawfile_top_dir',     help='Path to directory that holds each of the [samplename] directories that contain raw files')
    #optional arguments
    parser.add_argument('--raw_image_dims',       default=[1344,1004,35], type=split_csv_to_list_of_ints,
        help='Comma-separated list of raw image dimensions in (x, y, #layers)')
    parser.add_argument('--layers',               default=[-1],           type=split_csv_to_list_of_ints,         
        help='Comma-separated list of image layer numbers to consider (default -1 means all layers)')
    parser.add_argument('--max_images',           default=-1,             type=int,         
        help='Number of images to load from the inputted list of samples')
    parser.add_argument('--selection',            default='random',       choices=['random','first','last'],
        help='Select "first", "last", or "random" n images (where n=max_images) from the inputted sample list. Default is "random".')
    parser.add_argument('--n_threads',            default=10,             type=int,         
        help='Number of threads to run at once in speeding up file I/O')
    parser.add_argument('--rawfile_ext',          default='.Data.dat',
        help='Extension of raw files to load (default is ".Data.dat")')
    parser.add_argument('--workingdir_name',      default='flatfield_test',
        help='Name of working directory to save created files in')
    parser.add_argument('--flatfield_image_name', default='flatfield',
        help='Stem for meanimage file names')
    args = parser.parse_args()
    checkArgs(args)
    #get the list of filepaths and break them into chunks to run in parallel
    filepaths = getFilepathsToRun(args)
    filepath_chunks = [[]]
    for i,fp in enumerate(filepaths,start=1) :
        if len(filepath_chunks[-1])>=args.n_threads :
            filepath_chunks.append([])
        filepath_chunks[-1].append((fp,f'({i} of {len(filepaths)})'))
    #Start up a new mean image
    mean_image = MeanImage(args.flatfield_image_name,IMG_X,IMG_Y,IMG_NLAYERS if args.layers==[-1] else len(args.layers),IMG_DTYPE)
    #for each chunk, get the image arrays from the multithreaded function and then add them to to stack
    flatfield_logger.info('Stacking raw images....')
    for fp_chunk in filepath_chunks :
        if len(fp_chunk)<1 :
            continue
        new_img_arrays = readImagesMT(fp_chunk,args.layers)
        for im_array in new_img_arrays :
            mean_image.addNewImage(im_array)
    #take the mean of the stacked images, smooth it and make the flatfield image by dividing each layer by its mean pixel value
    flatfield_logger.info('Getting/smoothing mean image and making flatfield....')
    mean_image.makeFlatFieldImage()
    #save the images
    flatfield_logger.info('Saving images....')
    with cd(args.workingdir_name) :
        mean_image.saveImages(args.flatfield_image_name)
    #make some visualizations of the images
    flatfield_logger.info('Saving plots....')
    with cd(args.workingdir_name) :
        mean_image.savePlots()
    #write out a text file of all the filenames that were added
    flatfield_logger.info('Writing filepath text file....')
    with cd(args.workingdir_name) :
        with open(FILEPATH_TEXT_FILE_NAME,'w') as fp :
            for path in filepaths :
                fp.write(f'{path}\n')
    flatfield_logger.info('All Done!')

if __name__=='__main__' :
    main()
    