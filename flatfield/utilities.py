from .config import *
from ..utilities.img_file_io import getRawAsHWL
from concurrent.futures import ThreadPoolExecutor

#helper function to parallelize calls to getRawAsHWL
def getRawImageArray(fpt) :
    flatfield_logger.info(f'  reading file {fpt[0]} {fpt[1]}')
    raw_img_arr = getRawAsHWL(fpt[0],fpt[2][0],fpt[2][1],fpt[2][2])
    return raw_img_arr

#helper function to parallelize getting and smoothing raw images
def getSmoothedImageArray(fpt) :
    raw_img_arr = getRawImageArray(fpt)
    flatfield_logger.info(f'  smoothing image from file {fpt[0]} {fpt[1]}')
    smoothed_img_arr = cv2.GaussianBlur(raw_img_arr,(0,0),GENTLE_GAUSSIAN_SMOOTHING_SIGMA,borderType=cv2.BORDER_REPLICATE)
    return smoothed_img_arr

#helper function to read and return a group of raw images with multithreading
#set 'smoothed' to True when calling to smooth images with gentle gaussian filter as they're read in
def readImagesMT(sample_image_filepath_tuples,smoothed=False) :
    e = ThreadPoolExecutor(len(sample_image_filepath_tuples))
    if smoothed :
        new_img_arrays = list(e.map(getSmoothedImageArray,[fp for fp in sample_image_filepath_tuples]))    
    else :
        new_img_arrays = list(e.map(getRawImageArray,[fp for fp in sample_image_filepath_tuples]))
    e.shutdown()
    return new_img_arrays

#helper function to split a list of filenames into chunks to be read in in parallel
def chunkListOfFilepaths(fps,dims,n_threads) :
    filepath_chunks = [[]]
    for i,fp in enumerate(fps,start=1) :
        if len(filepath_chunks[-1])>=n_threads :
            filepath_chunks.append([])
        filepath_chunks[-1].append((fp,f'({i} of {len(fps)})',dims))
    return filepath_chunks

#helper function to return the sample name in an whole filepath
def sampleNameFromFilepath(fp) :
    return fp.split(os.sep)[-1].split('[')[0][:-1]