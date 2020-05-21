from .config import flatfield_logger
from ..utilities.img_file_io import getRawAsHWL
from concurrent.futures import ThreadPoolExecutor
import os

#helper function to parallelize calls to getRawAsHWL
def getRawImageArray(fpt) :
    flatfield_logger.info(f'  reading file {fpt[0]} {fpt[1]}')
    raw_img_arr = getRawAsHWL(fpt[0],fpt[2][0],fpt[2][1],fpt[2][2])
    return raw_img_arr

#helper function to read and return a group of raw images with multithreading
def readImagesMT(sample_image_filepath_tuples) :
    e = ThreadPoolExecutor(len(sample_image_filepath_tuples))
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