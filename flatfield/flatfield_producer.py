#imports
from .mean_image import MeanImage
from .config import *
from ..utilities.img_file_io import getRawAsHWL
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import os, glob

#main class
class FlatfieldProducer :
	"""
	Main class used in producing the flatfield correction image
	"""
	def __init__(self,img_dims,filepaths,sample_names,n_threads) :
		"""
		img_dims     = dimensions of images in files in order as (height, width, # of layers) 
		filepaths    = list of all filepaths that will be run
		sample_names = list of names of samples that will be considered in this run
		n_threads    = max number of threads/processes to open at once
		"""
	    #get the list of filepaths and break them into chunks to run in parallel
	    filepath_chunks = [[]]
	    for i,fp in enumerate(filepaths,start=1) :
	        if len(filepath_chunks[-1])>=args.n_threads :
	            filepath_chunks.append([])
	        filepath_chunks[-1].append((fp,f'({i} of {len(filepaths)})',dims))
	    #Start up a new mean image
	    mean_image = MeanImage(args.flatfield_image_name,
	                            dims[0],dims[1],dims[2] if args.layers==[-1] else len(args.layers),IMG_DTYPE_IN,
	                            args.n_threads,args.workingdir_name,args.skip_masking)

	def findBackgroundThreshold(self) :
		pass

	def stackImages(self) :
		#for each chunk, get the image arrays from the multithreaded function and then add them to to stack
	    flatfield_logger.info('Stacking raw images....')
	    for fp_chunk in filepath_chunks :
	        if len(fp_chunk)<1 :
	            continue
	        new_img_arrays = readImagesMT(fp_chunk,args.layers)
	        mean_image.addGroupOfImages(new_img_arrays,args.save_masking_plots)

	def makeFlatField(self) :
		#take the mean of the stacked images, smooth it and make the flatfield image by dividing each layer by its mean pixel value
	    flatfield_logger.info('Getting/smoothing mean image and making flatfield....')
	    mean_image.makeFlatFieldImage()

	def writeOutInfo(self) :
		#save the images
	    flatfield_logger.info('Saving layer-by-layer images....')
	    with cd(args.workingdir_name) :
	        mean_image.saveImages(args.flatfield_image_name)
	    #make some visualizations of the images
	    flatfield_logger.info('Saving plots....')
	    mean_image.savePlots()
	    #write out a text file of all the filenames that were added
	    flatfield_logger.info('Writing filepath text file....')
	    with cd(args.workingdir_name) :
	        with open(FILEPATH_TEXT_FILE_NAME,'w') as fp :
	            for path in filepaths :
	                fp.write(f'{path}\n')
	    flatfield_logger.info('All Done!')

#################### FILE-SCOPE HELPER FUNCTIONS ####################

#helper function to parallelize calls to getRawAsHWL
def getRawImageArray(fpt) :
    flatfield_logger.info(f'  reading file {fpt[0]} {fpt[1]}')
    raw_img_arr = getRawAsHWL(fpt[0],fpt[2][0],fpt[2][1],fpt[2][2])
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
            to_add = np.ndarray((IMG_Y,IMG_X,len(layerlist)),dtype=IMG_DTYPE_IN)
            for i,layer in enumerate(layerlist) :
                to_add[:,:,i] = new_img_array[:,:,layer-1]
            to_return.append(to_add)
        return to_return