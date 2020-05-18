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
	def __init__(self,img_dims,filepaths,sample_names,workingdir_name,skip_masking) :
		"""
		img_dims        = dimensions of images in files in order as (height, width, # of layers) 
		filepaths       = list of all filepaths that will be run
		sample_names    = list of names of samples that will be considered in this run
		workingdir_name = name of the directory to save everything in
		skip_masking    = if True, image layers won't be masked before being added to the stack
		"""
		self.dims = img_dims
		self.all_filepaths = filepaths
		self.sample_names = sample_names
	    #Start up a new mean image
	    self.mean_image = MeanImage(flatfield_image_name,self.dims[2],workingdir_name,skip_masking)

	#################### PUBLIC FUNCTIONS ####################

	def findBackgroundThreshold(self,n_threads) :
		"""
		Function to determine, using HPFs that image edges of the tissue in each slide, what threshold to use for masking out background
		n_threads = max number of threads/processes to open at once
		"""
		#find the filepaths corresponding to the edges of the tissue in the samples
		self.tissue_edge_filepaths = self.__findTissueEdgeFilepaths()
		#chunk them together to be read in parallel
		tissue_edge_fp_chunks = chunkListOfFilepaths(self.tissue_edge_filepaths,self.dims,n_threads)

	def stackImages(self,n_threads,save_masking_plots) :
		"""
		Function to mask out background and stack portions of images up
		n_threads          = max number of threads/processes to open at once
		save_masking_plots = whether to save plots of the mask overlays as they're generated
		"""
		#break the list of filepaths into chunks to run in parallel
	    filepath_chunks = chunkListOfFilepaths(self.all_filepaths,self.dims,n_threads)
		#for each chunk, get the image arrays from the multithreaded function and then add them to to stack
	    flatfield_logger.info('Stacking raw images....')
	    for fp_chunk in filepath_chunks :
	        if len(fp_chunk)<1 :
	            continue
	        new_img_arrays = readImagesMT(fp_chunk)
	        self.mean_image.addGroupOfImages(new_img_arrays,save_masking_plots)

	def makeFlatField(self) :
		"""
		Take the mean of the stacked images, smooth it and make the flatfield image by dividing each layer by its mean pixel value
		"""
	    flatfield_logger.info('Getting/smoothing mean image and making flatfield....')
	    self.mean_image.makeFlatFieldImage()

	def writeOutInfo(self,name) :
		"""
		name            = stem to use for naming files that get created
		"""
		#save the images
	    flatfield_logger.info('Saving layer-by-layer images....')
        self.mean_image.saveImages(name)
	    #make some visualizations of the images
	    flatfield_logger.info('Saving plots....')
	    self.mean_image.savePlots()
	    #write out a text file of all the filenames that were added
	    flatfield_logger.info('Writing filepath text file....')
	    with cd(self.mean_image.workingdir_name) :
	        with open(FILEPATH_TEXT_FILE_NAME,'w') as fp :
	            for path in self.all_filepaths :
	                fp.write(f'{path}\n')

	#################### PRIVATE HELPER FUNCTIONS ####################

	#helper function to return the subset of the filepath list corresponding to HPFs on the edge of tissue
	def __findTissueEdgeFilepaths(self) :
		pass

#################### FILE-SCOPE HELPER FUNCTIONS ####################

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
        filepath_chunks[-1].append((fp,f'({i} of {len(fps)})',self.dims))
    return filepath_chunks