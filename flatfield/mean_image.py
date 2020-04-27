#imports
from ..utilities.img_file_io import writeImageToFile
import numpy as np
import skimage.filters, skimage.util

FILE_EXT='.bin'

class FlatFieldError(Exception) :
    """
    Class for errors encountered during flatfielding
    """
    pass

class MeanImage :
	"""
	Class to hold an image that is the mean of a bunch of stacked raw images 
	"""
	def __init__(self,name,x,y,nlayers,dtype) :
		"""
		name    = stem to use for naming files that get created
		x       = x dimension of images (pixels)
		y       = y dimension of images (pixels)
		nlayers = number of layers in images
		dtype   = datatype of image arrays that will be stacked
		"""
		self.namestem = name
		self.image_stack = np.zeros((y,x,nlayers),dtype=dtype)
		self.n_images_stacked = 0
		self.mean_image=None
		self.smoothed_mean_image=None
		self.flatfield_image=None

	def addNewImage(self,im_array) :
		"""
		im_array = array of new image to add to the stack
		"""
		self.image_stack+=im_array
		self.n_images_stacked+=1

	def takeMean(self) :
		self.mean_image = self.image_stack/self.n_images_stacked

	def smoothMeanImage(self,smoothsigma=12.5,smoothtruncate=4.0) :
		"""
		smoothsigma    = sigma (in pixels) of Gaussian filter to use 
		smoothtruncate = how many sigma to truncate the Gaussian filter at on either side
		defaults (12.5 and 4.0) give a 100 pixel-wide filter
		"""
		if self.mean_image is None :
			raise FlatFieldError('ERROR: cannot call smoothMeanImage before calling takeMean!')
		self.smoothed_mean_image = skimage.filters.gaussian(self.mean_image,sigma=smoothsigma,truncate=smoothtruncate,mode='reflect')

	def makeFlatFieldImage(self) :
		if self.smoothed_mean_image is None :
			raise FlatFieldError('ERROR: cannot call makeFlatFieldImage before calling smoothMeanImage!')
		self.flatfield_image = np.ndarray(self.smoothed_mean_image.shape,dtype=np.float64)
		for layer_i in range(self.flatfield_image.shape[-1]) :
			layermean = np.mean(self.smoothed_mean_image[:,:,layer_i])
			self.flatfield_image[:,:,layer_i]=self.smoothed_mean_image[:,:,layer_i]/layermean

	def saveImages(self,namestem) :
		"""
		namestem = stem of filename to use when saving images
		"""
		if self.mean_image is not None :
			shape = self.mean_image.shape
			meanimage_filename = f'{namestem}_mean_of_{self.n_images_stacked}_{shape[0]}x{shape[1]}x{shape[2]}_images{FILE_EXT}'
			writeImageToFile(np.transpose(self.mean_image,(2,1,0)),meanimage_filename,dtype=np.float64)
		if self.smoothed_mean_image is not None :
			smoothed_meanimage_filename = f'{namestem}_smoothed_mean_image{FILE_EXT}'
			writeImageToFile(np.transpose(self.smoothed_mean_image,(2,1,0)),smoothed_meanimage_filename,dtype=np.float64)
		if self.flatfield_image is not None :
			flatfieldimage_filename = f'{namestem}{FILE_EXT}'
			writeImageToFile(np.transpose(self.flatfield_image,(2,1,0)),flatfieldimage_filename,dtype=np.float64)



