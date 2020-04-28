#imports
from ..utilities.img_file_io import writeImageToFile
from ..utilities.misc import cd
import numpy as np, matplotlib.pyplot as plt
import skimage.filters, skimage.util
import os

FILE_EXT='.bin'
VISUALIZATION_DIRECTORY_NAME='plots'
FIG_WIDTH=7.5 #width of figures created in inches

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

    #################### PUBLIC FUNCTIONS ####################

    def addNewImage(self,im_array) :
        """
        A function to add a single new image to the stack
        im_array = array of new image to add to the stack
        """
        self.image_stack+=im_array
        self.n_images_stacked+=1

    def makeFlatFieldImage(self,smoothsigma=12.5,smoothtruncate=4.0) :
        """
        A function to take the mean of the image stack, smooth it, and normalize each of its layers to make the flatfield image
        smoothsigma    = sigma (in pixels) of Gaussian filter to use 
        smoothtruncate = how many sigma to truncate the Gaussian filter at on either side
        defaults (12.5 and 4.0) give a 100 pixel-wide filter
        """
        self.__takeMean()
        self.__smoothMeanImage(smoothsigma,smoothtruncate)
        self.flatfield_image = np.ndarray(self.smoothed_mean_image.shape,dtype=np.float64)
        for layer_i in range(self.flatfield_image.shape[-1]) :
            layermean = np.mean(self.smoothed_mean_image[:,:,layer_i])
            self.flatfield_image[:,:,layer_i]=self.smoothed_mean_image[:,:,layer_i]/layermean

    def saveImages(self,namestem) :
        """
        Save mean image, smoothed mean image, and flatfield image all as float16s
        namestem = stem of filename to use when saving images
        """
        if self.mean_image is not None :
            shape = self.mean_image.shape
            meanimage_filename = f'{namestem}_mean_of_{self.n_images_stacked}_{shape[0]}x{shape[1]}x{shape[2]}_images{FILE_EXT}'
            writeImageToFile(np.transpose(self.mean_image.astype(np.float16,casting='same_kind'),(2,1,0)),meanimage_filename,dtype=np.float16)
        if self.smoothed_mean_image is not None :
            smoothed_meanimage_filename = f'{namestem}_smoothed_mean_image{FILE_EXT}'
            writeImageToFile(np.transpose(self.smoothed_mean_image.astype(np.float16,casting='same_kind'),(2,1,0)),smoothed_meanimage_filename,dtype=np.float16)
        if self.flatfield_image is not None :
            flatfieldimage_filename = f'{namestem}{FILE_EXT}'
            writeImageToFile(np.transpose(self.flatfield_image.astype(np.float16,casting='same_kind'),(2,1,0)),flatfieldimage_filename,dtype=np.float16)

    def saveVisualizations(self) :
        """
        Make and save several visualizations of the image layers
        """
        #figure out the size of the figures to save
        fig_size=(FIG_WIDTH,FIG_WIDTH*(self.mean_image.shape[0]/self.mean_image.shape[1]))
        #make the directory if its not already created
        if not os.path.isdir(VISUALIZATION_DIRECTORY_NAME) :
            os.mkdir(VISUALIZATION_DIRECTORY_NAME)
        with cd(VISUALIZATION_DIRECTORY_NAME) :
            #save a little plot of each layer in each image
            for layer_i in range(self.mean_image.shape[-1]) :
                layer_titlestem = f'layer {layer_i+1}'
                layer_fnstem = f'layer_{layer_i+1}'
                #for the mean image
                plt.figure(figsize=fig_size)
                plt.imshow(self.mean_image[:,:,layer_i])
                plt.title(f'mean image, {layer_titlestem}')
                plt.savefig(f'mean_image_{layer_fnstem}.png')
                plt.close()
                #for the smoothed mean image
                plt.figure(figsize=fig_size)
                plt.imshow(self.smoothed_mean_image[:,:,layer_i])
                plt.title(f'smoothed mean image, {layer_titlestem}')
                plt.savefig(f'smoothed_mean_image_{layer_fnstem}.png')
                plt.close()
                #for the flatfield image
                plt.figure(figsize=fig_size)
                plt.imshow(self.flatfield_image[:,:,layer_i])
                plt.title(f'flatfield, {layer_titlestem}')
                plt.savefig(f'flatfield_{layer_fnstem}.png')
                plt.close()

    #################### HELPER FUNCTIONS ####################

    def __takeMean(self) :
        self.mean_image = self.image_stack/self.n_images_stacked

    def __smoothMeanImage(self,smoothsigma=12.5,smoothtruncate=4.0) :
        if self.mean_image is None :
            raise FlatFieldError('ERROR: cannot call smoothMeanImage before calling takeMean!')
        self.smoothed_mean_image = skimage.filters.gaussian(self.mean_image,sigma=smoothsigma,truncate=smoothtruncate,mode='reflect')



