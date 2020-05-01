#imports
from .config import *
from ..utilities.img_file_io import writeImageToFile
from ..utilities.misc import cd
import numpy as np, matplotlib.pyplot as plt, multiprocessing as mp
import os, cv2

class FlatFieldError(Exception) :
    """
    Class for errors encountered during flatfielding
    """
    pass

class MeanImage :
    """
    Class to hold an image that is the mean of a bunch of stacked raw images 
    """
    def __init__(self,name,x,y,nlayers,dtype,max_nprocs,skip_masking=False,smoothkernel=(101,101),flux_threshold=35.) :
        """
        name           = stem to use for naming files that get created
        x              = x dimension of images (pixels)
        y              = y dimension of images (pixels)
        nlayers        = number of layers in images
        dtype          = datatype of image arrays that will be stacked
        max_nprocs     = max number of parallel processes to run when smoothing images
        smoothsigma    = sigma (in pixels) of Gaussian filter to use 
        smoothtruncate = how many sigma to truncate the Gaussian filter at on either side
        defaults (50 and 4.0) give a 100 pixel-wide filter
        """
        self.namestem = name
        self.xpix = x
        self.ypix = y
        self.nlayers = nlayers
        self.max_nprocs = max_nprocs
        self.skip_masking = skip_masking
        self.smoothkernel = smoothkernel
        self.flux_threshold = flux_threshold
        self.image_stack = np.zeros((y,x,nlayers),dtype=dtype)
        self.mask_stack  = np.zeros((y,x,nlayers),dtype=np.uint8) #the masks are always 8-bit unsigned ints 
        self.smoothed_image_stack = np.zeros(self.image_stack.shape,dtype=IMG_DTYPE_OUT)
        self.n_images_stacked = 0
        self.mean_image=None
        self.smoothed_mean_image=None
        self.flatfield_image=None

    #################### PUBLIC FUNCTIONS ####################

    def addGroupOfImages(self,im_array_list) :
        """
        A function to add a list of raw image arrays to the image stack
        If masking is requested this function is parallelized and run on the GPU
        """
        #if the images aren't meant to be masked then we can just add them up trivially
        if self.skip_masking :
            for i,im_array in enumerate(im_array_list,start=1) :
                flatfield_logger.info(f'  adding image {self.n_images_stacked+1} in the stack....')
                self.image_stack+=im_array
                self.n_images_stacked+=1
                return
        #otherwise produce the image masks, apply them to the raw images, and be sure to add them to the list in the same order as the images
        manager = mp.Manager()
        return_dict = manager.dict()
        procs = []
        for i,im_array in enumerate(im_array_list) :
            flatfield_logger.info(f'  masking and adding image {self.n_images_stacked+i} in the stack....')
            p = mp.Process(target=getImageMaskWorker, args=(im_array,self.flux_threshold,i,return_dict))
            procs.append(p)
            p.start()
        for proc in procs:
            proc.join()
        for i,im_array in enumerate(im_array_list) :
            thismask = return_dict[i]
            self.image_stack+=(im_array*thismask)
            self.mask_stack+=thismask
            self.n_images_stacked+=1

    def makeFlatFieldImage(self) :
        """
        A function to get the mean of the image stack and smooth/normalize each of its layers to make the flatfield image
        """
        self.mean_image = self.__makeMeanImage()
        return_list = []
        smoothImageLayerByLayerWorker(self.mean_image,self.smoothkernel,return_list)
        self.smoothed_mean_image = return_list[0]
        self.flatfield_image = np.empty_like(self.smoothed_mean_image)
        for layer_i in range(self.nlayers) :
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
            writeImageToFile(np.transpose(self.mean_image,(2,1,0)),meanimage_filename,dtype=IMG_DTYPE_OUT)
        if self.smoothed_mean_image is not None :
            smoothed_meanimage_filename = f'{namestem}_smoothed_mean_image{FILE_EXT}'
            writeImageToFile(np.transpose(self.smoothed_mean_image,(2,1,0)),smoothed_meanimage_filename,dtype=IMG_DTYPE_OUT)
        if self.flatfield_image is not None :
            flatfieldimage_filename = f'{namestem}{FILE_EXT}'
            writeImageToFile(np.transpose(self.flatfield_image,(2,1,0)),flatfieldimage_filename,dtype=IMG_DTYPE_OUT)
        #if masks were calculated, save the stack of them
        if not self.skip_masking :
            writeImageToFile(np.transpose(self.mask_stack,(2,1,0)),f'mask_stack{FILE_EXT}',dtype=np.uint8)

    def savePlots(self) :
        """
        Make and save several visualizations of the image layers
        """
        #figure out the size of the figures to save
        fig_size=(IMG_LAYER_FIG_WIDTH,IMG_LAYER_FIG_WIDTH*(self.mean_image.shape[0]/self.mean_image.shape[1]))
        #keep track of the flatfield images' minimum and maximum (and 5/95%ile) pixel intensities while the other plots are made
        ff_min_pixel_intensities=[]
        ff_low_pixel_intensities=[]
        ff_max_pixel_intensities=[]
        ff_high_pixel_intensities=[]
        #iterate over the layers
        for layer_i in range(self.nlayers) :
            layer_titlestem = f'layer {layer_i+1}'
            layer_fnstem = f'layer_{layer_i+1}'
            #save a little figure of each layer in each image
            #for the mean image
            plt.figure(figsize=fig_size)
            plt.imshow(self.mean_image[:,:,layer_i])
            plt.title(f'mean image, {layer_titlestem}')
            plt.savefig(f'mean_image_{layer_fnstem}.png')
            plt.close()
            ##for the smoothed mean image
            #plt.figure(figsize=fig_size)
            #plt.imshow(self.smoothed_mean_image[:,:,layer_i])
            #plt.title(f'smoothed mean image, {layer_titlestem}')
            #plt.savefig(f'smoothed_mean_image_{layer_fnstem}.png')
            #plt.close()
            #for the flatfield image
            plt.figure(figsize=fig_size)
            plt.imshow(self.flatfield_image[:,:,layer_i])
            plt.title(f'flatfield, {layer_titlestem}')
            plt.savefig(f'flatfield_{layer_fnstem}.png')
            plt.close()
            #for the mask stack (if applicable) 
            plt.figure(figsize=fig_size)
            plt.imshow(self.mask_stack[:,:,layer_i])
            plt.title(f'stacked binary image masks, {layer_titlestem}')
            plt.savefig(f'mask_stack_{layer_fnstem}.png')
            plt.close()
            #find the min, max, and 5/95%ile pixel intensities for this image layer
            sorted_ff_layer = np.sort((self.flatfield_image[:,:,layer_i]).flatten())
            ff_min_pixel_intensities.append(sorted_ff_layer[0])
            ff_low_pixel_intensities.append(sorted_ff_layer[int(0.05*len(sorted_ff_layer))])
            ff_max_pixel_intensities.append(sorted_ff_layer[-1])
            ff_high_pixel_intensities.append(sorted_ff_layer[int(0.95*len(sorted_ff_layer))])
        #plot the inensity plots together, with the broadband filter breaks
        xaxis_vals = list(range(1,self.nlayers+1))
        plt.figure(figsize=(INTENSITY_FIG_WIDTH,(9./16.)*INTENSITY_FIG_WIDTH))
        plt.plot([xaxis_vals[0],xaxis_vals[-1]],[1.0,1.0],color='mediumseagreen',linestyle='dashed',label='mean intensity')
        for i in range(len(LAST_FILTER_LAYERS)+1) :
            f_i = 0 if i==0 else LAST_FILTER_LAYERS[i-1]
            l_i = xaxis_vals[-1] if i==len(LAST_FILTER_LAYERS) else LAST_FILTER_LAYERS[i]
            if i==0 :
                plt.plot(xaxis_vals[f_i:l_i],ff_min_pixel_intensities[f_i:l_i],color='darkblue',marker='o',linewidth=2,label='minimum intensity')
                plt.plot(xaxis_vals[f_i:l_i],ff_low_pixel_intensities[f_i:l_i],color='royalblue',marker='o',linewidth=2,linestyle='dotted',label=r'5th %ile intensity')
                plt.plot(xaxis_vals[f_i:l_i],ff_max_pixel_intensities[f_i:l_i],color='darkred',marker='o',linewidth=2,label='maximum intensity')
                plt.plot(xaxis_vals[f_i:l_i],ff_high_pixel_intensities[f_i:l_i],color='lightcoral',marker='o',linewidth=2,linestyle='dotted',label=r'95th %ile intensity')
                plt.plot([l_i+0.5,l_i+0.5],[min(ff_min_pixel_intensities)-0.06,max(ff_max_pixel_intensities)+0.06],color='black',linewidth=2,linestyle='dotted',label='broadband filter changeover')
            else :
                plt.plot(xaxis_vals[f_i:l_i],ff_min_pixel_intensities[f_i:l_i],color='darkblue',marker='o',linewidth=2)
                plt.plot(xaxis_vals[f_i:l_i],ff_low_pixel_intensities[f_i:l_i],color='royalblue',marker='o',linewidth=2,linestyle='dotted')
                plt.plot(xaxis_vals[f_i:l_i],ff_max_pixel_intensities[f_i:l_i],color='darkred',marker='o',linewidth=2)
                plt.plot(xaxis_vals[f_i:l_i],ff_high_pixel_intensities[f_i:l_i],color='lightcoral',marker='o',linewidth=2,linestyle='dotted')
                if i!=len(LAST_FILTER_LAYERS) :
                    plt.plot([l_i+0.5,l_i+0.5],[min(ff_min_pixel_intensities)-0.06,max(ff_max_pixel_intensities)+0.06],color='black',linewidth=2,linestyle='dotted')
        plt.title(f'flatfield image layer normalized pixel intensities',fontsize=14)
        plt.xlabel('layer number',fontsize=14)
        plt.ylabel('pixel intensity',fontsize=14)
        plt.legend(loc='best')
        plt.savefig('pixel_intensity_plot.png')

        #################### PRIVATE HELPER FUNCTIONS ####################

        #helper function to create and return the mean image from the image (and mask, if applicable, stack)
        def __makeMeanImage(self) :
            #if the images haven't been masked then this is trivial
            if self.skip_masking :
                return self.image_stack/self.n_images_stacked
            #otherwise though we have to be a bit careful and take the mean value pixel-wise, 
            #being careful to fix any pixels that never got added to so there's no division by zero
            self.mask_stack[self.mask_stack==0] = 1
            return self.image_stack/self.mask_stack

#################### FILE-SCOPE HELPER FUNCTIONS ####################

#helper function to create a layered binary image mask for a given image array
#this can be run in parallel
def getImageMaskWorker(im_array,flux_threshold,i,return_dict) :
    #parameters for how to morph each layer mask
    ERODE1_KERNEL = np.ones((3,3),np.uint8)
    DILATE1_KERNEL = np.ones((7,7),np.uint8)
    ERODE2_KERNEL = np.ones((3,3),np.uint8)
    DILATE2_KERNEL = np.ones((5,5),np.uint8)
    ERODE1_ITERS = 1
    DILATE1_ITERS = 2
    ERODE2_ITERS = 9
    DILATE2_ITERS = 2
    #start up a new multilayer mask
    image_mask = np.zeros(im_array.shape,np.uint8)
    for li in range(im_array.shape[-1]) :
        #gently smooth each layer to remove some noise on the GPU
        layer_in_umat = cv2.UMat(im_array[:,:,li])
        layer_out_umat = cv2.UMat(np.zeros_like(im_array[:,:,li]))
        cv2.GaussianBlur(layer_in_umat,(5,5),0,layer_out_umat,0,cv2.BORDER_REFLECT)
        smoothed_greyscale_layer = (layer_out_umat.get()).astype('uint8')
        #threshold the layer at the given flux to produce its binary mask
        _,layermask = cv2.threshold(smoothed_greyscale_layer,flux_threshold,1,cv2.THRESH_BINARY)
        #erode and dilate the initial mask twice to clean it up
        layermask = cv2.erode(layermask,ERODE1_KERNEL,iterations=ERODE1_ITERS)
        layermask = cv2.dilate(layermask,DILATE1_KERNEL,iterations=DILATE1_ITERS)
        layermask = cv2.erode(layermask,ERODE2_KERNEL,iterations=ERODE2_ITERS)
        layermask = cv2.dilate(layermask,DILATE2_KERNEL,iterations=DILATE2_ITERS)
        #set it to the current layer in the return mask
        image_mask[:,:,li] = layermask
    #add the total mask to the dict
    return_dict[i] = image_mask

#helper function to smooth each layer of an image independently on the GPU
#this can be run in parallel
def smoothImageLayerByLayerWorker(im_array,smoothkernel,return_list) :
    smoothed_im_array = np.zeros_like(im_array)
    nlayers = im_array.shape[-1]
    for li in range(nlayers) :
        layer_in_umat = cv2.UMat(im_array[:,:,li])
        layer_out_umat = cv2.UMat(smoothed_im_array[:,:,li])
        cv2.GaussianBlur(layer_in_umat,smoothkernel,0,layer_out_umat,0,cv2.BORDER_REFLECT)
        smoothed_im_array[:,:,li]=layer_out_umat.get()
    return_list.append(smoothed_im_array)

