#imports
from .utilities import flatfield_logger, FlatFieldError, smoothImageWorker, getImageArrayLayerHistograms, findLayerThresholds
from .config import CONST 
from ..utilities.img_file_io import getRawAsHWL, writeImageToFile
from ..utilities.misc import cd
import numpy as np, matplotlib.pyplot as plt, multiprocessing as mp
import cv2, os, copy

class MeanImage :
    """
    Class representing an image that is the mean of a bunch of stacked raw images 
    """

    #################### PROPERTIES ####################
    @property
    def workingdir_name(self):
        return self._workingdir_name #name of the working directory where everything gets saved
    @property
    def dims(self):
        return self._dims
    
    
    #################### CLASS CONSTANTS ####################

    #outputted images
    MEAN_IMAGE_FILE_NAME_STEM                    = 'mean_image'                      #name of the outputted mean image file
    SMOOTHED_MEAN_IMAGE_FILE_NAME_STEM           = 'smoothed_mean_image'             #name of the outputted smoothed mean image file
    CORRECTED_MEAN_IMAGE_FILE_NAME_STEM          = 'corrected_mean_image'            #name of the outputted corrected mean image file
    MASK_STACK_FILE_NAME_STEM                    = 'mask_stack'                      #name of the outputted mask stack file
    APPLIED_FLATFIELD_TEXT_FILE_NAME             = 'applied_flatfield_file_path.txt' #name of the text file to write out the applied flatfield file path
    #postrun plot directory
    POSTRUN_PLOT_DIRECTORY_NAME = 'postrun_info' #name of directory to hold postrun plots (pixel intensity, image layers, etc.) and other info
    #image layer plots
    IMAGE_LAYER_PLOT_DIRECTORY_NAME = 'image_layer_pngs' #name of directory to hold image layer plots within postrun plot directory
    IMG_LAYER_FIG_WIDTH             = 6.4                #width of image layer figures created in inches
    #pixel intensity plots
    PIXEL_INTENSITY_PLOT_NAME = 'pixel_intensity_plot.png' #name of the pixel intensity plot
    LAST_FILTER_LAYERS_35     = [9,18,25,32]               #last image layers of each broadband filter for 35-layer images
    LAST_FILTER_LAYERS_43     = [9,11,17,20,29,36]         #last image layers of each broadband filter for 43-layer images (unconfirmed)
    INTENSITY_FIG_WIDTH       = 16.8                       #width of the intensity plot figure
    #illumination variation reduction
    ILLUMINATION_VARIATION_PLOT_WIDTH    = 9.6                                   #width of the illumination variation plot
    ILLUMINATION_VARIATION_PLOT_NAME     = 'ilumination_variation_by_layer.png'  #name of the illumination variation plot
    ILLUMINATION_VARIATION_CSV_FILE_NAME = 'illumination_variation_by_layer.csv' #name of the illumination variation .csv file
    #images stacked per layer
    N_IMAGES_STACKED_PER_LAYER_PLOT_NAME      = 'n_images_stacked_per_layer.png' #name of the images stacked per layer plot
    N_IMAGES_STACKED_PER_LAYER_TEXT_FILE_NAME = 'n_images_stacked_per_layer.txt' #name of the images stacked per layer text file
    #masking plots
    MASKING_PLOT_DIR_NAME = 'masking_plots' #name of the masking plot directory

    #################### PUBLIC FUNCTIONS ####################

    def __init__(self,y,x,nlayers,workingdir_name,skip_masking=False,smoothsigma=100) :
        """
        y               = image height in pixels
        x               = image width in pixels
        nlayers         = number of layers in images
        workingdir_name = name of the directory to save everything in
        skip_masking    = if True, image layers won't be masked before being added to the stack
        smoothsigma     = Gaussian sigma for final smoothing of stacked flatfield image
        """
        self._dims=(y,x,nlayers)
        self.nlayers = nlayers
        self.LAST_FILTER_LAYERS = None
        if self.nlayers==35 :
            self.LAST_FILTER_LAYERS=self.LAST_FILTER_LAYERS_35
        elif self.nlayers==43 :
            self.LAST_FILTER_LAYERS=self.LAST_FILTER_LAYERS_43
        else :
            raise FlatFieldError(f'ERROR: no defined list of broadband filter breaks for images with {self.nlayers} layers!')
        self._workingdir_name = workingdir_name
        self.skip_masking = skip_masking
        self.smoothsigma = smoothsigma
        self.image_stack = np.zeros((y,x,nlayers),dtype=np.uint32) #WARNING: may overflow if more than 65,535 images are stacked 
        self.mask_stack  = np.zeros((y,x,nlayers),dtype=np.uint16) #WARNING: may overflow if more than 65,535 masks are stacked 
        self.smoothed_image_stack = np.zeros(self.image_stack.shape,dtype=CONST.IMG_DTYPE_OUT)
        self.n_images_read = 0
        self.n_images_stacked_by_layer = np.zeros((nlayers),dtype=np.uint16) #WARNING: may overflow if more than 65,535 images are stacked 
        self.mean_image=None
        self.smoothed_mean_image=None
        self.flatfield_image=None
        self.corrected_mean_image=None
        self.smoothed_corrected_mean_image=None

    def addGroupOfImages(self,im_array_list,sample,min_selected_pixels,masking_plot_indices=[]) :
        """
        A function to add a list of raw image arrays to the image stack
        If masking is requested this function's subroutines are parallelized and also run on the GPU
        im_array_list        = list of image arrays to add
        sample               = sample object corresponding to this group of images
        min_selected_pixels  = fraction (0->1) of how many pixels must be selected as signal for an image to be stacked
        masking_plot_indices = list of image array list indices whose masking plots will be saved
        """
        #if the images aren't meant to be masked then we can just add them up trivially
        if self.skip_masking :
            for i,im_array in enumerate(im_array_list,start=1) :
                flatfield_logger.info(f'  adding image {self.n_images_read+1} to the stack....')
                self.image_stack+=im_array
                self.n_images_read+=1
                self.n_images_stacked_by_layer+=1
            return
        #otherwise produce the image masks, apply them to the raw images, and be sure to add them to the list in the same order as the images
        with cd(self._workingdir_name) :
            if not os.path.isdir(self.MASKING_PLOT_DIR_NAME) :
                os.mkdir(self.MASKING_PLOT_DIR_NAME)
        masking_plot_dirpath = os.path.join(self._workingdir_name,self.MASKING_PLOT_DIR_NAME)
        manager = mp.Manager()
        return_dict = manager.dict()
        procs = []
        for i,im_array in enumerate(im_array_list,) :
            stack_i = i+self.n_images_read+1
            flatfield_logger.info(f'  masking and adding image {stack_i} to the stack....')
            make_plots=i in masking_plot_indices
            p = mp.Process(target=getImageMaskWorker, 
                           args=(im_array,sample.background_thresholds_for_masking,sample.name,min_selected_pixels,
                                 make_plots,masking_plot_dirpath,
                                 stack_i,return_dict))
            procs.append(p)
            p.start()
        for proc in procs:
            proc.join()
        for stack_i,im_array in enumerate(im_array_list,start=self.n_images_read+1) :
            thismask = return_dict[stack_i]
            #check, layer-by-layer, that this mask would select at least the minimum amount of pixels to be added to the stack
            for li in range(self.nlayers) :
                thismasklayer = thismask[:,:,li]
                if 1.*np.sum(thismasklayer)/(self._dims[0]*self._dims[1])>=min_selected_pixels :
                    self.image_stack[:,:,li]+=(im_array[:,:,li]*thismasklayer)
                    self.mask_stack[:,:,li]+=thismasklayer
                    self.n_images_stacked_by_layer[li]+=1
            self.n_images_read+=1

    def makeFlatFieldImage(self) :
        """
        A function to get the mean of the image stack and smooth/normalize each of its layers to make the flatfield image
        """
        if self.n_images_read<1 :
            raise FlatFieldError('ERROR: not enough images were read to produce a meanimage!')
        for li,nlis in enumerate(self.n_images_stacked_by_layer,start=1) :
            if nlis<1 :
                flatfield_logger.warn(f'WARNING: {nlis} images were stacked in layer {li}; this layer of the meanimage/flatfield will be meaningless!')
        self.mean_image = self.__makeMeanImage()
        self.smoothed_mean_image = smoothImageWorker(self.mean_image,self.smoothsigma)
        self.flatfield_image = np.empty_like(self.smoothed_mean_image)
        for layer_i in range(self.nlayers) :
            layermean = np.mean(self.smoothed_mean_image[:,:,layer_i])
            if layermean==0 :
                self.flatfield_image[:,:,layer_i]=1.0
            else :
                self.flatfield_image[:,:,layer_i]=self.smoothed_mean_image[:,:,layer_i]/layermean

    def makeCorrectedMeanImage(self,flatfield_file_path) :
        """
        A function to get the mean of the image stack, smooth it, and divide it by the given flatfield to correct it
        """
        self.mean_image = self.__makeMeanImage()
        self.smoothed_mean_image = smoothImageWorker(self.mean_image,self.smoothsigma)
        flatfield_image=getRawAsHWL(flatfield_file_path,*(self._dims),dtype=CONST.IMG_DTYPE_OUT)
        self.corrected_mean_image=self.mean_image/flatfield_image
        self.smoothed_corrected_mean_image=smoothImageWorker(self.corrected_mean_image,self.smoothsigma)
        with cd(self._workingdir_name) :
            with open(self.APPLIED_FLATFIELD_TEXT_FILE_NAME,'w') as fp :
                fp.write(f'{flatfield_file_path}\n')

    def saveImages(self) :
        """
        Save the various images that are created
        """
        with cd(self._workingdir_name) :
            if self.mean_image is not None :
                meanimage_filename = f'{self.MEAN_IMAGE_FILE_NAME_STEM}{CONST.FILE_EXT}'
                writeImageToFile(np.transpose(self.mean_image,(2,1,0)),meanimage_filename,dtype=CONST.IMG_DTYPE_OUT)
            if self.smoothed_mean_image is not None :
                smoothed_meanimage_filename = f'{self.SMOOTHED_MEAN_IMAGE_FILE_NAME_STEM}{CONST.FILE_EXT}'
                writeImageToFile(np.transpose(self.smoothed_mean_image,(2,1,0)),smoothed_meanimage_filename,dtype=CONST.IMG_DTYPE_OUT)
            if self.flatfield_image is not None :
                flatfieldimage_filename = f'{CONST.FLATFIELD_FILE_NAME_STEM}{CONST.FILE_EXT}'
                writeImageToFile(np.transpose(self.flatfield_image,(2,1,0)),flatfieldimage_filename,dtype=CONST.IMG_DTYPE_OUT)
            if self.corrected_mean_image is not None :
                corrected_mean_image_filename = f'{self.CORRECTED_MEAN_IMAGE_FILE_NAME_STEM}{CONST.FILE_EXT}'
                writeImageToFile(np.transpose(self.corrected_mean_image,(2,1,0)),corrected_mean_image_filename,dtype=CONST.IMG_DTYPE_OUT)
            if self.smoothed_corrected_mean_image is not None :
                smoothed_corrected_mean_image_filename = f'{CONST.SMOOTHED_CORRECTED_MEAN_IMAGE_FILE_NAME_STEM}{CONST.FILE_EXT}'
                writeImageToFile(np.transpose(self.smoothed_corrected_mean_image,(2,1,0)),smoothed_corrected_mean_image_filename,dtype=CONST.IMG_DTYPE_OUT)
            #if masks were calculated, save the stack of them
            if (not self.skip_masking) and (self.mask_stack is not None) :
                writeImageToFile(np.transpose(self.mask_stack,(2,1,0)),f'{self.MASK_STACK_FILE_NAME_STEM}{CONST.FILE_EXT}',dtype=np.uint16)

    def savePlots(self) :
        """
        Make and save several visualizations of the image layers and details of the masking and flatfielding
        """
        with cd(self._workingdir_name) :
            #make the plot directory if its not already created
            if not os.path.isdir(self.POSTRUN_PLOT_DIRECTORY_NAME) :
                os.mkdir(self.POSTRUN_PLOT_DIRECTORY_NAME)
            with cd(self.POSTRUN_PLOT_DIRECTORY_NAME) :
                #.pngs of the mean image, flatfield, and mask stack layers
                self.__saveImageLayerPlots()
                #plots of the minimum and and maximum (and 5/95%ile) pixel intensities
                if self.flatfield_image is not None :
                    #for the flatfield image
                    self.__saveFlatFieldImagePixelIntensityPlot()   
                if self.smoothed_mean_image is not None and self.corrected_mean_image is not None :
                    #for the corrected mean image
                    self.__saveCorrectedMeanImagePixelIntensityPlots()
                #plot and write a text file of how many images were stacked per layer
                self.__plotAndWriteNImagesStackedPerLayer()

    #################### PRIVATE HELPER FUNCTIONS ####################

    #helper function to create and return the mean image from the image (and mask, if applicable, stack)
    def __makeMeanImage(self) :
        #if the images haven't been masked then this is trivial
        if self.skip_masking :
            return self.image_stack/self.n_images_read
        #otherwise though we have to be a bit careful and take the mean value pixel-wise, 
        #being careful to fix any pixels that never got added to so there's no division by zero
        zero_fixed_mask_stack = copy.deepcopy(self.mask_stack)
        zero_fixed_mask_stack[zero_fixed_mask_stack==0] = np.min(zero_fixed_mask_stack[zero_fixed_mask_stack!=0])
        return self.image_stack/zero_fixed_mask_stack

    #################### VISUALIZATION HELPER FUNCTIONS ####################

    #helper function to make and save the .png images of the mean image, flatfield, and mask stack layers
    def __saveImageLayerPlots(self) :
        if not os.path.isdir(self.IMAGE_LAYER_PLOT_DIRECTORY_NAME) :
            os.mkdir(self.IMAGE_LAYER_PLOT_DIRECTORY_NAME)
        with cd(self.IMAGE_LAYER_PLOT_DIRECTORY_NAME) :
            #figure out the size of the figures to save
            fig_size=(self.IMG_LAYER_FIG_WIDTH,self.IMG_LAYER_FIG_WIDTH*(self.mean_image.shape[0]/self.mean_image.shape[1]))
            #iterate over the layers
            for layer_i in range(self.nlayers) :
                layer_titlestem = f'layer {layer_i+1}'
                layer_fnstem = f'layer_{layer_i+1}'
                #save a little figure of each layer in each image
                if self.mean_image is not None :
                    #for the mean image
                    f,ax = plt.subplots(figsize=fig_size)
                    pos = ax.imshow(self.mean_image[:,:,layer_i])
                    ax.set_title(f'mean image, {layer_titlestem}')
                    f.colorbar(pos,ax=ax)
                    plt.savefig(f'mean_image_{layer_fnstem}.png')
                    plt.close()
                if self.smoothed_mean_image is not None :
                    #for the smoothed mean image
                    f,ax = plt.subplots(figsize=fig_size)
                    pos = ax.imshow(self.smoothed_mean_image[:,:,layer_i])
                    ax.set_title(f'smoothed mean image, {layer_titlestem}')
                    f.colorbar(pos,ax=ax)
                    plt.savefig(f'smoothed_mean_image_{layer_fnstem}.png')
                    plt.close()
                if self.flatfield_image is not None :
                    #for the flatfield image
                    f,ax = plt.subplots(figsize=fig_size)
                    pos = ax.imshow(self.flatfield_image[:,:,layer_i])
                    ax.set_title(f'flatfield, {layer_titlestem}')
                    f.colorbar(pos,ax=ax)
                    plt.savefig(f'flatfield_{layer_fnstem}.png')
                    plt.close()
                if self.corrected_mean_image is not None :
                    #for the corrected mean image
                    f,ax = plt.subplots(figsize=fig_size)
                    pos = ax.imshow(self.corrected_mean_image[:,:,layer_i])
                    ax.set_title(f'corrected mean image, {layer_titlestem}')
                    f.colorbar(pos,ax=ax)
                    plt.savefig(f'corrected_mean_image_{layer_fnstem}.png')
                    plt.close()
                if self.smoothed_corrected_mean_image is not None :
                    #for the corrected mean image
                    f,ax = plt.subplots(figsize=fig_size)
                    pos = ax.imshow(self.smoothed_corrected_mean_image[:,:,layer_i])
                    ax.set_title(f'smoothed corrected mean image, {layer_titlestem}')
                    f.colorbar(pos,ax=ax)
                    plt.savefig(f'smoothed_corrected_mean_image_{layer_fnstem}.png')
                    plt.close()
                if (not self.skip_masking) and (self.mask_stack is not None) :
                    #for the mask stack
                    f,ax = plt.subplots(figsize=fig_size)
                    pos = ax.imshow(self.mask_stack[:,:,layer_i])
                    ax.set_title(f'stacked binary image masks, {layer_titlestem}')
                    f.colorbar(pos,ax=ax)
                    plt.savefig(f'mask_stack_{layer_fnstem}.png')
                    plt.close()

    #helper function to plot the pixel intensities of the layers in the flatfield image
    def __saveFlatFieldImagePixelIntensityPlot(self) :
        #keep track of the flatfield images' minimum and maximum (and 5/95%ile) pixel intensities while the other plots are made
        ff_min_pixel_intensities=[]
        ff_low_pixel_intensities=[]
        ff_max_pixel_intensities=[]
        ff_high_pixel_intensities=[]
        plt.figure(figsize=(self.INTENSITY_FIG_WIDTH,(27./64.)*self.INTENSITY_FIG_WIDTH))
        xaxis_vals = list(range(1,self.nlayers+1))
        #iterate over the layers
        for layer_i in range(self.nlayers) :
            #find the min, max, and 5/95%ile pixel intensities for this image layer
            sorted_ff_layer = np.sort((self.flatfield_image[:,:,layer_i]).flatten())
            ff_min_pixel_intensities.append(sorted_ff_layer[0])
            ff_low_pixel_intensities.append(sorted_ff_layer[int(0.05*len(sorted_ff_layer))])
            stddev = np.std(sorted_ff_layer)
            ff_max_pixel_intensities.append(sorted_ff_layer[-1])
            ff_high_pixel_intensities.append(sorted_ff_layer[int(0.95*len(sorted_ff_layer))])
            if layer_i==0 :
                plt.fill([layer_i+0.5,layer_i+0.5,layer_i+1.5,layer_i+1.5],[1.-stddev,1.+stddev,1.+stddev,1.-stddev],'mediumseagreen',alpha=0.5,label='intensity std. dev.')
            else :
                plt.fill([layer_i+0.5,layer_i+0.5,layer_i+1.5,layer_i+1.5],[1.-stddev,1.+stddev,1.+stddev,1.-stddev],'mediumseagreen',alpha=0.5)
        #plot the relative intensity plots together, with the broadband filter breaks
        plt.plot([xaxis_vals[0],xaxis_vals[-1]],[1.0,1.0],color='mediumseagreen',linestyle='dashed',label='mean intensity')
        for i in range(len(self.LAST_FILTER_LAYERS)+1) :
            f_i = 0 if i==0 else self.LAST_FILTER_LAYERS[i-1]
            l_i = xaxis_vals[-1] if i==len(self.LAST_FILTER_LAYERS) else self.LAST_FILTER_LAYERS[i]
            if i==0 :
                plt.plot(xaxis_vals[f_i:l_i],ff_min_pixel_intensities[f_i:l_i],color='darkblue',marker='o',label='minimum intensity')
                plt.plot(xaxis_vals[f_i:l_i],ff_low_pixel_intensities[f_i:l_i],color='royalblue',marker='o',linestyle='dotted',label=r'5th %ile intensity')
                plt.plot(xaxis_vals[f_i:l_i],ff_max_pixel_intensities[f_i:l_i],color='darkred',marker='o',label='maximum intensity')
                plt.plot(xaxis_vals[f_i:l_i],ff_high_pixel_intensities[f_i:l_i],color='lightcoral',marker='o',linestyle='dotted',label=r'95th %ile intensity')
                plt.plot([l_i+0.5,l_i+0.5],[min(ff_min_pixel_intensities)-0.01,max(ff_max_pixel_intensities)+0.01],color='black',linewidth=2,linestyle='dotted',label='broadband filter changeover')
            else :
                plt.plot(xaxis_vals[f_i:l_i],ff_min_pixel_intensities[f_i:l_i],color='darkblue',marker='o')
                plt.plot(xaxis_vals[f_i:l_i],ff_low_pixel_intensities[f_i:l_i],color='royalblue',marker='o',linestyle='dotted')
                plt.plot(xaxis_vals[f_i:l_i],ff_max_pixel_intensities[f_i:l_i],color='darkred',marker='o')
                plt.plot(xaxis_vals[f_i:l_i],ff_high_pixel_intensities[f_i:l_i],color='lightcoral',marker='o',linestyle='dotted')
                if i!=len(self.LAST_FILTER_LAYERS) :
                    plt.plot([l_i+0.5,l_i+0.5],[min(ff_min_pixel_intensities)-0.01,max(ff_max_pixel_intensities)+0.01],color='black',linewidth=2,linestyle='dotted')
        plt.title('flatfield image layer normalized pixel intensities',fontsize=14)
        plt.xlabel('layer number',fontsize=14)
        #fix the range on the x-axis to accommodate the legend
        plt.xlim(0,self.nlayers+10)
        plt.ylabel('pixel intensity',fontsize=14)
        plt.legend(loc='lower right')
        plt.savefig(self.PIXEL_INTENSITY_PLOT_NAME)
        plt.close()

    #helper function to plot how the average intensities of the mean image change in each layer after application of the flatfield corrections
    def __saveCorrectedMeanImagePixelIntensityPlots(self) :
        #keep track of the uncorrected and corrected images' minimum and maximum (and 5/95%ile) pixel intensities while the other plots are made
        u_min_pixel_intensities=[]; u_low_pixel_intensities=[]; u_max_pixel_intensities=[]; u_high_pixel_intensities=[]
        c_min_pixel_intensities=[]; c_low_pixel_intensities=[]; c_max_pixel_intensities=[]; c_high_pixel_intensities=[]
        u_std_devs=[]; c_std_devs=[]
        plt.figure(figsize=(self.INTENSITY_FIG_WIDTH,(27./64.)*self.INTENSITY_FIG_WIDTH))
        xaxis_vals = list(range(1,self.nlayers+1))
        #iterate over the layers
        for layer_i in range(self.nlayers) :
            #find the min, max, and 5/95%ile pixel intensities for this uncorrected and corrected image layer
            sorted_u_layer = np.sort((self.smoothed_mean_image[:,:,layer_i]).flatten())/np.mean(self.smoothed_mean_image[:,:,layer_i])
            u_min_pixel_intensities.append(sorted_u_layer[0]); u_low_pixel_intensities.append(sorted_u_layer[int(0.05*len(sorted_u_layer))])
            u_stddev = np.std(sorted_u_layer); u_std_devs.append(u_stddev)
            u_max_pixel_intensities.append(sorted_u_layer[-1]); u_high_pixel_intensities.append(sorted_u_layer[int(0.95*len(sorted_u_layer))])
            sorted_c_layer = np.sort((self.smoothed_corrected_mean_image[:,:,layer_i]).flatten())/np.mean(self.smoothed_corrected_mean_image[:,:,layer_i])
            c_min_pixel_intensities.append(sorted_c_layer[0]); c_low_pixel_intensities.append(sorted_c_layer[int(0.05*len(sorted_c_layer))])
            c_stddev = np.std(sorted_c_layer); c_std_devs.append(c_stddev)
            c_max_pixel_intensities.append(sorted_c_layer[-1]); c_high_pixel_intensities.append(sorted_c_layer[int(0.95*len(sorted_c_layer))])
            if layer_i==0 :
                plt.fill([layer_i+0.5,layer_i+0.5,layer_i+1.5,layer_i+1.5],[1.-u_stddev,1.+u_stddev,1.+u_stddev,1.-u_stddev],'mediumseagreen',alpha=0.5,label='uncorrected std. dev.')
                plt.fill([layer_i+0.5,layer_i+0.5,layer_i+1.5,layer_i+1.5],[1.-c_stddev,1.+c_stddev,1.+c_stddev,1.-c_stddev],'goldenrod',alpha=0.5,label='corrected std. dev.')
            else :
                plt.fill([layer_i+0.5,layer_i+0.5,layer_i+1.5,layer_i+1.5],[1.-u_stddev,1.+u_stddev,1.+u_stddev,1.-u_stddev],'mediumseagreen',alpha=0.5)
                plt.fill([layer_i+0.5,layer_i+0.5,layer_i+1.5,layer_i+1.5],[1.-c_stddev,1.+c_stddev,1.+c_stddev,1.-c_stddev],'goldenrod',alpha=0.5)
        #plot the relative intensity plots together, with the broadband filter breaks
        plt.plot([xaxis_vals[0],xaxis_vals[-1]],[1.0,1.0],color='darkgreen',linestyle='dashed',label='mean')
        totalmin=min(min(u_min_pixel_intensities),min(c_min_pixel_intensities))
        totalmax=max(max(u_max_pixel_intensities),max(c_max_pixel_intensities))
        for i in range(len(self.LAST_FILTER_LAYERS)+1) :
            f_i = 0 if i==0 else self.LAST_FILTER_LAYERS[i-1]
            l_i = xaxis_vals[-1] if i==len(self.LAST_FILTER_LAYERS) else self.LAST_FILTER_LAYERS[i]
            if i==0 :
                plt.plot(xaxis_vals[f_i:l_i],u_min_pixel_intensities[f_i:l_i],color='darkred',marker='v',label='uncorrected min.')
                plt.plot(xaxis_vals[f_i:l_i],u_low_pixel_intensities[f_i:l_i],color='lightcoral',marker='v',linestyle='dotted',label=r'uncorrected 5th %ile')
                plt.plot(xaxis_vals[f_i:l_i],u_max_pixel_intensities[f_i:l_i],color='darkred',marker='^',label='uncorrected max.')
                plt.plot(xaxis_vals[f_i:l_i],u_high_pixel_intensities[f_i:l_i],color='lightcoral',marker='^',linestyle='dotted',label=r'uncorrected 95th %ile')
                plt.plot(xaxis_vals[f_i:l_i],c_min_pixel_intensities[f_i:l_i],color='darkblue',marker='v',label='corrected min.')
                plt.plot(xaxis_vals[f_i:l_i],c_low_pixel_intensities[f_i:l_i],color='royalblue',marker='v',linestyle='dotted',label=r'corrected 5th %ile')
                plt.plot(xaxis_vals[f_i:l_i],c_max_pixel_intensities[f_i:l_i],color='darkblue',marker='^',label='corrected max.')
                plt.plot(xaxis_vals[f_i:l_i],c_high_pixel_intensities[f_i:l_i],color='royalblue',marker='^',linestyle='dotted',label=r'corrected 95th %ile')
                plt.plot([l_i+0.5,l_i+0.5],[totalmin-0.1,totalmax+0.1],color='black',linewidth=2,linestyle='dotted',label='broadband filter changeover')
            else :
                plt.plot(xaxis_vals[f_i:l_i],u_min_pixel_intensities[f_i:l_i],color='darkred',marker='v')
                plt.plot(xaxis_vals[f_i:l_i],u_low_pixel_intensities[f_i:l_i],color='lightcoral',marker='v',linestyle='dotted')
                plt.plot(xaxis_vals[f_i:l_i],u_max_pixel_intensities[f_i:l_i],color='darkred',marker='^')
                plt.plot(xaxis_vals[f_i:l_i],u_high_pixel_intensities[f_i:l_i],color='lightcoral',marker='^',linestyle='dotted')
                plt.plot(xaxis_vals[f_i:l_i],c_min_pixel_intensities[f_i:l_i],color='darkblue',marker='v')
                plt.plot(xaxis_vals[f_i:l_i],c_low_pixel_intensities[f_i:l_i],color='royalblue',marker='v',linestyle='dotted')
                plt.plot(xaxis_vals[f_i:l_i],c_max_pixel_intensities[f_i:l_i],color='darkblue',marker='^')
                plt.plot(xaxis_vals[f_i:l_i],c_high_pixel_intensities[f_i:l_i],color='royalblue',marker='^',linestyle='dotted')
                if i!=len(self.LAST_FILTER_LAYERS) :
                    plt.plot([l_i+0.5,l_i+0.5],[totalmin-0.1,totalmax+0.1],color='black',linewidth=2,linestyle='dotted')
        plt.title('uncorrected/corrected smoothed mean image relative pixel intensities',fontsize=14)
        plt.xlabel('layer number',fontsize=14)
        #fix the range on the x-axis to accommodate the legend
        plt.xlim(0,self.nlayers+10)
        plt.ylabel('pixel intensity relative to layer mean',fontsize=14)
        plt.legend(loc='lower right')
        plt.savefig(self.PIXEL_INTENSITY_PLOT_NAME)
        plt.close()
        #plot the reduction in illumination variation and save the values to a csv file as well
        u_hi_lo_spread = [u_high_pixel_intensities[li]-u_low_pixel_intensities[li] for li in range(self.nlayers)]
        c_hi_lo_spread = [c_high_pixel_intensities[li]-c_low_pixel_intensities[li] for li in range(self.nlayers)]
        f,ax=plt.subplots(figsize=(self.ILLUMINATION_VARIATION_PLOT_WIDTH,0.5*self.ILLUMINATION_VARIATION_PLOT_WIDTH))
        ax.plot(xaxis_vals,u_hi_lo_spread,marker='v',linestyle='dashed',linewidth=2,label='uncorrected 5th-95th %ile')
        ax.plot(xaxis_vals,c_hi_lo_spread,marker='^',linestyle='dashed',linewidth=2,label='corrected 5th-95th %ile')
        ax.plot(xaxis_vals,u_std_devs,marker='<',linestyle='dotted',linewidth=2,label='uncorrected std. dev.')
        ax.plot(xaxis_vals,c_std_devs,marker='>',linestyle='dotted',linewidth=2,label='corrected std. dev.')
        ax.set_ylim(0,(ax.get_ylim())[-1]*1.25)
        ax.set_title('illumination variation reduction by layer')
        ax.set_xlabel('image layer')
        ax.set_ylabel('relative flux')
        ax.legend(loc='best')
        plt.savefig(self.ILLUMINATION_VARIATION_PLOT_NAME)
        plt.close()
        with open(self.ILLUMINATION_VARIATION_CSV_FILE_NAME,'w') as ivrcfp :
            ivrcfp.write('layer,uncorrected_5th_to_95th_percentile,corrected_5th_to_95th_percentile,uncorrected_std_dev,corrected_std_dev\n')
            for li,(uhls,chls,usd,csd) in enumerate(zip(u_hi_lo_spread,c_hi_lo_spread,u_std_devs,c_std_devs),start=1) :
                ivrcfp.write(f'{li:d},{uhls:.5f},{chls:.5f},{usd:.5f},{csd:.5f}\n')


    #helper function to plot and save how many images ended up being stacked in each layer of the meanimage
    def __plotAndWriteNImagesStackedPerLayer(self) :
        xvals = list(range(1,self.nlayers+1))
        f,ax = plt.subplots(figsize=(9.6,4.8))
        ax.plot([xvals[0],xvals[-1]],[self.n_images_read,self.n_images_read],linewidth=2,color='k',label=f'total images read ({self.n_images_read})')
        ax.plot(xvals,self.n_images_stacked_by_layer,marker='o',linewidth=2,label='n images stacked')
        for pi,(xv,yv) in enumerate(zip(xvals,self.n_images_stacked_by_layer)) :
            plt.annotate(f'{yv:d}',
                         (xv,yv),
                         textcoords="offset points", # how to position the text
                         xytext=(0,10) if pi%2==0 else (0,-12), # distance from text to points (x,y)
                         ha='center') # horizontal alignment can be left, right or center 
        ax.set_title('Number of images selected to be stacked by layer')
        ax.set_xlabel('image layer')
        ax.set_ylabel('number of images')
        ax.legend(loc='best')
        plt.savefig(self.N_IMAGES_STACKED_PER_LAYER_PLOT_NAME)
        plt.close()
        with open(self.N_IMAGES_STACKED_PER_LAYER_TEXT_FILE_NAME,'w') as fp :
            for li in range(self.nlayers) :
                fp.write(f'{self.n_images_stacked_by_layer[li]}\n')

#################### FILE-SCOPE HELPER FUNCTIONS ####################

#helper function to create a layered binary image mask for a given image array
#this can be run in parallel with a given index and return dict
def getImageMaskWorker(im_array,thresholds_per_layer,samp_name,min_selected_pixels,make_plots=False,plotdir_path=None,i=None,return_dict=None) :
    nlayers = im_array.shape[-1]
    #create a new mask
    init_image_mask = np.empty(im_array.shape,np.uint8)
    #create a list to hold the threshold values
    thresholds = thresholds_per_layer
    #gently smooth the layer (on the GPU) to remove some noise
    smoothed_image = smoothImageWorker(im_array,CONST.GENTLE_GAUSSIAN_SMOOTHING_SIGMA)
    #if the thresholds haven't already been determined from the background, find them for all the layers by repeated Otsu thresholding
    if thresholds is None :
        layer_hists=getImageArrayLayerHistograms(smoothed_image)
        thresholds=findLayerThresholds(layer_hists)
    #for each layer, save the mask and its threshold
    for li in range(nlayers) :
        init_image_mask[:,:,li] = np.where(smoothed_image[:,:,li]>thresholds[li],1,0)
    #morph each layer of the mask through a series of operations on the GPU
    init_mask_umat  = cv2.UMat(init_image_mask)
    intermediate_mask=cv2.UMat(np.empty_like(init_image_mask))
    co1_mask=cv2.UMat(np.empty_like(init_image_mask))
    co2_mask=cv2.UMat(np.empty_like(init_image_mask))
    close3_mask=cv2.UMat(np.empty_like(init_image_mask))
    open3_mask=cv2.UMat(np.empty_like(init_image_mask))
    #do the morphology transformations
    #small-scale close/open to remove noise and fill in small holes
    cv2.morphologyEx(init_mask_umat,cv2.MORPH_CLOSE,CONST.CO1_EL,intermediate_mask,borderType=cv2.BORDER_REPLICATE)
    cv2.morphologyEx(intermediate_mask,cv2.MORPH_OPEN,CONST.CO1_EL,co1_mask,borderType=cv2.BORDER_REPLICATE)
    #medium-scale close/open for the same reason with larger regions
    cv2.morphologyEx(co1_mask,cv2.MORPH_CLOSE,CONST.CO2_EL,intermediate_mask,borderType=cv2.BORDER_REPLICATE)
    cv2.morphologyEx(intermediate_mask,cv2.MORPH_OPEN,CONST.CO2_EL,co2_mask,borderType=cv2.BORDER_REPLICATE)
    #large close to define the bulk of the mask
    cv2.morphologyEx(co2_mask,cv2.MORPH_CLOSE,CONST.C3_EL,close3_mask,borderType=cv2.BORDER_REPLICATE)
    #repeated small open to eat its edges and remove small regions outside of the bulk of the mask
    cv2.morphologyEx(close3_mask,cv2.MORPH_OPEN,CONST.CO1_EL,open3_mask,iterations=CONST.OPEN_3_ITERATIONS,borderType=cv2.BORDER_REPLICATE)
    #the final mask is this last mask
    morphed_mask = open3_mask.get()
    #make the plots if requested
    if make_plots :
        flatfield_logger.info(f'Saving masking plots for image {i}')
        this_image_masking_plot_dirname = f'image_{i}_from_{samp_name}_mask_layers'
        with cd(plotdir_path) :
            if not os.path.isdir(this_image_masking_plot_dirname) :
                os.mkdir(this_image_masking_plot_dirname)
        plotdir_path = os.path.join(plotdir_path,this_image_masking_plot_dirname)
        with cd(plotdir_path) :
            co1_mask = co1_mask.get()
            co2_mask = co2_mask.get()
            close3_mask = close3_mask.get()
            for li in range(nlayers) :
                f,ax = plt.subplots(4,2,figsize=CONST.MASKING_PLOT_FIG_SIZE)
                im = im_array[:,:,li]
                im_grayscale = im/np.max(im)
                im = (np.clip(im,0,255)).astype('uint8')
                ax[0][0].imshow(im_grayscale,cmap='gray')
                ax[0][0].set_title('raw image in grayscale',fontsize=14)
                ax[0][1].imshow(init_image_mask[:,:,li])
                ax[0][1].set_title(f'init mask (thresh. = {thresholds[li]:.1f})',fontsize=14)
                ax[1][0].imshow(co1_mask[:,:,li])
                ax[1][0].set_title('mask after small-scale open+close',fontsize=14)
                ax[1][1].imshow(co2_mask[:,:,li])
                ax[1][1].set_title('mask after medium-scale open+close',fontsize=14)
                ax[2][0].imshow(close3_mask[:,:,li])
                ax[2][0].set_title('mask after large-scale close',fontsize=14)
                m = morphed_mask[:,:,li]
                pixelfrac = 1.*np.sum(m)/(im_array.shape[0]*im_array.shape[1])
                will_be_stacked_text = 'WILL' if pixelfrac>=min_selected_pixels else 'WILL_NOT'
                ax[2][1].imshow(m)
                ax[2][1].set_title('final mask after repeated small-scale opening',fontsize=14)
                overlay_clipped = np.array([im,im*m,im*m]).transpose(1,2,0)
                overlay_grayscale = np.array([im_grayscale*m,im_grayscale*m,0.15*m]).transpose(1,2,0)
                ax[3][0].imshow(overlay_clipped)
                ax[3][0].set_title(f'mask overlaid with clipped image; selected fraction={pixelfrac:.3f}')
                ax[3][1].imshow(overlay_grayscale)
                ax[3][1].set_title(f'mask overlaid with grayscale image; image {will_be_stacked_text} be stacked')
                figname = f'image_{i}_layer_{li+1}_masks.png'
                plt.savefig(figname)
                plt.close()
    if i is not None and return_dict is not None :
        #add the total mask to the dict, along with its initial thresholds and number of optimal Otsu iterations per layer
        return_dict[i] = morphed_mask
    else :
        return morphed_mask
