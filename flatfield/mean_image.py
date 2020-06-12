#imports
from .config import *
from .utilities import smoothImageWorker
from ..utilities.img_file_io import getRawAsHWL, writeImageToFile
import matplotlib.pyplot as plt, multiprocessing as mp
import statistics, copy

class MeanImage :
    """
    Class to hold an image that is the mean of a bunch of stacked raw images 
    """
    def __init__(self,y,x,nlayers,workingdir_name,skip_masking=False,smoothsigma=100) :
        """
        y               = image height in pixels
        x               = image width in pixels
        nlayers         = number of layers in images
        workingdir_name = name of the directory to save everything in
        skip_masking    = if True, image layers won't be masked before being added to the stack
        smoothsigma     = Gaussian sigma for final smoothing of stacked flatfield image
        """
        self.dims=(y,x,nlayers)
        self.nlayers = nlayers
        self.workingdir_name = workingdir_name
        self.skip_masking = skip_masking
        self.smoothsigma = smoothsigma
        self.image_stack = np.zeros((y,x,nlayers),dtype=np.uint32) #WARNING: may overflow if more than 65,535 images are stacked 
        self.mask_stack  = np.zeros((y,x,nlayers),dtype=np.uint16) #WARNING: may overflow if more than 65,535 masks are stacked 
        self.smoothed_image_stack = np.zeros(self.image_stack.shape,dtype=IMG_DTYPE_OUT)
        self.n_images_read = 0
        self.n_images_stacked_by_layer = np.zeros((nlayers),dtype=np.uint16) #WARNING: may overflow if more than 65,535 images are stacked 
        self.mean_image=None
        self.smoothed_mean_image=None
        self.flatfield_image=None
        self.corrected_mean_image=None
        self.smoothed_corrected_mean_image=None

    #################### PUBLIC FUNCTIONS ####################

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
        manager = mp.Manager()
        return_dict = manager.dict()
        procs = []
        for i,im_array in enumerate(im_array_list,) :
            stack_i = i+self.n_images_read+1
            flatfield_logger.info(f'  masking and adding image {stack_i} to the stack....')
            make_plots=i in masking_plot_indices
            p = mp.Process(target=getImageMaskWorker, 
                           args=(im_array,sample.background_thresholds_for_masking,stack_i,min_selected_pixels,make_plots,self.workingdir_name,return_dict))
            procs.append(p)
            p.start()
        for proc in procs:
            proc.join()
        for stack_i,im_array in enumerate(im_array_list,start=self.n_images_read+1) :
            thismask = return_dict[stack_i]
            #check, layer-by-layer, that this mask would select at least the minimum amount of pixels to be added to the stack
            for li in range(self.nlayers) :
                thismasklayer = thismask[:,:,li]
                if 1.*np.sum(thismasklayer)/(self.dims[0]*self.dims[1])>=min_selected_pixels :
                    self.image_stack[:,:,li]+=(im_array[:,:,li]*thismasklayer)
                    self.mask_stack[:,:,li]+=thismasklayer
                    self.n_images_stacked_by_layer[li]+=1
            self.n_images_read+=1

    def makeFlatFieldImage(self) :
        """
        A function to get the mean of the image stack and smooth/normalize each of its layers to make the flatfield image
        """
        self.mean_image = self.__makeMeanImage()
        self.smoothed_mean_image = smoothImageWorker(self.mean_image,self.smoothsigma)
        self.flatfield_image = np.empty_like(self.smoothed_mean_image)
        for layer_i in range(self.nlayers) :
            layermean = np.mean(self.smoothed_mean_image[:,:,layer_i])
            self.flatfield_image[:,:,layer_i]=self.smoothed_mean_image[:,:,layer_i]/layermean

    def makeCorrectedMeanImage(self,flatfield_file_path) :
        """
        A function to get the mean of the image stack, smooth it, and divide it by the given flatfield to correct it
        """
        self.mean_image = self.__makeMeanImage()
        self.smoothed_mean_image = smoothImageWorker(self.mean_image,self.smoothsigma)
        flatfield_image=getRawAsHWL(flatfield_file_path,*(self.dims),dtype=IMG_DTYPE_OUT)
        self.corrected_mean_image=self.mean_image/flatfield_image
        self.smoothed_corrected_mean_image=smoothImageWorker(self.corrected_mean_image,self.smoothsigma)
        with cd(self.workingdir_name) :
            with open(APPLIED_FLATFIELD_TEXT_FILE_NAME,'w') as fp :
                fp.write(f'{flatfield_file_path}\n')

    def saveImages(self) :
        """
        Save the various images that are created
        """
        with cd(self.workingdir_name) :
            if self.mean_image is not None :
                meanimage_filename = f'{MEAN_IMAGE_FILE_NAME_STEM}{FILE_EXT}'
                writeImageToFile(np.transpose(self.mean_image,(2,1,0)),meanimage_filename,dtype=IMG_DTYPE_OUT)
            if self.smoothed_mean_image is not None :
                smoothed_meanimage_filename = f'{SMOOTHED_MEAN_IMAGE_FILE_NAME_STEM}{FILE_EXT}'
                writeImageToFile(np.transpose(self.smoothed_mean_image,(2,1,0)),smoothed_meanimage_filename,dtype=IMG_DTYPE_OUT)
            if self.flatfield_image is not None :
                flatfieldimage_filename = f'{FLATFIELD_FILE_NAME_STEM}{FILE_EXT}'
                writeImageToFile(np.transpose(self.flatfield_image,(2,1,0)),flatfieldimage_filename,dtype=IMG_DTYPE_OUT)
            if self.corrected_mean_image is not None :
                corrected_mean_image_filename = f'{CORRECTED_MEAN_IMAGE_FILE_NAME_STEM}{FILE_EXT}'
                writeImageToFile(np.transpose(self.corrected_mean_image,(2,1,0)),corrected_mean_image_filename,dtype=IMG_DTYPE_OUT)
            if self.smoothed_corrected_mean_image is not None :
                smoothed_corrected_mean_image_filename = f'{SMOOTHED_CORRECTED_MEAN_IMAGE_FILE_NAME_STEM}{FILE_EXT}'
                writeImageToFile(np.transpose(self.smoothed_corrected_mean_image,(2,1,0)),smoothed_corrected_mean_image_filename,dtype=IMG_DTYPE_OUT)
            #if masks were calculated, save the stack of them
            if (not self.skip_masking) and (self.mask_stack is not None) :
                writeImageToFile(np.transpose(self.mask_stack,(2,1,0)),f'{MASK_STACK_FILE_NAME_STEM}{FILE_EXT}',dtype=np.uint16)

    def savePlots(self) :
        """
        Make and save several visualizations of the image layers and details of the masking and flatfielding
        """
        with cd(self.workingdir_name) :
            #make the plot directory if its not already created
            if not os.path.isdir(POSTRUN_PLOT_DIRECTORY_NAME) :
                os.mkdir(POSTRUN_PLOT_DIRECTORY_NAME)
            with cd(POSTRUN_PLOT_DIRECTORY_NAME) :
                #.pngs of the mean image, flatfield, and mask stack layers
                self.__saveImageLayerPlots()
                #plots of the minimum and and maximum (and 5/95%ile) pixel intensities
                if self.flatfield_image is not None :
                    #for the flatfield image
                    self.__saveFlatFieldImagePixelIntensityPlot()   
                if self.smoothed_mean_image is not None and self.corrected_mean_image is not None :
                    #for the corrected mean image
                    self.__saveCorrectedMeanImagePixelIntensityPlot()
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
        if not os.path.isdir(IMAGE_LAYER_PLOT_DIRECTORY_NAME) :
            os.mkdir(IMAGE_LAYER_PLOT_DIRECTORY_NAME)
        with cd(IMAGE_LAYER_PLOT_DIRECTORY_NAME) :
            #figure out the size of the figures to save
            fig_size=(IMG_LAYER_FIG_WIDTH,IMG_LAYER_FIG_WIDTH*(self.mean_image.shape[0]/self.mean_image.shape[1]))
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
        plt.figure(figsize=INTENSITY_FIG_SIZE)
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
        for i in range(len(LAST_FILTER_LAYERS)+1) :
            f_i = 0 if i==0 else LAST_FILTER_LAYERS[i-1]
            l_i = xaxis_vals[-1] if i==len(LAST_FILTER_LAYERS) else LAST_FILTER_LAYERS[i]
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
                if i!=len(LAST_FILTER_LAYERS) :
                    plt.plot([l_i+0.5,l_i+0.5],[min(ff_min_pixel_intensities)-0.01,max(ff_max_pixel_intensities)+0.01],color='black',linewidth=2,linestyle='dotted')
        plt.title(f'flatfield image layer normalized pixel intensities',fontsize=14)
        plt.xlabel('layer number',fontsize=14)
        #fix the range on the x-axis to accommodate the legend
        plt.xlim(0,self.nlayers+10)
        plt.ylabel('pixel intensity',fontsize=14)
        plt.legend(loc='best')
        plt.savefig(PIXEL_INTENSITY_PLOT_NAME)
        plt.close()

    #helper function to plot how the average intensities of the mean image change in each layer after application of the flatfield corrections
    def __saveCorrectedMeanImagePixelIntensityPlot(self) :
        #keep track of the uncorrected and corrected images' minimum and maximum (and 5/95%ile) pixel intensities while the other plots are made
        u_min_pixel_intensities=[]; u_low_pixel_intensities=[]; u_max_pixel_intensities=[]; u_high_pixel_intensities=[]
        c_min_pixel_intensities=[]; c_low_pixel_intensities=[]; c_max_pixel_intensities=[]; c_high_pixel_intensities=[]
        plt.figure(figsize=INTENSITY_FIG_SIZE)
        xaxis_vals = list(range(1,self.nlayers+1))
        #iterate over the layers
        for layer_i in range(self.nlayers) :
            #find the min, max, and 5/95%ile pixel intensities for this uncorrected and corrected image layer
            sorted_u_layer = np.sort((self.smoothed_mean_image[:,:,layer_i]).flatten())/np.mean(self.smoothed_mean_image[:,:,layer_i])
            u_min_pixel_intensities.append(sorted_u_layer[0]); u_low_pixel_intensities.append(sorted_u_layer[int(0.05*len(sorted_u_layer))])
            u_stddev = np.std(sorted_u_layer)
            u_max_pixel_intensities.append(sorted_u_layer[-1]); u_high_pixel_intensities.append(sorted_u_layer[int(0.95*len(sorted_u_layer))])
            sorted_c_layer = np.sort((self.smoothed_corrected_mean_image[:,:,layer_i]).flatten())/np.mean(self.smoothed_corrected_mean_image[:,:,layer_i])
            c_min_pixel_intensities.append(sorted_c_layer[0]); c_low_pixel_intensities.append(sorted_c_layer[int(0.05*len(sorted_c_layer))])
            c_stddev = np.std(sorted_c_layer)
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
        for i in range(len(LAST_FILTER_LAYERS)+1) :
            f_i = 0 if i==0 else LAST_FILTER_LAYERS[i-1]
            l_i = xaxis_vals[-1] if i==len(LAST_FILTER_LAYERS) else LAST_FILTER_LAYERS[i]
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
                if i!=len(LAST_FILTER_LAYERS) :
                    plt.plot([l_i+0.5,l_i+0.5],[totalmin-0.1,totalmax+0.1],color='black',linewidth=2,linestyle='dotted')
        plt.title(f'uncorrected/corrected smoothed mean image relative pixel intensities',fontsize=14)
        plt.xlabel('layer number',fontsize=14)
        #fix the range on the x-axis to accommodate the legend
        plt.xlim(0,self.nlayers+10)
        plt.ylabel('pixel intensity relative to layer mean',fontsize=14)
        plt.legend(loc='best')
        plt.savefig(PIXEL_INTENSITY_PLOT_NAME)
        plt.close()

    #helper function to plot and save how many images ended up being stacked in each layer of the meanimage
    def __plotAndWriteNImagesStackedPerLayer(self) :
        xvals = list(range(1,self.nlayers+1))
        plt.plot([xvals[0],xvals[-1]],[self.n_images_read,self.n_images_read],linewidth=2,color='k',label=f'total images read ({self.n_images_read})')
        plt.plot(xvals,self.n_images_stacked_by_layer,marker='o',linewidth=2,label='n images stacked')
        plt.title(f'Number of images selected to be stacked by layer')
        plt.xlabel('image layer')
        plt.ylabel('number of images')
        plt.legend(loc='best')
        plt.savefig(N_IMAGES_STACKED_PER_LAYER_PLOT_NAME)
        plt.close()
        with open(N_IMAGES_STACKED_PER_LAYER_TEXT_FILE_NAME,'w') as fp :
            for li in range(self.nlayers) :
                fp.write(f'{self.n_images_stacked_by_layer[li]}\n')

#################### FILE-SCOPE HELPER FUNCTIONS ####################

#helper function to create a layered binary image mask for a given image array
#this can be run in parallel
def getImageMaskWorker(im_array,thresholds_per_layer,i,min_selected_pixels,make_plots,workingdir_name,return_dict) :
    nlayers = im_array.shape[-1]
    #create a new mask
    init_image_mask = np.empty(im_array.shape,np.uint8)
    #create a list to hold the threshold values
    thresholds = thresholds_per_layer
    #gently smooth the layer (on the GPU) to remove some noise
    smoothed_image = smoothImageWorker(im_array,GENTLE_GAUSSIAN_SMOOTHING_SIGMA)
    #if the thresholds haven't already been determined from the background, find them for all the layers by repeated Otsu thresholding
    if thresholds is None :
        thresholds=[]
        for li in range(nlayers) :
            sm_layer_array = smoothed_image[:,:,li]
            #rescale the image layer to 0-255 relative grayscale
            rs = (1.*np.iinfo(init_image_mask.dtype).max/np.max(sm_layer_array))
            im_layer_rescaled = (rs*sm_layer_array).astype('uint8')
            #iterate calculating and applying the Otsu threshold values (can't be done on the GPU)
            threshold=1000; iterations=0; bg_std = 100.; test_bg_std = 100.
            while test_bg_std>MAX_BG_PIXEL_MEAN_NORM_STD_DEV :
                test_threshold,_ = cv2.threshold(im_layer_rescaled[im_layer_rescaled<threshold],0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                test_threshold_rs = test_threshold/rs
                bg_pixels = (sm_layer_array[sm_layer_array<test_threshold_rs]).flatten()
                test_bg_std = np.std(bg_pixels)/np.mean(bg_pixels) if len(bg_pixels)>0 else 0
                if iterations==0 or test_bg_std>0 :
                    threshold = test_threshold
                    bg_std = test_bg_std
                    iterations+=1
            thresholds.append(threshold/rs)
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
    cv2.morphologyEx(init_mask_umat,cv2.MORPH_CLOSE,CO1_EL,intermediate_mask,borderType=cv2.BORDER_REPLICATE)
    cv2.morphologyEx(intermediate_mask,cv2.MORPH_OPEN,CO1_EL,co1_mask,borderType=cv2.BORDER_REPLICATE)
    #medium-scale close/open for the same reason with larger regions
    cv2.morphologyEx(co1_mask,cv2.MORPH_CLOSE,CO2_EL,intermediate_mask,borderType=cv2.BORDER_REPLICATE)
    cv2.morphologyEx(intermediate_mask,cv2.MORPH_OPEN,CO2_EL,co2_mask,borderType=cv2.BORDER_REPLICATE)
    #large close to define the bulk of the mask
    cv2.morphologyEx(co2_mask,cv2.MORPH_CLOSE,C3_EL,close3_mask,borderType=cv2.BORDER_REPLICATE)
    #repeated small open to eat its edges and remove small regions outside of the bulk of the mask
    cv2.morphologyEx(close3_mask,cv2.MORPH_OPEN,CO1_EL,open3_mask,iterations=OPEN_3_ITERATIONS,borderType=cv2.BORDER_REPLICATE)
    #the final mask is this last mask
    morphed_mask = open3_mask.get()
    #make the plots if requested
    if make_plots :
        flatfield_logger.info(f'Saving masking plots for image {i}')
        this_image_masking_plot_dirname = f'image_{i}_mask_layers'
        with cd(workingdir_name) :
            if not os.path.isdir(MASKING_PLOT_DIR_NAME) :
                os.mkdir(MASKING_PLOT_DIR_NAME)
            with cd(MASKING_PLOT_DIR_NAME) :
                if not os.path.isdir(this_image_masking_plot_dirname) :
                    os.mkdir(this_image_masking_plot_dirname)
        masking_plot_dirpath = os.path.join(workingdir_name,MASKING_PLOT_DIR_NAME,this_image_masking_plot_dirname)
        with cd(masking_plot_dirpath) :
            co1_mask = co1_mask.get()
            co2_mask = co2_mask.get()
            close3_mask = close3_mask.get()
            for li in range(nlayers) :
                f,ax = plt.subplots(4,2,figsize=MASKING_PLOT_FIG_SIZE)
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
    #add the total mask to the dict, along with its initial thresholds and number of optimal Otsu iterations per layer
    return_dict[i] = morphed_mask
