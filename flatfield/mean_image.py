#imports
from .config import *
from ..utilities.img_file_io import writeImageToFile
from ..utilities.misc import cd
import numpy as np, matplotlib.pyplot as plt, multiprocessing as mp
import os, cv2, statistics, copy

class FlatFieldError(Exception) :
    """
    Class for errors encountered during flatfielding
    """
    pass

class MeanImage :
    """
    Class to hold an image that is the mean of a bunch of stacked raw images 
    """
    def __init__(self,name,y,x,nlayers,dtype,max_nprocs,workingdir_name,skip_masking=False,smoothsigma=100) :
        """
        name            = stem to use for naming files that get created
        y               = y dimension of images (pixels)
        x               = x dimension of images (pixels)
        nlayers         = number of layers in images
        dtype           = datatype of image arrays that will be stacked
        max_nprocs      = max number of parallel processes to run when smoothing images
        workingdir_name = name of the directory to save everything in
        skip_masking    = if True, image layers won't be masked before being added to the stack
        smoothsigma     = Gaussian sigma for final smoothing of stacked flatfield image
        """
        self.namestem = name
        self.xpix = x
        self.ypix = y
        self.nlayers = nlayers
        self.max_nprocs = max_nprocs
        self.workingdir_name = workingdir_name
        self.skip_masking = skip_masking
        self.smoothsigma = smoothsigma
        self.image_stack = np.zeros((y,x,nlayers),dtype=np.uint32) #WARNING: may overflow if more than 65,535 masks are stacked 
        self.mask_stack  = np.zeros((y,x,nlayers),dtype=np.uint16) #WARNING: may overflow if more than 65,535 masks are stacked 
        self.smoothed_image_stack = np.zeros(self.image_stack.shape,dtype=IMG_DTYPE_OUT)
        self.n_images_stacked = 0
        self.threshold_lists_by_layer = []
        self.otsu_iteration_counts_by_layer = []
        self.bg_stddev_lists_by_layer = []
        for i in range(self.nlayers) :
            self.threshold_lists_by_layer.append([])
            self.otsu_iteration_counts_by_layer.append({})
            self.bg_stddev_lists_by_layer.append([])
        self.mean_image=None
        self.smoothed_mean_image=None
        self.flatfield_image=None

    #################### PUBLIC FUNCTIONS ####################

    def addGroupOfImages(self,im_array_list,make_plots=False) :
        """
        A function to add a list of raw image arrays to the image stack
        If masking is requested this function is parallelized and run on the GPU
        """
        #if the images aren't meant to be masked then we can just add them up trivially
        if self.skip_masking :
            for i,im_array in enumerate(im_array_list,start=1) :
                flatfield_logger.info(f'  adding image {self.n_images_stacked+1} to the stack....')
                self.image_stack+=im_array
                self.n_images_stacked+=1
            return
        #otherwise produce the image masks, apply them to the raw images, and be sure to add them to the list in the same order as the images
        manager = mp.Manager()
        return_dict = manager.dict()
        procs = []
        for i,im_array in enumerate(im_array_list,start=self.n_images_stacked+1) :
            flatfield_logger.info(f'  masking and adding image {i} to the stack....')
            p = mp.Process(target=getImageMaskWorker, args=(im_array,i,make_plots,self.workingdir_name,return_dict))
            procs.append(p)
            p.start()
        for proc in procs:
            proc.join()
        for i,im_array in enumerate(im_array_list,start=self.n_images_stacked+1) :
            thismask = return_dict[i]['mask']
            self.image_stack+=(im_array*thismask)
            self.mask_stack+=thismask
            self.n_images_stacked+=1
            initial_thresholds = return_dict[i]['thresholds']
            otsu_iterations = return_dict[i]['otsu_iterations']
            bg_stddevs = return_dict[i]['bg_stddevs']
            for li in range(self.nlayers) :
                (self.threshold_lists_by_layer[li]).append((initial_thresholds[li]))
                if otsu_iterations[li] not in self.otsu_iteration_counts_by_layer[li].keys() :
                    self.otsu_iteration_counts_by_layer[li][otsu_iterations[li]] = 0
                self.otsu_iteration_counts_by_layer[li][otsu_iterations[li]]+=1
                (self.bg_stddev_lists_by_layer[li]).append(bg_stddevs[li])

    def makeFlatFieldImage(self) :
        """
        A function to get the mean of the image stack and smooth/normalize each of its layers to make the flatfield image
        """
        self.mean_image = self.__makeMeanImage()
        return_list = []
        smoothImageLayerByLayerWorker(self.mean_image,self.smoothsigma,return_list)
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
            writeImageToFile(np.transpose(self.mask_stack,(2,1,0)),f'mask_stack{FILE_EXT}',dtype=np.uint16)

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
                #plot of the flatfield images' minimum and and maximum (and 5/95%ile) pixel intensities
                self.__saveFlatFieldImagePixelIntensityPlot()
                if not self.skip_masking :
                    #save the plot of the initial masking thresholds by layer
                    self.__saveMaskThresholdsPlot()
                    #save the plot of the chosen number of Otsu iterations per layer
                    self.__saveOtsuIterationPlot()
                    #save the plot of the mean-normalized standard deviation of the background pixels per layer
                    self.__saveBackgroundPixelStdDevPlot()
        

    #################### PRIVATE HELPER FUNCTIONS ####################

    #helper function to create and return the mean image from the image (and mask, if applicable, stack)
    def __makeMeanImage(self) :
        #if the images haven't been masked then this is trivial
        if self.skip_masking :
            return self.image_stack/self.n_images_stacked
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
                #for the mean image
                f,ax = plt.subplots(figsize=fig_size)
                pos = ax.imshow(self.mean_image[:,:,layer_i])
                ax.set_title(f'mean image, {layer_titlestem}')
                f.colorbar(pos,ax=ax)
                plt.savefig(f'mean_image_{layer_fnstem}.png')
                plt.close()
                #for the flatfield image
                f,ax = plt.subplots(figsize=fig_size)
                pos = ax.imshow(self.flatfield_image[:,:,layer_i])
                ax.set_title(f'flatfield, {layer_titlestem}')
                f.colorbar(pos,ax=ax)
                plt.savefig(f'flatfield_{layer_fnstem}.png')
                plt.close()
                #for the mask stack (if applicable) 
                if not self.skip_masking :
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
        plt.figure(figsize=(INTENSITY_FIG_WIDTH,(9./16.)*INTENSITY_FIG_WIDTH))
        xaxis_vals = list(range(1,self.nlayers+1))
        yaxis_min_val = 100.
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
            if 1.-stddev<yaxis_min_val :
                yaxis_min_val=1.-stddev
        #plot the inensity plots together, with the broadband filter breaks
        plt.plot([xaxis_vals[0],xaxis_vals[-1]],[1.0,1.0],color='mediumseagreen',linestyle='dashed',label='mean intensity')
        for i in range(len(LAST_FILTER_LAYERS)+1) :
            f_i = 0 if i==0 else LAST_FILTER_LAYERS[i-1]
            l_i = xaxis_vals[-1] if i==len(LAST_FILTER_LAYERS) else LAST_FILTER_LAYERS[i]
            if i==0 :
                plt.plot(xaxis_vals[f_i:l_i],ff_min_pixel_intensities[f_i:l_i],color='darkblue',marker='o',linewidth=2,label='minimum intensity')
                plt.plot(xaxis_vals[f_i:l_i],ff_low_pixel_intensities[f_i:l_i],color='royalblue',marker='o',linewidth=2,linestyle='dotted',label=r'5th %ile intensity')
                plt.plot(xaxis_vals[f_i:l_i],ff_max_pixel_intensities[f_i:l_i],color='darkred',marker='o',linewidth=2,label='maximum intensity')
                plt.plot(xaxis_vals[f_i:l_i],ff_high_pixel_intensities[f_i:l_i],color='lightcoral',marker='o',linewidth=2,linestyle='dotted',label=r'95th %ile intensity')
                plt.plot([l_i+0.5,l_i+0.5],[min(ff_min_pixel_intensities)-0.1,max(ff_max_pixel_intensities)+0.1],color='black',linewidth=2,linestyle='dotted',label='broadband filter changeover')
            else :
                plt.plot(xaxis_vals[f_i:l_i],ff_min_pixel_intensities[f_i:l_i],color='darkblue',marker='o',linewidth=2)
                plt.plot(xaxis_vals[f_i:l_i],ff_low_pixel_intensities[f_i:l_i],color='royalblue',marker='o',linewidth=2,linestyle='dotted')
                plt.plot(xaxis_vals[f_i:l_i],ff_max_pixel_intensities[f_i:l_i],color='darkred',marker='o',linewidth=2)
                plt.plot(xaxis_vals[f_i:l_i],ff_high_pixel_intensities[f_i:l_i],color='lightcoral',marker='o',linewidth=2,linestyle='dotted')
                if i!=len(LAST_FILTER_LAYERS) :
                    plt.plot([l_i+0.5,l_i+0.5],[min(ff_min_pixel_intensities)-0.1,max(ff_max_pixel_intensities)+0.1],color='black',linewidth=2,linestyle='dotted')
        if min(ff_min_pixel_intensities)<yaxis_min_val :
            yaxis_min_val=min(ff_min_pixel_intensities)
        plt.title(f'flatfield image layer normalized pixel intensities',fontsize=14)
        plt.xlabel('layer number',fontsize=14)
        #fix the range on the y-axis to accommodate the legend
        plt.ylim(yaxis_min_val-0.2,max(ff_max_pixel_intensities)+0.2)
        bot,top = plt.gca().get_ylim()
        newaxisrange=1.35*(top-bot)
        plt.ylim(bot,bot+newaxisrange)
        plt.ylabel('pixel intensity',fontsize=14)
        plt.legend(loc='best')
        plt.savefig('pixel_intensity_plot.png')
        plt.close()

    #helper function to plot the optimal initial flux thresholds for the masks chosen in each layer
    def __saveMaskThresholdsPlot(self) :
        xvals=list(range(1,self.nlayers+1))
        mean_thresholds  = [statistics.mean(layer_thresholds) for layer_thresholds in self.threshold_lists_by_layer]
        threshold_stdevs = [statistics.pstdev(layer_thresholds) for layer_thresholds in self.threshold_lists_by_layer]
        plt.errorbar(xvals,mean_thresholds,yerr=threshold_stdevs,marker='o',linewidth=2)
        plt.title('optimal initial thresholds by layer')
        plt.xlabel('image layer')
        plt.ylabel('threshold value')
        plt.savefig('initial_masking_thresholds_by_layer.png')
        plt.close()

    #helper function to plot how many images in each layer chose the first, second, and third otsu iterations as the optimal initial thresholds
    def __saveOtsuIterationPlot(self) :
        xvals=list(range(1,self.nlayers+1))
        yval_label_dict = {1:'first',2:'second',3:'third',4:'fourth',5:'fifth',6:'sixth',7:'seventh',8:'eighth',9:'ninth',10:'tenth'}
        yval_marker_dict = {1:'o',2:'v',3:'^',4:'<',5:'>',6:'s',7:'D',8:'P',9:'*',10:'X'}
        yvals_dict = {}
        for li1 in range(self.nlayers) :
            for noi,nimages in self.otsu_iteration_counts_by_layer[li1].items() :
                if noi not in yvals_dict.keys() :
                    yvals_dict[noi]=[]
                    for li2 in range(self.nlayers) :
                        yvals_dict[noi].append(0)
                yvals_dict[noi][li1]=nimages
        for noi,yvals in sorted(yvals_dict.items()) :
            plt.plot(xvals,yvals,marker=yval_marker_dict[noi],linewidth=2,label=f'# of {yval_label_dict[noi]} iterations')
        plt.plot([xvals[0],xvals[-1]],[self.n_images_stacked,self.n_images_stacked],linewidth=2,color='k',linestyle='dotted',label='total images stacked')
        plt.title('Optimal Otsu iteration choices by image layer')
        plt.xlabel('image layer')
        bot,top = plt.gca().get_ylim()
        newaxisrange=1.2*(top-bot)
        plt.ylim(bot,bot+newaxisrange)
        plt.ylabel('# of images from the sample')
        plt.legend(loc='best')
        plt.savefig('otsu_threshold_choices_by_layer.png')
        plt.close()

    #helper function to save a plot of the mean-normalized final background pixel standard deviations in each layer
    def __saveBackgroundPixelStdDevPlot(self) :
        xvals=list(range(1,self.nlayers+1))
        mean_bg_stddevs  = [statistics.mean(layer_bg_stddevs) for layer_bg_stddevs in self.bg_stddev_lists_by_layer]
        bg_stddev_stdevs = [statistics.pstdev(layer_bg_stddevs) for layer_bg_stddevs in self.bg_stddev_lists_by_layer]
        plt.errorbar(xvals,mean_bg_stddevs,yerr=bg_stddev_stdevs,marker='o',linewidth=2)
        plt.title('mean-normalized std. devs. of background pixel flux  by layer')
        plt.xlabel('image layer')
        plt.ylabel('mean-normalized background pixel std. dev.')
        plt.savefig('background_pixel_stddevs_by_layer.png')
        plt.close()

#################### FILE-SCOPE HELPER FUNCTIONS ####################

#helper function to create a layered binary image mask for a given image array
#this can be run in parallel
def getImageMaskWorker(im_array,i,make_plots,workingdir_name,return_dict) :
    nlayers = im_array.shape[-1]
    #create a new mask
    init_image_mask = np.empty(im_array.shape,np.uint8)
    #create lists to hold the threshold values, numbers of Otsu iterations, and final background pixel standard deviations for each layer
    thresholds = []; otsu_iterations = []; bg_stddevs = []
    #gently smooth the layer (on the GPU) to remove some noise
    smoothed_image = cv2.GaussianBlur(im_array,(0,0),GENTLE_GAUSSIAN_SMOOTHING_SIGMA,borderType=cv2.BORDER_REPLICATE)
    #determine the initial masks for all the layers by repeated Otsu thresholding
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
        #save the mask, its threshold, how many iterations it took, and the standard deviation of the raw image background pixels
        _,otsu_mask = cv2.threshold(im_layer_rescaled,threshold,1,cv2.THRESH_BINARY)
        init_image_mask[:,:,li] = otsu_mask
        thresholds.append(threshold/rs)
        otsu_iterations.append(iterations)
        bg_stddevs.append(bg_std)
    #morph each layer of the mask through a series of operations on the GPU
    init_mask_umat  = cv2.UMat(init_image_mask)
    erode1_mask=cv2.UMat(np.empty_like(init_image_mask))
    erode2_mask=cv2.UMat(np.empty_like(init_image_mask))
    erode3_mask=cv2.UMat(np.empty_like(init_image_mask))
    dilate1_mask=cv2.UMat(np.empty_like(init_image_mask))
    dilate2_mask=cv2.UMat(np.empty_like(init_image_mask))
    dilate3_mask=cv2.UMat(np.empty_like(init_image_mask))
    #erode away small noisy regions
    cv2.erode(init_mask_umat,ERODE1_EL,erode1_mask)
    #dilate to fill in between chunks
    cv2.dilate(erode1_mask,DILATE1_EL,dilate1_mask)
    cv2.dilate(dilate1_mask,DILATE2_EL,dilate2_mask)
    #erode away the regions that began as very small pixel clusters
    cv2.erode(dilate2_mask,ERODE2_EL,erode2_mask)
    cv2.erode(erode2_mask,ERODE3_EL,erode3_mask)
    #dilate back the edges that were lost in the bulk of the mask
    cv2.dilate(erode3_mask,DILATE3_EL,dilate3_mask,iterations=DILATE3_ITERATIONS)
    #the final mask is this dilated mask
    morphed_mask = dilate3_mask.get()
    ##finally, refine the masks by stacking them all up and adding/removing the pixels about which we're very confident
    #mask_stack = np.sum(morphed_mask,axis=-1)
    #mask_stack_add  = np.where(mask_stack>=(ADD_IF_SHARED_IN_AT_LEAST*nlayers),1,0)
    #mask_stack_drop = np.where(mask_stack<=((1.-DROP_IF_ABSENT_IN_AT_LEAST)*nlayers),1,0)
    #refined_mask = np.empty_like(init_image_mask_1)
    #for li in range(nlayers) :
    #    refined_mask[:,:,li] = np.clip(morphed_mask[:,:,li]+mask_stack_add-mask_stack_drop,0,1)
    #add the total mask to the dict, along with its initial thresholds and number of optimal Otsu iterations per layer
    return_dict[i] = {'mask':morphed_mask,'thresholds':thresholds,'otsu_iterations':otsu_iterations,'bg_stddevs':bg_stddevs}
    #make the plots if requested
    if make_plots :
        this_image_masking_plot_dirname = f'image_{i}_mask_layers'
        with cd(workingdir_name) :
            if not os.path.isdir(MASKING_PLOT_DIR_NAME) :
                os.mkdir(MASKING_PLOT_DIR_NAME)
            with cd(MASKING_PLOT_DIR_NAME) :
                if not os.path.isdir(this_image_masking_plot_dirname) :
                    os.mkdir(this_image_masking_plot_dirname)
        masking_plot_dirpath = os.path.join(workingdir_name,MASKING_PLOT_DIR_NAME,this_image_masking_plot_dirname)
        with cd(masking_plot_dirpath) :
            erode1_mask = erode1_mask.get()
            dilate2_mask = dilate2_mask.get()
            erode3_mask = erode3_mask.get()
            for li in range(nlayers) :
                f,ax = plt.subplots(4,2,figsize=MASKING_PLOT_FIG_SIZE)
                im = im_array[:,:,li]
                im_grayscale = im/np.max(im)
                im = (np.clip(im,0,255)).astype('uint8')
                ax[0][0].imshow(im_grayscale,cmap='gray')
                ax[0][0].set_title('raw image in grayscale',fontsize=14)
                ax[0][1].imshow(init_image_mask[:,:,li])
                ax[0][1].set_title(f'init mask (thresh. = {thresholds[li]:.1f}, {otsu_iterations[li]} it.s, bg std={bg_stddevs[li]:.3f})',fontsize=14)
                ax[1][0].imshow(erode1_mask[:,:,li])
                ax[1][0].set_title('mask after initial erode',fontsize=14)
                ax[1][1].imshow(dilate2_mask[:,:,li])
                ax[1][1].set_title('mask after 1st and 2nd dilations',fontsize=14)
                ax[2][0].imshow(erode3_mask[:,:,li])
                ax[2][0].set_title('mask after 2nd and 3rd erosions',fontsize=14)
                m = morphed_mask[:,:,li]
                ax[2][1].imshow(m)
                ax[2][1].set_title('final mask after 3rd dilation',fontsize=14)
                overlay_clipped = np.array([im,im*m,im*m]).transpose(1,2,0)
                overlay_grayscale = np.array([im_grayscale*m,im_grayscale*m,0.15*m]).transpose(1,2,0)
                ax[3][0].imshow(overlay_clipped)
                ax[3][0].set_title('mask overlaid with clipped image',fontsize=14)
                ax[3][1].imshow(overlay_grayscale)
                ax[3][1].set_title('mask overlaid with grayscale image',fontsize=14)
                figname = f'image_{i}_layer_{li+1}_masks.png'
                plt.savefig(figname)
                plt.close()

#helper function to smooth each layer of an image (done on the CPU so they can be done all at once)
#this can be run in parallel
def smoothImageLayerByLayerWorker(im_array,smoothsigma,return_list) :
    smoothed_im_array = cv2.GaussianBlur(im_array,(0,0),smoothsigma,borderType=cv2.BORDER_REPLICATE)
    return_list.append(smoothed_im_array)

