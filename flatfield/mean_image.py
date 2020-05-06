#imports
from .config import *
from ..utilities.img_file_io import writeImageToFile
from ..utilities.misc import cd
import numpy as np, matplotlib.pyplot as plt, multiprocessing as mp
import os, cv2, statistics

class FlatFieldError(Exception) :
    """
    Class for errors encountered during flatfielding
    """
    pass

class MeanImage :
    """
    Class to hold an image that is the mean of a bunch of stacked raw images 
    """
    def __init__(self,name,y,x,nlayers,dtype,max_nprocs,skip_masking=False,smoothsigma=100) :
        """
        name           = stem to use for naming files that get created
        y              = y dimension of images (pixels)
        x              = x dimension of images (pixels)
        nlayers        = number of layers in images
        dtype          = datatype of image arrays that will be stacked
        max_nprocs     = max number of parallel processes to run when smoothing images
        smoothsigma    = Gaussian sigma for final smoothing of stacked flatfield image
        """
        self.namestem = name
        self.xpix = x
        self.ypix = y
        self.nlayers = nlayers
        self.max_nprocs = max_nprocs
        self.skip_masking = skip_masking
        self.smoothsigma = smoothsigma
        self.image_stack = np.zeros((y,x,nlayers),dtype=dtype)
        self.mask_stack  = np.zeros((y,x,nlayers),dtype=np.uint8) #the masks are always 8-bit unsigned ints 
        self.smoothed_image_stack = np.zeros(self.image_stack.shape,dtype=IMG_DTYPE_OUT)
        self.n_images_stacked = 0
        self.threshold_lists_by_layer = []
        for i in range(self.nlayers) :
            self.threshold_lists_by_layer.append([])
        self.n_otsu1_masks_per_layer  = self.nlayers*[0]
        self.n_otsu2_masks_per_layer  = self.nlayers*[0]
        self.n_otsu3_masks_per_layer  = self.nlayers*[0]
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
                flatfield_logger.info(f'  adding image {self.n_images_stacked+1} to the stack....')
                self.image_stack+=im_array
                self.n_images_stacked+=1
            return
        #otherwise produce the image masks, apply them to the raw images, and be sure to add them to the list in the same order as the images
        manager = mp.Manager()
        return_dict = manager.dict()
        procs = []
        for i,im_array in enumerate(im_array_list,start=1) :
            flatfield_logger.info(f'  masking and adding image {self.n_images_stacked+i} to the stack....')
            p = mp.Process(target=getImageMaskWorker, args=(im_array,i,return_dict))
            procs.append(p)
            p.start()
        for proc in procs:
            proc.join()
        for i,im_array in enumerate(im_array_list,start=1) :
            thismask = return_dict[i]['mask']
            self.image_stack+=(im_array*thismask)
            self.mask_stack+=thismask
            self.n_images_stacked+=1
            initial_thresholds = return_dict[i]['thresholds']
            print(f'initial thresholds={initial_thresholds}')
            otsu_choices = return_dict[i]['otsu_iterations']
            for li in range(self.nlayers) :
                print(f'{i}, {li}')
                print(self.threshold_lists_by_layer)
                (self.threshold_lists_by_layer[li]).append((initial_thresholds[li]))
                if otsu_choices[li]==1 :
                    self.n_otsu1_masks_per_layer[li]+=1
                elif otsu_choices[li]==2 :
                    self.n_otsu2_masks_per_layer[li]+=1
                elif otsu_choices[li]==3 :
                    self.n_otsu3_masks_per_layer[li]+=1

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
            writeImageToFile(np.transpose(self.mask_stack,(2,1,0)),f'mask_stack{FILE_EXT}',dtype=np.uint8)

    def savePlots(self) :
        """
        Make and save several visualizations of the image layers and details of the masking and flatfielding
        """
        #.pngs of the mean image, flatfield, and mask stack layers
        self.__saveImageLayerPlots()
        #plot of the flatfield images' minimum and and maximum (and 5/95%ile) pixel intensities
        self.__saveFlatFieldImagePixelIntensityPlot()
        #save the plot of the initial masking thresholds by layer
        self.__saveMaskThresholdsPlot()
        #save the plot of the chosen number of Otsu iterations per layer
        self.__saveOtsuIterationPlot()
        

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

    #helper function to plot the pixel intensities of the layers in the flatfield image
    def __saveFlatFieldImagePixelIntensityPlot(self) :
        #keep track of the flatfield images' minimum and maximum (and 5/95%ile) pixel intensities while the other plots are made
        ff_min_pixel_intensities=[]
        ff_low_pixel_intensities=[]
        ff_max_pixel_intensities=[]
        ff_high_pixel_intensities=[]
        plt.figure(figsize=(INTENSITY_FIG_WIDTH,(9./16.)*INTENSITY_FIG_WIDTH))
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
        plt.close()

    #helper function to plot the optimal initial flux thresholds for the masks chosen in each layer
    def __saveMaskThresholdsPlot(self) :
        xvals=list(range(1,self.nlayers+1))
        mean_thresholds  = [statistics.mean(layer_thresholds) for layer_thresholds in self.threshold_lists_by_layer]
        threshold_stdevs = [statistics.pstdev(layer_thresholds) for layer_thresholds in self.threshold_lists_by_layer]
        plt.errorbar(xvals,mean_thresholds,yerr=threshold_stdevs,marker='o')
        plt.title('optimal initial thresholds by layer')
        plt.xlabel('image layer')
        plt.ylabel('threshold value (0-255)')
        plt.savefig('initial_masking_thresholds_by_layer.png')
        plt.close()

    #helper function to plot how many images in each layer chose the first, second, and third otsu iterations as the optimal initial thresholds
    def __saveOtsuIterationPlot(self) :
        xvals=list(range(1,self.nlayers+1))
        plt.plot(xvals,self.n_otsu1_masks_per_layer,marker='o',color='r',linewidth=2,label='# of first iterations')
        plt.plot(xvals,self.n_otsu2_masks_per_layer,marker='o',color='g',linewidth=2,label='# of second iterations')
        plt.plot(xvals,self.n_otsu3_masks_per_layer,marker='o',color='b',linewidth=2,label='# of third iterations')
        plt.title('Optimal Otsu iteration choices by image layer')
        plt.xlabel('image layer')
        plt.ylabel('# of images from the sample')
        plt.legend(loc='best')
        plt.savefig('otsu_threshold_choices_by_layer.png')
        plt.close()

#################### FILE-SCOPE HELPER FUNCTIONS ####################

#helper function to create a layered binary image mask for a given image array
#this can be run in parallel
def getImageMaskWorker(im_array,i,return_dict) :
    nlayers = im_array.shape[-1]
    #create three multilayer masks, one for each thresholding point that will be calculated
    init_image_mask_1 = np.empty(im_array.shape,np.uint8)
    init_image_mask_2 = np.empty_like(init_image_mask_1)
    init_image_mask_3 = np.empty_like(init_image_mask_1)
    #create lists to hold the threshold values for each layer
    thresholds_1 = []; thresholds_2 = []; thresholds_3 = []
    #create the initial masks for all the layers
    for li in range(nlayers) :
        #gently smooth the layer (on the GPU) to remove some noise
        layer_in_umat = cv2.UMat(im_array[:,:,li])
        layer_out_umat = cv2.UMat(np.empty_like(im_array[:,:,li]))
        cv2.GaussianBlur(layer_in_umat,GENTLE_GAUSSIAN_SMOOTHING_KERNEL,0,layer_out_umat,0,cv2.BORDER_REFLECT)
        layer_array = (layer_out_umat.get()).astype('uint8')
        #calculate and apply the three Otsu threshold values (can't be done on the GPU)
        t1,init_image_mask_1[:,:,li]=cv2.threshold(layer_array,0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        t2,_=cv2.threshold(layer_array[layer_array<t1],0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)        
        _,init_image_mask_2[:,:,li]=cv2.threshold(layer_array,t2,1,cv2.THRESH_BINARY)
        t3,_=cv2.threshold(layer_array[layer_array<t2],0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        _,init_image_mask_3[:,:,li]=cv2.threshold(layer_array,t3,1,cv2.THRESH_BINARY)
        #add the threshold values to the lists
        thresholds_1.append(t1); thresholds_2.append(t2); thresholds_3.append(t3)
    #morph each layer of the three mask stacks through a series of operations on the GPU
    init_mask_umats  = [cv2.UMat(init_image_mask_1),cv2.UMat(init_image_mask_2),cv2.UMat(init_image_mask_3)]
    morphed_masks = []
    for init_mask in init_mask_umats :
        erode1_mask=cv2.UMat(np.empty_like(init_image_mask_1))
        erode2_mask=cv2.UMat(np.empty_like(init_image_mask_1))
        erode3_mask=cv2.UMat(np.empty_like(init_image_mask_1))
        dilate1_mask=cv2.UMat(np.empty_like(init_image_mask_1))
        dilate2_mask=cv2.UMat(np.empty_like(init_image_mask_1))
        dilate3_mask=cv2.UMat(np.empty_like(init_image_mask_1))
        #erode away small noisy regions
        cv2.erode(init_mask,ERODE1_EL,erode1_mask)
        #dilate to fill in between chunks
        cv2.dilate(erode1_mask,DILATE1_EL,dilate1_mask)
        cv2.dilate(dilate1_mask,DILATE2_EL,dilate2_mask)
        #erode away the regions that began as very small pixel clusters
        cv2.erode(dilate2_mask,ERODE2_EL,erode2_mask)
        cv2.erode(erode2_mask,ERODE3_EL,erode3_mask)
        #dilate back the edges that were lost in the bulk of the mask
        cv2.dilate(erode3_mask,DILATE3_EL,dilate3_mask,iterations=DILATE3_ITERATIONS)
        #the final mask is this dilated mask
        morphed_masks.append(dilate3_mask.get())
    morphed_mask_1, morphed_mask_2, morphed_mask_3 = morphed_masks
    #choose which of the three masks is the best option layer-by-layer (storing the thresholds and number of Otsu steps)
    chosen_mask = np.empty_like(init_image_mask_1); chosen_thresholds = []; chosen_otsu_iterations = []
    for li in range(nlayers) :
        im = im_array[:,:,li].astype('uint8')
        im_brightest = np.max(im)
        first_pixels_added  = im*(morphed_mask_2[:,:,li]-morphed_mask_1[:,:,li])
        second_pixels_added = im*(morphed_mask_3[:,:,li]-morphed_mask_2[:,:,li])
        first_nonzero_added  = first_pixels_added[first_pixels_added!=0].flatten()
        second_nonzero_added = second_pixels_added[second_pixels_added!=0].flatten()
        first_pa_ninetyfifth  = np.sort(first_nonzero_added)[int(0.95*len(first_nonzero_added))] if len(first_nonzero_added)!=0 else im_brightest
        second_pa_ninetyfifth = np.sort(second_nonzero_added)[int(0.95*len(second_nonzero_added))] if len(second_nonzero_added)!=0 else im_brightest
        choice=1        
        if first_pa_ninetyfifth/im_brightest>CHOICE_THRESHOLD_1 :
            choice=2
            if second_pa_ninetyfifth/im_brightest>CHOICE_THRESHOLD_2 :
                choice=3
        chosen_otsu_iterations.append(choice)
        if choice==1 :
            chosen_mask[:,:,li] = morphed_mask_1[:,:,li]; chosen_thresholds.append(thresholds_1[li])
        elif choice==2 :
            chosen_mask[:,:,li] = morphed_mask_2[:,:,li]; chosen_thresholds.append(thresholds_2[li])
        elif choice==3 :
            chosen_mask[:,:,li] = morphed_mask_3[:,:,li]; chosen_thresholds.append(thresholds_3[li])
    #finally, refine the masks by stacking them all up and adding/removing the pixels about which we're very confident
    mask_stack = np.sum(chosen_mask,axis=-1)
    mask_stack_add  = np.where(mask_stack>=(ADD_IF_SHARED_IN_AT_LEAST*nlayers),1,0)
    mask_stack_drop = np.where(mask_stack<=((1.-DROP_IF_ABSENT_IN_AT_LEAST)*nlayers),1,0)
    refined_mask = np.empty_like(init_image_mask_1)
    for li in range(nlayers) :
        refined_mask[:,:,li] = np.clip(chosen_mask[:,:,li]+mask_stack_add-mask_stack_drop,0,1)
    #add the total mask to the dict, along with its initial thresholds and number of optimal Otsu iterations per layer
    return_dict[i] = {'mask':refined_mask,'thresholds':chosen_thresholds,'otsu_iterations':chosen_otsu_iterations}

#helper function to smooth each layer of an image independently on the GPU
#this can be run in parallel
def smoothImageLayerByLayerWorker(im_array,smoothsigma,return_list) :
    smoothed_im_array = np.zeros_like(im_array)
    nlayers = im_array.shape[-1]
    for li in range(nlayers) :
        layer_in_umat = cv2.UMat(im_array[:,:,li])
        layer_out_umat = cv2.UMat(smoothed_im_array[:,:,li])
        cv2.GaussianBlur(layer_in_umat,(0,0),smoothsigma,layer_out_umat,smoothsigma,cv2.BORDER_CONSTANT)
        smoothed_im_array[:,:,li]=layer_out_umat.get()
    return_list.append(smoothed_im_array)

