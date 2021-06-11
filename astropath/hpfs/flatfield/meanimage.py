#imports
from .config import CONST 
from ..image_masking.image_mask import ImageMask
from ...utilities.img_file_io import writeImageToFile, smooth_image_worker, smoothImageWithUncertaintyWorker
from ...utilities.tableio import writetable
from ...utilities.misc import cd, dummylogger
from ...utilities.config import CONST as UNIV_CONST
from multiprocessing.pool import Pool
import numpy as np
import pathlib

class MeanImage :
    """
    Class representing an image that is the mean of a bunch of stacked raw images 
    Includes tools to build meanimages in a variety of ways including masking out empty background and artifacts in images as they're stacked
    """

    #################### PUBLIC FUNCTIONS ####################

    def __init__(self,logger=dummylogger) :
        """
        logger = the logging object to use (passed from whatever is using this meanimage)
        """
        self.__logger = logger
        self.__image_stack = None
        self.__image_squared_stack = None
        self.__mask_stack = None
        self.__labelled_mask_regions = []
        self.__n_images_read = 0
        self.__n_images_stacked_by_layer = None
        self.__mean_image=None
        self.__std_err_of_mean_image=None

    def stack_images(self,rectangles,thresholds,med_ets,skip_masking=False,n_threads=1) :
        """
        Loop over a set of given images and add them to the stack, masking them as they're added and keeping track of that information
        
        rectangles = the list of rectangles whose images should be added to the stack
        thresholds = the list of background thresholds in counts per layer
        med_ets = the slide median exposure times to which each rectangle image has been corrected 
                  (if None then corrections were skipped and individual image exposure times will be used instead)
        skip_masking = if True, image layers won't be masked before being added to the stack
        n_threads  = the maximum number of processes to open at once while adding images
        """
        self.__logger.info(f'Stacking {len(self.tissue_bulk_rects)} images in the bulk of the tissue')
        #initialize the image and mask stacks based on the image dimensions
        img_dims = rectangles[0].imageshapeinoutput
        self.__image_stack = np.zeros(img_dims,dtype=np.float64)
        self.__image_squared_stack = np.zeros(img_dims,dtype=np.float64)
        if not skip_masking :
            self.__mask_stack = np.zeros(img_dims,dtype=np.uint64)
        self.__n_images_stacked_by_layer = np.zeros((img_dims[-1]),dtype=np.uint64)
        #if the images aren't going to be masked, this whole step is pretty trivial. Otherwise it's a bit more complex
        if skip_masking :
            self.__stack_images_no_masking(rectangles,med_ets)
        else :
            self.__stack_images_with_masking(rectangles,thresholds,med_ets,n_threads)

        #################### PRIVATE HELPER FUNCTIONS ####################

        def __stack_images_no_masking(self,rectangles,med_ets) :
            """
            Simply add all the images to the image_stack and image_squared_stack without masking them
            """
            if med_ets is not None :
                med_ets = med_ets[np.newaxis,np.newaxis,:]
            for ri,r in enumerate(rectangles) :
                self.__logger.info(f'Adding {r.file.rstrip(UNIV_CONST.IM3_EXT)} to the meanimage stack ({ri+1} of {len(rectangles)})....')
                with r.using_image() as im :
                    normalized_image = im / med_ets if med_ets is not None else im / r.allexposuretimes[np.newaxis,np.newaxis,:]
                    self.__image_stack+=normalized_image
                    self.__image_squared_stack+=np.power(normalized_image,2)
                    self.__n_images_read+=1
                    self.__n_images_stacked_by_layer+=1

        def __stack_images_with_masking(self,rectangles,thresholds,med_ets,n_threads) :
            """
            Compute all of the image masks and add the masked images and their masks to the stacks 
            """
            #get the masks and add masked images to the stack in parallel
            if n_threads>1 :
                proc_results = {}
                with Pool(processes=n_threads) as pool :
                    for ri,r in enumerate(rectangles) :
                        self.__logger.info(f'Masking and adding {r.file.rstrip(UNIV_CONST.IM3_EXT)} to the meanimage stack ({ri+1} of {len(rectangles)})....')
                        with r.using_image() as im :
                            proc_results[r] = pool.apply_async(ImageMask,(im,
                                                                          r.file.rstrip(UNIV_CONST.IM3_EXT),
                                                                          thresholds,
                                                                          med_ets if med_ets is not None else r.allexposuretimes,
                                                                          self.__image_masking_dirpath,
                                                                         )
                                                              )
                for rect,res in proc_results.items() :
                    try :
                        mask = res.get()
                        onehot_mask = mask.onehot_mask
                        self.__labelled_mask_regions+=mask.labelled_mask_regions
                        layers_to_add = np.where(np.sum(onehot_mask,axis=(0,1))/(onehot_mask.shape[0]*onehot_mask.shape[1])>=CONST.MIN_PIXEL_FRAC,1,0)
                        with rect.using_image() as im :
                            masked_im = im*layers_to_add[np.newaxis,np.newaxis,:]*onehot_mask
                        normalized_masked_im = masked_im / med_ets if med_ets is not None else masked_im / rect.allexposuretimes[np.newaxis,np.newaxis,:]
                        self.__image_stack+=normalized_masked_im
                        self.__image_squared_stack+=np.power(normalized_masked_im,2)
                        self.__n_images_read+=1
                        self.__n_images_stacked_by_layer+=layers_to_add
                    except Exception as e :
                        warnmsg = f'WARNING: getting image mask for rectangle {rect.n} ({rect.file.rstrip(UNIV_CONST.IM3_EXT)}) failed '
                        warnmsg+= f'with the error "{e}" and this rectangle WILL BE SKIPPED when stacking images in the meanimage!'
                        self.__logger.warning(warnmsg)
            #do the same as above except serially
            else :
                for ri,r in enumerate(rectangles) :
                    self.__logger.info(f'Masking and adding {r.file.rstrip(UNIV_CONST.IM3_EXT)} to the meanimage stack ({ri+1} of {len(rectangles)})....')
                    try :
                        with r.using_image() as im :
                            mask = ImageMask(im,
                                             r.file.rstrip(UNIV_CONST.IM3_EXT),
                                             thresholds,
                                             med_ets if med_ets is not None else r.allexposuretimes,
                                             self.__image_masking_dirpath,
                                            )
                            onehot_mask = mask.onehot_mask
                            self.__labelled_mask_regions+=mask.labelled_mask_regions
                            layers_to_add = np.where(np.sum(onehot_mask,axis=(0,1))/(onehot_mask.shape[0]*onehot_mask.shape[1])>=CONST.MIN_PIXEL_FRAC,1,0)
                            masked_im = im*layers_to_add[np.newaxis,np.newaxis,:]*onehot_mask
                        normalized_masked_im = masked_im / med_ets if med_ets is not None else masked_im / rect.allexposuretimes[np.newaxis,np.newaxis,:]
                        self.__image_stack+=normalized_masked_im
                        self.__image_squared_stack+=np.power(normalized_masked_im,2)
                        self.__n_images_read+=1
                        self.__n_images_stacked_by_layer+=layers_to_add
                    except Exception as e :
                        warnmsg = f'WARNING: getting image mask for rectangle {r.n} ({r.file.rstrip(UNIV_CONST.IM3_EXT)}) failed '
                        warnmsg+= f'with the error "{e}" and this rectangle WILL BE SKIPPED when stacking images in the meanimage!'
                        self.__logger.warning(warnmsg)

#    def addGroupOfImages(self,im_array_list,rfps,slide,min_pixel_frac,ets_for_normalization=None,masking_plot_indices=[],logger=None) :
#        """
#        A function to add a list of raw image arrays to the image stack
#        If masking is requested this function's subroutines are parallelized and also run on the GPU
#        im_array_list         = list of image arrays to add
#        rfps                  = list of image rawfile paths corresponding to image arrays
#        slide                 = flatfield_slide object corresponding to this group of images
#        min_pixel_frac        = fraction (0->1) of how many pixels must be selected as signal for an image to be stacked
#        ets_for_normalization = list of exposure times to use for normalizating images to counts/ms before stacking
#        masking_plot_indices  = list of image array list indices whose masking plots will be saved
#        logger                = a RunLogger object whose context is entered, if None the default log will be used
#        """
#        #make sure the exposure time normalization can be done
#        if (ets_for_normalization is not None) and (len(ets_for_normalization)!=self.nlayers) :
#            raise ValueError(f'ERROR: list of layer exposure times has length {len(ets_for_normalization)} but images have {self.nlayers} layers!')
#        stacked_in_layers = []
#        #if the images aren't meant to be masked then we can just add them up trivially
#        if self.skip_masking :
#            for i,im_array in enumerate(im_array_list,start=1) :
#                msg = f'Adding image {self.n_images_read+1} to the stack'
#                if logger is not None :
#                    logger.imageinfo(msg,slide.name,slide.root_dir)
#                else :
#                    flatfield_logger.info(msg)
#                im_array=1.*im_array
#                if ets_for_normalization is not None :
#                    for li in range(self.nlayers) :
#                        im_array[:,:,li]/=ets_for_normalization[li]
#                self.image_stack+=im_array
#                self.image_squared_stack+=(im_array*im_array)
#                self.n_images_read+=1
#                self.n_images_stacked_by_layer+=1
#                stacked_in_layers.append(list(range(1,self.nlayers+1)))
#            return stacked_in_layers
#        #otherwise produce the image masks, apply them to the raw images, and be sure to add them to the list in the same order as the images
#        manager = mp.Manager()
#        return_dict = manager.dict()
#        procs = []
#        for i,(im_array,rfp) in enumerate(zip(im_array_list,rfps)) :
#            stack_i = i+self.n_images_read+1
#            msg = f'Masking and adding image {stack_i} to the stack'
#            if logger is not None :
#                logger.imageinfo(msg,slide.name,slide.root_dir)
#            else :
#                flatfield_logger.info(msg)
#            make_plots=i in masking_plot_indices
#            p = mp.Process(target=ImageMask.create_mp,
#                           args=(im_array,rfp,slide.rawfile_top_dir,slide.background_thresholds_for_masking,
#                                 ets_for_normalization,slide.exp_time_hists,make_plots,self.masking_plot_dirpath,stack_i,return_dict))
#            procs.append(p)
#            p.start()
#        for proc in procs:
#            proc.join()
#        chunk_labelled_mask_regions = []
#        for stack_i,im_array in enumerate(im_array_list,start=self.n_images_read+1) :
#            stacked_in_layers.append([])
#            this_image_mask_obj = return_dict[stack_i]
#            onehot_mask = this_image_mask_obj.onehot_mask
#            chunk_labelled_mask_regions+=this_image_mask_obj.labelled_mask_regions
#            #check, layer-by-layer, that this mask would select at least the minimum amount of pixels to be added to the stack
#            for li in range(self.nlayers) :
#                thismasklayer = onehot_mask[:,:,li]
#                if 1.*np.sum(thismasklayer)/(self._dims[0]*self._dims[1])>=min_pixel_frac :
#                    #Optionally normalize for exposure time, and add to the stack
#                    im_to_add = 1.*im_array[:,:,li]*thismasklayer
#                    if ets_for_normalization is not None :
#                        im_to_add/=ets_for_normalization[li]
#                    self.image_stack[:,:,li]+=im_to_add
#                    self.image_squared_stack[:,:,li]+=(im_to_add*im_to_add)
#                    self.mask_stack[:,:,li]+=thismasklayer
#                    self.n_images_stacked_by_layer[li]+=1
#                    stacked_in_layers[-1].append(li+1)
#            self.n_images_read+=1
#        self._labelled_mask_regions+=chunk_labelled_mask_regions
#        return stacked_in_layers
#
#    def makeMeanImage(self,logger=None) :
#        """
#        A function to get the mean of the image stack
#        logger = a RunLogger object whose context is entered, if None the default log will be used
#        """
#        for li,nlis in enumerate(self.n_images_stacked_by_layer,start=1) :
#            if nlis<1 :
#                msg = f'WARNING: {nlis} images were stacked in layer {li}, so this layer of the meanimage will be meaningless!'
#                if logger is not None :
#                    logger.warningglobal(msg)
#                else :
#                    flatfield_logger.warn(msg)
#        self.mean_image, self.std_err_of_mean_image = self.__getMeanImage(logger)
#
#    def makeFlatFieldImage(self,logger=None) :
#        """
#        A function to smooth/normalize each of the mean image layers to make the flatfield image
#        logger = a RunLogger object whose context is entered, if None the default log will be used
#        """
#        if self.n_images_read<1 :
#            raise FlatFieldError('ERROR: not enough images were read to produce a flatfield!')
#        for li,nlis in enumerate(self.n_images_stacked_by_layer,start=1) :
#            if nlis<1 :
#                msg = f'WARNING: {nlis} images were stacked in layer {li}, so this layer of the flatfield will be meaningless!'
#                if logger is not None :
#                    logger.warningglobal(msg)
#                else :
#                    flatfield_logger.warn(msg)
#        if self.mean_image is None :
#            self.mean_image, self.std_err_of_mean_image = self.__getMeanImage(logger)
#        self.smoothed_mean_image,sm_mean_img_err = smoothImageWithUncertaintyWorker(self.mean_image,self.std_err_of_mean_image,self.final_smooth_sigma)
#        self.flatfield_image = np.empty_like(self.smoothed_mean_image)
#        for layer_i in range(self.nlayers) :
#            if np.min(self.smoothed_mean_image[:,:,layer_i])==0 and np.max(self.smoothed_mean_image[:,:,layer_i])==0 :
#                self.flatfield_image[:,:,layer_i]=1.0
#            else :
#                weights = np.divide(np.ones_like(sm_mean_img_err[:,:,layer_i]),(sm_mean_img_err[:,:,layer_i])**2,
#                                    out=np.zeros_like(sm_mean_img_err[:,:,layer_i]),where=sm_mean_img_err[:,:,layer_i]>0.)
#                if np.sum(weights)<=0 :
#                    msg = f'WARNING: sum of weights in layer {layer_i+1} is {np.sum(weights)}, this layer of the flatfield is all ones!'
#                    if logger is not None :
#                        logger.warningglobal(msg)
#                    else :
#                        flatfield_logger.warn(msg)    
#                    self.flatfield_image[:,:,layer_i]=1.0
#                else :
#                    layermean = np.average(self.smoothed_mean_image[:,:,layer_i],weights=weights)
#                    self.flatfield_image[:,:,layer_i]=self.smoothed_mean_image[:,:,layer_i]/layermean
#
#    def makeCorrectedMeanImage(self,flatfield_file_path) :
#        """
#        A function to get the mean of the image stack, smooth it, and divide it by the given flatfield to correct it
#        """
#        if self.mean_image is None :
#            self.mean_image, self.std_err_of_mean_image = self.__getMeanImage()
#        self.smoothed_mean_image = smoothImageWorker(self.mean_image,self.final_smooth_sigma)
#        flatfield_image=getRawAsHWL(flatfield_file_path,*(self._dims),dtype=UNIV_CONST.FLATFIELD_IMAGE_DTYPE)
#        self.corrected_mean_image=self.mean_image/flatfield_image
#        self.smoothed_corrected_mean_image=smoothImageWorker(self.corrected_mean_image,self.final_smooth_sigma)
#        with cd(self._workingdir_path) :
#            with open(self.APPLIED_FLATFIELD_TEXT_FILE_NAME,'w') as fp :
#                fp.write(f'{flatfield_file_path}\n')
#
#    def saveImages(self,batch_or_slide_ID=None) :
#        """
#        Save the various images that are created
#        batch_or_slide_ID = the integer batch ID or string slideID to append to the outputted file names
#        """
#        #figure out what to append to the filenames
#        prepend = ''; append = ''
#        if batch_or_slide_ID is not None :
#            if isinstance(batch_or_slide_ID,int) :
#                append = f'_BatchID_{batch_or_slide_ID:02d}'
#            else :
#                prepend = f'{batch_or_slide_ID}-'
#        #write out the files
#        with cd(self._workingdir_path) :
#            if self.flatfield_image is not None :
#                flatfieldimage_filename = f'{prepend}{CONST.FLATFIELD_FILE_NAME_STEM}{append}{CONST.FILE_EXT}'
#                writeImageToFile(self.flatfield_image,flatfieldimage_filename,dtype=UNIV_CONST.FLATFIELD_IMAGE_DTYPE)
#            if self.mean_image is not None :
#                meanimage_filename = f'{prepend}{CONST.MEAN_IMAGE_FILE_NAME_STEM}{append}{CONST.FILE_EXT}'
#                writeImageToFile(self.mean_image,meanimage_filename,dtype=UNIV_CONST.FLATFIELD_IMAGE_DTYPE)
#            if self.std_err_of_mean_image is not None :
#                std_err_of_meanimage_filename = f'{prepend}{CONST.STD_ERR_MEAN_IMAGE_FILE_NAME_STEM}{append}{CONST.FILE_EXT}'
#                writeImageToFile(self.std_err_of_mean_image,std_err_of_meanimage_filename,dtype=UNIV_CONST.FLATFIELD_IMAGE_DTYPE)
#            if append=='' :
#                if self.image_squared_stack is not None :
#                    sumimagessquared_filename = f'{prepend}{CONST.SUM_IMAGES_SQUARED_FILE_NAME_STEM}{append}{CONST.FILE_EXT}'
#                    writeImageToFile(self.image_squared_stack,sumimagessquared_filename,dtype=UNIV_CONST.FLATFIELD_IMAGE_DTYPE)
#                if (not self.skip_masking) and (self.mask_stack is not None) :
#                    writeImageToFile(self.mask_stack,f'{prepend}{CONST.MASK_STACK_FILE_NAME_STEM}{append}{CONST.FILE_EXT}',dtype=CONST.MASK_STACK_DTYPE_OUT)
#                if self.smoothed_mean_image is not None :
#                    smoothed_meanimage_filename = f'{prepend}{self.SMOOTHED_MEAN_IMAGE_FILE_NAME_STEM}{append}{CONST.FILE_EXT}'
#                    writeImageToFile(self.smoothed_mean_image,smoothed_meanimage_filename,dtype=UNIV_CONST.FLATFIELD_IMAGE_DTYPE)
#                if self.corrected_mean_image is not None :
#                    corrected_mean_image_filename = f'{prepend}{self.CORRECTED_MEAN_IMAGE_FILE_NAME_STEM}{append}{CONST.FILE_EXT}'
#                    writeImageToFile(self.corrected_mean_image,corrected_mean_image_filename,dtype=UNIV_CONST.FLATFIELD_IMAGE_DTYPE)
#                if self.smoothed_corrected_mean_image is not None :
#                    smoothed_corrected_mean_image_filename = f'{prepend}{CONST.SMOOTHED_CORRECTED_MEAN_IMAGE_FILE_NAME_STEM}{append}{CONST.FILE_EXT}'
#                    writeImageToFile(self.smoothed_corrected_mean_image,smoothed_corrected_mean_image_filename,dtype=UNIV_CONST.FLATFIELD_IMAGE_DTYPE)
#
#    def savePlots(self) :
#        """
#        Make and save several visualizations of the image layers and details of the masking and flatfielding
#        """
#        with cd(self._workingdir_path) :
#            #make the plot directory if its not already created
#            if not pathlib.Path.is_dir(pathlib.Path(CONST.POSTRUN_PLOT_DIRECTORY_NAME)) :
#                pathlib.Path.mkdir(pathlib.Path(CONST.POSTRUN_PLOT_DIRECTORY_NAME))
#            with cd(CONST.POSTRUN_PLOT_DIRECTORY_NAME) :
#                #.pngs of the mean image, flatfield, and mask stack layers
#                self.__saveImageLayerPlots()
#                #plots of the minimum and and maximum (and 5/95%ile) pixel intensities
#                if self.flatfield_image is not None :
#                    #for the flatfield image
#                    flatfieldImagePixelIntensityPlot(self.flatfield_image,CONST.PIXEL_INTENSITY_PLOT_NAME)
#                if self.smoothed_mean_image is not None and self.smoothed_corrected_mean_image is not None :
#                    #for the corrected mean image
#                    correctedMeanImagePIandIVplots(self.smoothed_mean_image,self.smoothed_corrected_mean_image,
#                                                   pi_savename=self.CORRECTED_MEAN_IMAGE_PIXEL_INTENSITY_PLOT_NAME,
#                                                   iv_plot_name=self.ILLUMINATION_VARIATION_PLOT_NAME,
#                                                   iv_csv_name=self.ILLUMINATION_VARIATION_CSV_FILE_NAME)
#                #plot and write a text file of how many images were stacked per layer
#                self.__plotAndWriteNImagesStackedPerLayer()
#        #save the table of labelled mask regions (if applicable)
#        if len(self._labelled_mask_regions)>0 :
#            with cd(pathlib.Path(f'{self._workingdir_path}/{self.MASKING_SUBDIR_NAME}')) :
#                writetable(CONST.LABELLED_MASK_REGIONS_CSV_FILE_NAME,self._labelled_mask_regions)
#
#    def addLMRs(self,lmrs_to_add) :
#        """
#        add a list of Labelled Mask Region objects to this mean image's list
#        """
#        self._labelled_mask_regions+=lmrs_to_add
#
#    #################### PRIVATE HELPER FUNCTIONS ####################
#
#    #helper function to create and return the mean image from the image (and mask, if applicable, stack)
#    def __getMeanImage(self,logger=None) :
#        #make sure there actually was at least some number of images used
#        if self.n_images_read<1 :
#            msg = f'WARNING: {self.n_images_read} images were read in and so the mean image will be zero everywhere!'
#            if logger is not None :
#                logger.warningglobal(msg)
#            else :
#                flatfield_logger.warn(msg)
#            return self.image_stack, self.image_squared_stack
#        if np.max(self.n_images_stacked_by_layer)<1 :
#            msg = 'WARNING: There are no layers with images stacked in them and so the mean image will be zero everywhere!'
#            if logger is not None :
#                logger.warningglobal(msg)
#            else :
#                flatfield_logger.warn(msg)
#            return self.image_stack, self.image_squared_stack
#        #if the images haven't been masked then this is trivial
#        if self.skip_masking :
#            meanimage = self.image_stack/self.n_images_read
#            std_err_of_meanimage = np.sqrt(np.abs(self.image_squared_stack/self.n_images_read-(meanimage*meanimage))/self.n_images_read)
#            return meanimage, std_err_of_meanimage
#        #otherwise though we have to be a bit careful and take the mean value pixel-wise, 
#        #being careful to fix any pixels that never got added to so there's no division by zero
#        zero_fixed_mask_stack = np.copy(self.mask_stack)
#        zero_fixed_mask_stack[zero_fixed_mask_stack==0] = np.min(zero_fixed_mask_stack[zero_fixed_mask_stack!=0])
#        meanimage = self.image_stack/zero_fixed_mask_stack
#        std_err_of_meanimage = np.sqrt(np.abs(self.image_squared_stack/zero_fixed_mask_stack-(meanimage*meanimage))/zero_fixed_mask_stack)
#        return meanimage, std_err_of_meanimage
#
#    #################### VISUALIZATION HELPER FUNCTIONS ####################
#
#    #helper function to make and save the .png images of the mean image, flatfield, and mask stack layers
#    def __saveImageLayerPlots(self) :
#        if self.mean_image is None :
#            self.makeMeanImage()
#        if not pathlib.Path.is_dir(pathlib.Path(CONST.IMAGE_LAYER_PLOT_DIRECTORY_NAME)) :
#            pathlib.Path.mkdir(pathlib.Path(CONST.IMAGE_LAYER_PLOT_DIRECTORY_NAME))
#        with cd(CONST.IMAGE_LAYER_PLOT_DIRECTORY_NAME) :
#            #figure out the size of the figures to save
#            fig_size=(self.IMG_LAYER_FIG_WIDTH,self.IMG_LAYER_FIG_WIDTH*(self.mean_image.shape[0]/self.mean_image.shape[1]))
#            #iterate over the layers
#            for layer_i in range(self.nlayers) :
#                layer_titlestem = f'layer {layer_i+1}'
#                layer_fnstem = f'layer_{layer_i+1}'
#                #save a little figure of each layer in each image
#                #for the mean image
#                f,ax = plt.subplots(figsize=fig_size)
#                pos = ax.imshow(self.mean_image[:,:,layer_i])
#                ax.set_title(f'mean image, {layer_titlestem}')
#                f.colorbar(pos,ax=ax)
#                fig_name = f'mean_image_{layer_fnstem}.png'
#                plt.savefig(fig_name); plt.close(); cropAndOverwriteImage(fig_name)
#                if self.std_err_of_mean_image is not None :
#                    #for the uncertainty on the mean image
#                    f,ax = plt.subplots(figsize=fig_size)
#                    pos = ax.imshow(self.std_err_of_mean_image[:,:,layer_i])
#                    ax.set_title(f'std. error of mean image, {layer_titlestem}')
#                    f.colorbar(pos,ax=ax)
#                    fig_name = f'std_err_of_mean_image_{layer_fnstem}.png'
#                    plt.savefig(fig_name); plt.close(); cropAndOverwriteImage(fig_name)
#                if self.smoothed_mean_image is not None :
#                    #for the smoothed mean image
#                    f,ax = plt.subplots(figsize=fig_size)
#                    pos = ax.imshow(self.smoothed_mean_image[:,:,layer_i])
#                    ax.set_title(f'smoothed mean image, {layer_titlestem}')
#                    f.colorbar(pos,ax=ax)
#                    fig_name = f'smoothed_mean_image_{layer_fnstem}.png'
#                    plt.savefig(fig_name); plt.close(); cropAndOverwriteImage(fig_name)
#                if self.flatfield_image is not None :
#                    #for the flatfield image
#                    f,ax = plt.subplots(figsize=fig_size)
#                    pos = ax.imshow(self.flatfield_image[:,:,layer_i])
#                    ax.set_title(f'flatfield, {layer_titlestem}')
#                    f.colorbar(pos,ax=ax)
#                    fig_name = f'flatfield_{layer_fnstem}.png'
#                    plt.savefig(fig_name); plt.close(); cropAndOverwriteImage(fig_name)
#                if self.corrected_mean_image is not None :
#                    #for the corrected mean image
#                    f,ax = plt.subplots(figsize=fig_size)
#                    pos = ax.imshow(self.corrected_mean_image[:,:,layer_i])
#                    ax.set_title(f'corrected mean image, {layer_titlestem}')
#                    f.colorbar(pos,ax=ax)
#                    fig_name = f'corrected_mean_image_{layer_fnstem}.png'
#                    plt.savefig(fig_name); plt.close(); cropAndOverwriteImage(fig_name)
#                if self.smoothed_corrected_mean_image is not None :
#                    #for the corrected mean image
#                    f,ax = plt.subplots(figsize=fig_size)
#                    pos = ax.imshow(self.smoothed_corrected_mean_image[:,:,layer_i])
#                    ax.set_title(f'smoothed corrected mean image, {layer_titlestem}')
#                    f.colorbar(pos,ax=ax)
#                    fig_name = f'smoothed_corrected_mean_image_{layer_fnstem}.png'
#                    plt.savefig(fig_name); plt.close(); cropAndOverwriteImage(fig_name)
#                if (not self.skip_masking) and (self.mask_stack is not None) :
#                    #for the mask stack
#                    f,ax = plt.subplots(figsize=fig_size)
#                    pos = ax.imshow(self.mask_stack[:,:,layer_i])
#                    ax.set_title(f'stacked binary image masks, {layer_titlestem}')
#                    f.colorbar(pos,ax=ax)
#                    fig_name = f'mask_stack_{layer_fnstem}.png'
#                    plt.savefig(fig_name); plt.close(); cropAndOverwriteImage(fig_name)
#
#    #helper function to plot and save how many images ended up being stacked in each layer of the meanimage
#    def __plotAndWriteNImagesStackedPerLayer(self) :
#        xvals = list(range(1,self.nlayers+1))
#        f,ax = plt.subplots(figsize=(9.6,4.8))
#        ax.plot([xvals[0],xvals[-1]],[self.n_images_read,self.n_images_read],linewidth=2,color='k',label=f'total images read ({self.n_images_read})')
#        ax.plot(xvals,self.n_images_stacked_by_layer,marker='o',linewidth=2,label='n images stacked')
#        #for pi,(xv,yv) in enumerate(zip(xvals,self.n_images_stacked_by_layer)) :
#        #    plt.annotate(f'{yv:d}',
#        #                 (xv,yv),
#        #                 textcoords="offset points", # how to position the text
#        #                 xytext=(0,10) if pi%2==0 else (0,-12), # distance from text to points (x,y)
#        #                 ha='center') # horizontal alignment can be left, right or center 
#        ax.set_title('Number of images selected to be stacked by layer')
#        ax.set_xlabel('image layer')
#        ax.set_ylabel('number of images')
#        ax.legend(loc='best')
#        plt.savefig(CONST.N_IMAGES_STACKED_PER_LAYER_PLOT_NAME)
#        plt.close()
#        cropAndOverwriteImage(CONST.N_IMAGES_STACKED_PER_LAYER_PLOT_NAME)
#        with open(self.N_IMAGES_STACKED_PER_LAYER_TEXT_FILE_NAME,'w') as fp :
#            for li in range(self.nlayers) :
#                fp.write(f'{self.n_images_stacked_by_layer[li]}\n')
#        with open(CONST.N_IMAGES_READ_TEXT_FILE_NAME,'w') as fp :
#            fp.write(f'{self.n_images_read}\n')
