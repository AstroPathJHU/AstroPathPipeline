#imports
import pathlib, cv2
import numpy as np
from ...utilities.miscfileio import cd
from ...utilities.img_file_io import smooth_image_worker, im3writeraw, write_image_to_file, get_raw_as_hwl
from .config import CONST
from .plotting import do_masking_plots_for_image
from .utilities import LabelledMaskRegion, get_enumerated_mask, get_size_filtered_mask
from .utilities import get_morphed_and_filtered_mask, get_exclusive_mask
from .utilities import get_image_layer_local_variance_of_normalized_laplacian

class ImageMask() :
    """
    Class to store and work with a mask for an image
    """

    #################### PROPERTIES ####################

    @property
    def packed_tissue_mask(self) : #the binary tissue mask packed using np.packbits
        if self.__tissue_mask is None :
            raise RuntimeError('ERROR: pasked_tissue_mask called before setting a tissue mask')
        return np.packbits(self.__tissue_mask)
    @property
    def compressed_mask(self): #the compressed mask with (# layers groups)+1 layers
        if self.__compressed_mask is None :
            raise RuntimeError('ERROR: compressed_mask called without first creating a mask!')
        return self.__compressed_mask
    @property
    def onehot_mask(self) : #the mask with good tissue=1, everything else=0
        return (np.where(self.uncompressed_full_mask==1,1,0)).astype(np.uint8)
    @property
    def uncompressed_full_mask(self): #the uncompressed mask with the real number of image layers
        if self.__compressed_mask is None :
            raise RuntimeError('ERROR: uncompressed_full_mask called without first creating a mask!')
        nlayers = 0
        for lgb in self.__layer_groups.values() :
            if lgb[1]>nlayers :
                nlayers=lgb[1]
        uncompressed_mask = np.ones((*self.__compressed_mask.shape[:-1],nlayers),dtype=np.uint8)
        for lgi,lgb in enumerate(self.__layer_groups.values()) :
            for ln in range(lgb[0],lgb[1]+1) :
                uncompressed_mask[:,:,ln-1] = self.__compressed_mask[:,:,lgi+1]
        return uncompressed_mask
    @property
    def labelled_mask_regions(self):
        return self._labelled_mask_regions #the list of labelled mask region objects for this mask

    #################### PUBLIC FUNCTIONS ####################

    def __init__(self,sample,im_array,im_key,bg_thresholds,norm_ets) :
        """
        sample            = the Sample object that this image array is coming from
        im_array          = the multilayer image array whose mask should be created 
                            (may already be corrected to a set of exposure times)
        im_key            = the string representing the key of the image filename 
                            (used as a prepend to the masking file name and in labelled mask regions)
        bg_thresholds     = a list of the background intensity thresholds in counts in each image layer
        norm_ets          = a list of the exposure times to which the image layers have been normalized 

        The last three arguments are only needed (and the last two are required) if plots for this image will be saved
        """
        #set the layer groups for the image
        self.__layer_groups=sample.layer_groups
        #set the number of layers in each group that can be missed and still have the region flagged
        self.__fold_flag_cuts = {}
        for lgn,lgb in self.__layer_groups.items() :
            nlayers = lgb[1]-lgb[0]+1
            if nlayers<6 :
                self.__fold_flag_cuts[lgn] = 0
            elif nlayers<9 :
                self.__fold_flag_cuts[lgn] = 1
            else :
                self.__fold_flag_cuts[lgn] = 3
        #set which layers are the brightest
        self.__bright_layers=sample.brightest_layers
        #apply smoothing to Vectra images only
        microscope_name = ((list(self.__layer_groups.keys())[0]).split('_'))[0]
        if microscope_name=='vectra' :
            self.__blur_mask_sm_img_array = smooth_image_worker(im_array,CONST.BLUR_MASK_SMOOTHING_SIGMA)
        elif microscope_name=='polaris' :
            self.__blur_mask_sm_img_array = im_array
        else :
            raise ValueError(f'ERROR: unrecognized microscope name "{microscope_name}"!')
        #set some other variables
        self.__im_array = im_array
        self.__image_key = im_key
        self.__bg_thresholds = np.array(bg_thresholds)
        self.__norm_ets = np.array(norm_ets)
        #create the tissue mask
        self.__tissue_mask = self.__get_image_tissue_mask()
        #create the blur mask (possibly with some plots)
        self.__blur_mask = self.__get_image_blur_mask()
        #create the saturation masks (one for each layer group)
        self.__saturation_masks = self.__get_image_saturation_masks()
        #make the compressed mask and the list of labelled mask regions
        self.__make_compressed_mask_and_list_of_mask_regions()

    def save_mask_files(self,savedir) :
        """
        Write out the actual mask files to the path given by savedir
        """
        #if there is anything flagged in the final blur and saturation masks, write out the compressed mask
        is_masked = np.min(self.__blur_mask)<1
        if not is_masked :
            for lgsm in self.__saturation_masks.values() :
                if np.min(lgsm)<1 :
                    is_masked=True
                    break
        if not savedir.is_dir() :
            savedir.mkdir()
        with cd(savedir) :
            im3writeraw(f'{self.__image_key}_tissue_mask.bin',self.packed_tissue_mask)
            if is_masked :
                write_image_to_file(self.__compressed_mask,f'{self.__image_key}_full_mask.bin',dtype=np.uint8)

    def save_plots(self,orig_ets,et_hists_and_bins,savedir=None) :
        """
        Make and save the sheet of plots for this image if requested

        orig_ets          = a list of the given image's ORIGINAL exposure times in each layer 
                            (before any corrections were applied)  
        et_hists_and_bins = a dict of exposure time histograms and their bins in each layer group. 
        savedir           = path to the directory in which the plot(s) should be saved 
                            (if None the plots will be written in the current directory)
        """
        all_plots = []
        for lgn,lgb in self.__layer_groups.items() :
            fold_nlv_cut = CONST.FOLD_NLV_CUTS[lgn]
            fold_nlv_max_mean = CONST.FOLD_MAX_MEANS[lgn]
            fold_flag_cut = self.__fold_flag_cuts[lgn]
            group_blur_mask,stacked_masks = self.__get_image_layer_group_blur_mask(lgn,fold_nlv_cut,
                                                                                   fold_nlv_max_mean,fold_flag_cut)
            plot_img_layer = self.__im_array[:,:,self.__bright_layers[lgn]]
            sorted_pil = np.sort(plot_img_layer[group_blur_mask==1].flatten())
            if len(sorted_pil)>0 :
                pil_max = sorted_pil[int(0.95*len(sorted_pil))]; pil_min = sorted_pil[0]
            else :
                pil_max = np.max(plot_img_layer); pil_min = np.min(plot_img_layer)
            norm = 255./(pil_max-pil_min)
            im_c = (np.clip(norm*(plot_img_layer-pil_min),0,255)).astype(np.uint8)
            overlay_c = np.array([im_c,im_c*group_blur_mask,im_c*group_blur_mask]).transpose(1,2,0)
            plots = [{'image':plot_img_layer,'title':f'raw IMAGE layer {self.__bright_layers[lgn]}'},
                     {'image':overlay_c,'title':f'layer {lgb[0]}-{lgb[1]} blur mask overlay (clipped)'}]
            plots.append({'bar':et_hists_and_bins[lgn][0],
                          'bins':et_hists_and_bins[lgn][1],
                          'xlabel':f'layer {lgb[0]}-{lgb[1]} exposure times (ms)',
                          'line_at':orig_ets[lgb[0]-1]})
            plots.append({'image':self.__im_nlv[:,:,self.__bright_layers[lgn]],
                          'title':'local variance of normalized laplacian'})
            plots.append({'hist':self.__im_nlv[:,:,self.__bright_layers[lgn]].flatten(),
                          'xlabel':'variance of normalized laplacian',
                          'line_at':fold_nlv_cut})
            plots.append({'image':stacked_masks,
                          'title':f'stacked layer masks (cut at {fold_flag_cut})',
                          'cmap':'gist_ncar',
                          'vmin':0,'vmax':lgb[1]-lgb[0]+1})
            all_plots.append(plots)
        do_masking_plots_for_image(self.__image_key,self.__tissue_mask,all_plots,self.__compressed_mask,savedir)

    #unpack, reshape, and return a tissue mask from its packed mask file
    @staticmethod
    def unpack_tissue_mask(filepath,dimensions) :
        if not pathlib.Path(filepath).is_file() :
            raise FileNotFoundError(f'ERROR: tissue mask file {filepath} does not exist!')
        packed_mask = np.memmap(filepath,dtype=np.uint8,mode='r')
        return (np.unpackbits(packed_mask)).reshape(dimensions)

    #get a one-hot, fully-layered mask from a given blur/saturation mask filepath
    @staticmethod
    def onehot_mask_from_full_mask_file(samp,filepath,dimensions) :
        if not pathlib.Path(filepath).is_file() :
            raise FileNotFoundError(f'ERROR: blur/saturation mask file {filepath} does not exist!')
        read_mask = get_raw_as_hwl(filepath,*(dimensions[:-1]),len(samp.layer_groups)+1,dtype=np.uint8)
        return_mask = np.zeros(dimensions,dtype=np.uint8)
        for lgi,lgb in enumerate(samp.layer_groups.values()) :
            return_mask[:,:,lgb[0]-1:lgb[1]][read_mask[:,:,lgi+1]==1] = 1
        return return_mask

    #get a one-hot, fully-layered mask from a given blur/saturation mask filepath, ignoring regions flagged for blur
    @staticmethod
    def onehot_mask_from_full_mask_file_no_blur(samp,filepath,dimensions) :
        if not pathlib.Path(filepath).is_file() :
            raise FileNotFoundError(f'ERROR: blur/saturation mask file {filepath} does not exist!')
        read_mask = get_raw_as_hwl(filepath,*(dimensions[:-1]),len(samp.layer_groups)+1,dtype=np.uint8)
        max_blur_index = np.max(read_mask[:,:,0])
        return_mask = np.zeros(dimensions,dtype=np.uint8)
        for lgi,lgb in enumerate(samp.layer_groups.values()) :
            tissue_or_blur_slice = np.logical_or(read_mask[:,:,lgi+1]==1,
                                                 np.logical_and(read_mask[:,:,lgi+1]>1,
                                                                read_mask[:,:,lgi+1]<=max_blur_index))
            return_mask[:,:,lgb[0]-1:lgb[1]][tissue_or_blur_slice] = 1
        return return_mask

    #################### PRIVATE HELPER FUNCTIONS ####################

    def __get_image_tissue_mask(self) :
        """
        return the fully-determined overall tissue mask as a 2d array of ones and zeroes for a given multilayer image
        """
        #smooth the image
        sm_img_array = smooth_image_worker(self.__im_array,CONST.TISSUE_MASK_SMOOTHING_SIGMA)
        #threshold all the image layers
        thresholded_image = (np.where(sm_img_array>self.__bg_thresholds[np.newaxis,np.newaxis,:],1,0)).astype(np.uint8)
        #make masks for each individual layer
        layer_masks = []
        for li in range(self.__im_array.shape[-1]) :
            #convert to UMat to use on the GPU
            layer_mask = cv2.UMat(thresholded_image[:,:,li])
            #small close/open
            cv2.morphologyEx(layer_mask,cv2.MORPH_CLOSE,CONST.SMALL_CO_EL,layer_mask,borderType=cv2.BORDER_REPLICATE)
            cv2.morphologyEx(layer_mask,cv2.MORPH_OPEN,CONST.SMALL_CO_EL,layer_mask,borderType=cv2.BORDER_REPLICATE)
            layer_masks.append(layer_mask.get())
        #find the well-defined tissue and background in each layer group
        overall_tissue_mask = np.zeros_like(layer_masks[0])
        overall_background_mask = np.zeros_like(layer_masks[0])
        total_stacked_masks = np.zeros_like(layer_masks[0])
        #for each layer group
        for lgn,lgb in self.__layer_groups.items() :
            stacked_masks = np.zeros_like(layer_masks[0])
            for ln in range(lgb[0],lgb[1]+1) :
                stacked_masks+=layer_masks[ln-1]
            total_stacked_masks+=stacked_masks
            #well-defined tissue is anything called tissue in at least all but two layers
            overall_tissue_mask[stacked_masks>(lgb[1]-lgb[0]-1)]+=10 if lgn.endswith('dapi') else 1
            #well-defined background is anything called background in at least half the layers
            overall_background_mask[stacked_masks<(lgb[1]-lgb[0]+1)/2.]+=10 if lgn.endswith('dapi') else 1
        #threshold tissue/background masks to include only those from the DAPI and at least one other layer group
        overall_tissue_mask = (np.where(overall_tissue_mask>10,1,0)).astype(np.uint8)
        overall_background_mask = (np.where(overall_background_mask>10,1,0)).astype(np.uint8)
        #final mask has tissue=1, background=0
        final_mask = np.zeros_like(layer_masks[0])+2
        final_mask[overall_tissue_mask==1] = 1
        final_mask[overall_background_mask==1] = 0
        #anything left over is signal if it's stacked in at least 60% of the total number of layers
        thresholded_stacked_masks = np.where(total_stacked_masks>(0.6*self.__im_array.shape[-1]),1,0)
        final_mask[final_mask==2] = thresholded_stacked_masks[final_mask==2]
        if np.min(final_mask) != np.max(final_mask) :
            #filter the tissue and background portions to get rid of the small islands
            final_mask = get_size_filtered_mask(final_mask,min_size=CONST.TISSUE_MIN_SIZE)
            #convert to UMat
            final_mask = cv2.UMat(final_mask)
            #medium size close/open to smooth out edges
            cv2.morphologyEx(final_mask,cv2.MORPH_CLOSE,CONST.MEDIUM_CO_EL,final_mask,borderType=cv2.BORDER_REPLICATE)
            cv2.morphologyEx(final_mask,cv2.MORPH_OPEN,CONST.MEDIUM_CO_EL,final_mask,borderType=cv2.BORDER_REPLICATE)
            final_mask = final_mask.get()
        return final_mask

    def __get_image_blur_mask(self) :
        """
        return the single blur mask for an image (multilayer "tissue fold" blur and single-layer "dust" blur combined)
        """
        #first set the normalized laplacian variance of the image
        self.__im_nlv = np.zeros(self.__im_array.shape,dtype=np.float32)
        for li in range(self.__im_array.shape[-1]) :
            layer_nlv = get_image_layer_local_variance_of_normalized_laplacian(self.__blur_mask_sm_img_array[:,:,li])
            self.__im_nlv[:,:,li] = layer_nlv
        #then set the local mean of the normalized laplacian variance
        self.__im_nlv_loc_mean = np.zeros_like(self.__im_nlv)
        for li in range(self.__im_array.shape[-1]) :
            layer_nlv_loc_mean = cv2.UMat(np.empty_like(self.__im_nlv[:,:,li]))
            cv2.filter2D(self.__im_nlv[:,:,li],cv2.CV_32F,CONST.SMALLER_WINDOW_EL,layer_nlv_loc_mean,
                         borderType=cv2.BORDER_REFLECT)
            layer_nlv_loc_mean=layer_nlv_loc_mean.get()
            layer_nlv_loc_mean/=np.sum(CONST.SMALLER_WINDOW_EL)
            self.__im_nlv_loc_mean[:,:,li] = layer_nlv_loc_mean
        #find the tissue fold mask, beginning with each layer group separately
        fold_masks_by_layer_group = {}
        for lgn in self.__layer_groups.keys() :
            lgtfm,_ = self.__get_image_layer_group_blur_mask(lgn,
                                                             CONST.FOLD_NLV_CUTS[lgn],
                                                             CONST.FOLD_MAX_MEANS[lgn],
                                                             self.__fold_flag_cuts[lgn])
            fold_masks_by_layer_group[lgn] = lgtfm
        #combine the layer group blur masks to get the final mask for all layers
        stacked_fold_masks = np.zeros_like(fold_masks_by_layer_group[list(self.__layer_groups.keys())[0]])
        for lgn,layer_group_fold_mask in fold_masks_by_layer_group.items() :
            to_add = 10 if lgn.endswith('dapi') or lgn.endswith('fitc') else 1
            stacked_fold_masks[layer_group_fold_mask==0]+=to_add
        #flag anything flagged in at least one of the DAPI and FITC layer groups plus at least 3 other layer groups, 
        # or both the DAPI and FITC layer groups
        overall_fold_mask = (np.where(stacked_fold_masks>12,0,1)).astype(np.uint8)
        #morph and filter the mask using the common operations
        tissue_fold_mask = get_morphed_and_filtered_mask(overall_fold_mask,self.__tissue_mask,
                                                         CONST.FOLD_MIN_PIXELS,CONST.FOLD_MIN_SIZE)
        #get dust masks for the blurriest areas of the DAPI layer group
        dapi_group_name = None
        for lgn in self.__layer_groups.keys() :
            if lgn.endswith('dapi') :
                if dapi_group_name is not None :
                    errmsg = f'ERROR: more than one DAPI layer group name found! Layer groups: {self.__layer_groups}'
                    raise ValueError(errmsg)
                dapi_group_name = lgn
        if dapi_group_name is None :
            raise RuntimeError(f'ERROR: no DAPI layer group found! Layer groups: {self.__layer_groups}')
        dapi_layer_group = self.__layer_groups[dapi_group_name]
        n_layers_dust_flag_cut = 0.5*(dapi_layer_group[1]-dapi_layer_group[0]+1)
        dapi_dust_mask,_ = self.__get_image_layer_group_blur_mask(dapi_group_name,
                                                                  CONST.DUST_NLV_CUTS[dapi_group_name],
                                                                  CONST.DUST_MAX_MEANS[dapi_group_name],
                                                                  n_layers_dust_flag_cut)
        #same morphology transformations as for the multilayer blur masks
        morphed_dapi_dust_mask = get_morphed_and_filtered_mask(dapi_dust_mask,self.__tissue_mask,
                                                               CONST.DUST_MIN_PIXELS,CONST.DUST_MIN_SIZE)
        #make sure any regions in that mask are sufficiently exclusive w.r.t. what's already flagged as blurry
        exclusive_dapi_dust_mask = get_exclusive_mask(morphed_dapi_dust_mask,tissue_fold_mask,0.25)
        #combine the multilayer and single layer blur masks into one by multiplying them together
        final_blur_mask = tissue_fold_mask*exclusive_dapi_dust_mask
        #return the blur mask
        return final_blur_mask

    def __get_image_layer_group_blur_mask(self,layer_group_name,nlv_cut,max_mean,n_layers_flag_cut) :
        """
        return a blur mask for a given image layer group

        layer_group_name  = the name of the layer group whose blur mask should be returned
        nlv_cut           = the max value of the normalized laplacian variance below which 
                            a region should be flagged as blurred
        max_mean          = the maximum allowed mean of the normalized laplacian variance within 
                            any window-sized region to allow the region to be flagged
        n_layers_flag_cut = the number of layers in the group that can remain unflagged 
                            while still calling the region overall blurry in this layer group
        """
        #start by making a mask for every layer in the group
        stacked_masks = np.zeros(self.__im_array.shape[:-1],dtype=np.uint8)
        for ln in range(self.__layer_groups[layer_group_name][0],self.__layer_groups[layer_group_name][1]+1) :
            #get the local variance of the normalized laplacian image
            layer_nlv = self.__im_nlv[:,:,ln-1]
            #get the mean of those local normalized laplacian variance values in the window size
            layer_nlv_loc_mean = self.__im_nlv_loc_mean[:,:,ln-1]
            #threshold on the local variance of the normalized laplacian and its local mean to make a binary mask
            layer_mask = (np.where((layer_nlv>nlv_cut) | (layer_nlv_loc_mean>max_mean),1,0)).astype(np.uint8)
            if np.min(layer_mask) != np.max(layer_mask) :
                #convert to UMat
                layer_mask = cv2.UMat(layer_mask)
                #small open/close to refine it
                cv2.morphologyEx(layer_mask,cv2.MORPH_OPEN,CONST.SMALL_CO_EL,layer_mask,
                                 borderType=cv2.BORDER_REPLICATE)
                cv2.morphologyEx(layer_mask,cv2.MORPH_CLOSE,CONST.SMALL_CO_EL,layer_mask,
                                 borderType=cv2.BORDER_REPLICATE)
                #erode by the smaller window element
                cv2.morphologyEx(layer_mask,cv2.MORPH_ERODE,CONST.SMALLER_WINDOW_EL,layer_mask,
                                 borderType=cv2.BORDER_REPLICATE)
                layer_mask = layer_mask.get()
            #add it to the stack 
            stacked_masks+=layer_mask
        #determine the final mask for this group by thresholding on how many individual layers contribute
        group_blur_mask = (np.where(stacked_masks>n_layers_flag_cut,1,0)).astype(np.uint8)
        #return the blur mask and the stack of masks for the layer group
        return group_blur_mask, stacked_masks

    def __get_image_saturation_masks(self) :
        """
        Return the dict of saturation masks by layer group for a given image
        """
        #normalize the image by its exposure time
        normalized_image_arr = self.__im_array / self.__norm_ets[np.newaxis,np.newaxis,:]
        #smooth the normalized image image
        sm_n_image_arr = smooth_image_worker(normalized_image_arr,CONST.TISSUE_MASK_SMOOTHING_SIGMA)
        #make masks for each layer group
        layer_group_saturation_masks = {}
        for lgn,lgb in self.__layer_groups.items() :
            #threshold the image layers to make a binary mask and sum them
            cut_at = CONST.SATURATION_INTENSITY_CUTS[lgn]
            stacked_masks = np.sum((np.where(sm_n_image_arr[:,:,lgb[0]-1:lgb[1]]>cut_at,0,1)).astype(np.uint8),axis=2)
            #the final mask is anything flagged in ANY layer
            group_mask = (np.where(stacked_masks>lgb[1]-lgb[0],1,0)).astype(np.uint8) 
            if np.min(group_mask)!=np.max(group_mask) :
                #convert to UMat
                group_mask = cv2.UMat(group_mask)
                #medium sized open/close to refine it
                cv2.morphologyEx(group_mask,cv2.MORPH_OPEN,CONST.MEDIUM_CO_EL,group_mask,
                                 borderType=cv2.BORDER_REPLICATE)
                cv2.morphologyEx(group_mask,cv2.MORPH_CLOSE,CONST.MEDIUM_CO_EL,group_mask,
                                 borderType=cv2.BORDER_REPLICATE)
                group_mask = group_mask.get()
                #filter the mask for the total number of pixels and regions by the minimum size
                group_mask = get_size_filtered_mask(group_mask,CONST.SATURATION_MIN_SIZE)
            if np.sum(group_mask==0)<CONST.SATURATION_MIN_PIXELS :
                group_mask = np.ones_like(group_mask)
            layer_group_saturation_masks[lgn] = group_mask
        return layer_group_saturation_masks

    def __make_compressed_mask_and_list_of_mask_regions(self) :
        """
        combine all the created masks into a compressed mask and also make the list of labelled mask region objects
        """
        #start the list of labelled mask regions
        self._labelled_mask_regions = []
        #make the compressed mask, which has (# of layer groups)+1 layers total
        #the first layer holds just the tissue and blur masks; 
        #the other layers have the tissue and blur masks plus the saturation mask for each layer group
        self.__compressed_mask = np.ones((*self.__tissue_mask.shape,len(self.__layer_groups)+1),dtype=np.uint8)
        #add in the blur mask, starting with index 2
        start_i = 2
        if np.min(self.__blur_mask)<1 :
            layers_string = 'all'
            enumerated_blur_mask = get_enumerated_mask(self.__blur_mask,start_i)
            for li in range(self.__compressed_mask.shape[-1]) :
                self.__compressed_mask[:,:,li][enumerated_blur_mask!=0] = enumerated_blur_mask[enumerated_blur_mask!=0]
            start_i = np.max(enumerated_blur_mask)+1
            region_indices = list(range(np.min(enumerated_blur_mask[enumerated_blur_mask!=0]),
                                        np.max(enumerated_blur_mask)+1))
            for ri in region_indices :
                r_size = np.sum(enumerated_blur_mask==ri)
                self._labelled_mask_regions.append(LabelledMaskRegion(self.__image_key,ri,layers_string,
                                                                      r_size,CONST.BLUR_FLAG_STRING))
        #add in the saturation masks 
        for lgi,(lgn,lgsm) in enumerate(self.__saturation_masks.items()) :
            if np.min(lgsm)<1 :
                layers_string = f'{self.__layer_groups[lgn][0]}-{self.__layer_groups[lgn][1]}'
                enumerated_sat_mask = get_enumerated_mask(lgsm,start_i)
                self.__compressed_mask[:,:,lgi+1][enumerated_sat_mask!=0] = enumerated_sat_mask[enumerated_sat_mask!=0]
                start_i = np.max(enumerated_sat_mask)+1
                region_indices = list(range(np.min(enumerated_sat_mask[enumerated_sat_mask!=0]),
                                            np.max(enumerated_sat_mask)+1))
                for ri in region_indices :
                    r_size = np.sum(enumerated_sat_mask==ri)
                    self._labelled_mask_regions.append(LabelledMaskRegion(self.__image_key,ri,layers_string,
                                                                          r_size,CONST.SATURATION_FLAG_STRING))
        #finally add in the tissue mask
        for li in range(self.__compressed_mask.shape[-1]) :
            self.__compressed_mask[:,:,li][self.__compressed_mask[:,:,li]==1] = self.__tissue_mask[self.__compressed_mask[:,:,li]==1]

#################### FILE-SCOPE CONVENIENCE FUNCTIONS ####################

def return_new_mask_labelled_regions(sample,im_array,im_key,bg_thresholds,norm_ets,savedir=None) :
    """
    Create an ImageMask, write out the files it creates, and return its list of labelled mask regions
    This function writes out the ImageMask it creates and doesn't return it in order to have the 
    smallest possible memory footprint
    And it also doesn't depend on passing any complex objects outside of it
    Both of which are useful for running many iterations of the function in parallel processes
    
    arguments are the same as ImageMask.__init__
    """
    mask = ImageMask(sample,im_array,im_key,bg_thresholds,norm_ets)
    mask.save_mask_files(savedir)
    return mask.labelled_mask_regions

def save_plots_for_image(sample,im_array,im_key,bg_thresholds,norm_ets,orig_ets,exp_time_hists_and_bins,savedir) :
    """
    Create the masks for a given image and write out plots of the process
    Useful if all you care about is getting the plots
    This function also has the lowest possible memory footprint/simplest possible I/O to be run in parallel processes

    arguments are the same as ImageMask.__init__ + ImageMask.save_plots
    """
    mask = ImageMask(sample,im_array,im_key,bg_thresholds,norm_ets)
    mask.save_plots(orig_ets,exp_time_hists_and_bins,savedir)
