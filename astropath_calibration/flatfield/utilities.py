from .config import CONST
from ..utilities.dataclasses import MyDataClass
from ..utilities.img_file_io import getRawAsHWL, smoothImageWorker
from ..utilities.img_correction import correctImageForExposureTime
from ..utilities.config import CONST as UNIV_CONST
from concurrent.futures import ThreadPoolExecutor
from typing import List
import numpy as np
import cv2, os, logging, math, more_itertools

#################### GENERAL USEFUL OBJECTS ####################

#Class for errors encountered during flatfielding
class FlatFieldError(Exception) :
    pass

#logger
flatfield_logger = logging.getLogger("flatfield")
flatfield_logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s  [%(funcName)s]","%Y-%m-%d %H:%M:%S"))
flatfield_logger.addHandler(handler)

#helper class for logging included/excluded fields
class FieldLog(MyDataClass) :
    slide   : str
    file     : str
    location : str
    use      : str
    stacked_in_layers : str = ''

#helper class for inputting slides with their names and raw/root directories
class FlatfieldSlideInfo(MyDataClass) :
    name : str
    rawfile_top_dir : str
    root_dir : str

#mask region information helper class
class LabelledMaskRegion(MyDataClass) :
    image_key          : str
    region_index       : int
    layers             : str
    n_pixels           : int
    reason_flagged     : str

#################### GENERAL HELPER FUNCTIONS ####################

#helper function to return the slide name from a whole filepath
def slideNameFromFilepath(fp) :
    return os.path.basename(os.path.dirname(os.path.normpath(fp)))

#helper function to make the automatic directory path for a single slide's mean image (and associated info)
def getSlideMeanImageWorkingDirPath(slide) :
    #path = os.path.join(os.path.abspath(os.getcwd()),CONST.AUTOMATIC_MEANIMAGE_DIRNAME)
    path = os.path.join(os.path.abspath(os.path.normpath(slide.root_dir)),slide.name,'im3',CONST.AUTOMATIC_MEANIMAGE_DIRNAME)
    if not os.path.isdir(os.path.dirname(path)) :
        raise FlatFieldError(f'ERROR: working directory location {os.path.dirname(path)} does not exist!')
    if not os.path.isdir(path) :
        os.mkdir(path)
    return path

#helper function to make the automatic directory path for running the flatfield for a batch of slides
def getBatchFlatfieldWorkingDirPath(rootdir,batchID) :
    #path = os.path.join(os.path.abspath(os.getcwd()),f'{CONST.BATCH_FF_DIRNAME_STEM}_{batchID:02d}')
    path = os.path.join(os.path.abspath(os.path.normpath(rootdir)),'Flatfield',f'{CONST.BATCH_FF_DIRNAME_STEM}_{batchID:02d}')
    if not os.path.isdir(os.path.dirname(path)) :
        raise FlatFieldError(f'ERROR: working directory location {os.path.dirname(path)} does not exist!')
    if not os.path.isdir(path) :
        os.mkdir(path)
    return path

#helper function to return the automatic path to a given slide's mean image file
def getSlideMeanImageFilepath(slide) :
    p = os.path.join(slide.root_dir,slide.name,'im3',CONST.AUTOMATIC_MEANIMAGE_DIRNAME,f'{slide.name}-{CONST.MEAN_IMAGE_FILE_NAME_STEM}{CONST.FILE_EXT}')
    return p

#helper function to return the automatic path to a given slide's sum of images squared file
def getSlideImageSquaredFilepath(slide)
    p = os.path.join(slide.root_dir,slide.name,'im3',CONST.AUTOMATIC_MEANIMAGE_DIRNAME,f'{slide.name}-{CONST.SUM_IMAGES_SQUARED_FILE_NAME_STEM}{CONST.FILE_EXT}')
    return p

#helper function to return the automatic path to a given slide's standard error of the mean image file
def getSlideStdErrMeanImageFilepath(slide) :
    p = os.path.join(slide.root_dir,slide.name,'im3',CONST.AUTOMATIC_MEANIMAGE_DIRNAME,f'{slide.name}-{CONST.STD_ERR_MEAN_IMAGE_FILE_NAME_STEM}{CONST.FILE_EXT}')
    return p

#helper function to return the automatic path to a given slide's mean image file
def getSlideMaskStackFilepath(slide) :
    p = os.path.join(slide.root_dir,slide.name,'im3',CONST.AUTOMATIC_MEANIMAGE_DIRNAME,f'{slide.name}-{CONST.MASK_STACK_FILE_NAME_STEM}{CONST.FILE_EXT}')
    return p

#helper function to convert an image array into a flattened pixel histogram
def getImageArrayLayerHistograms(img_array, mask=slice(None)) :
    nbins = np.iinfo(img_array.dtype).max+1
    nlayers = img_array.shape[2] if len(img_array.shape)>2 else 1
    if nlayers>1 :
        layer_hists = np.empty((nbins,nlayers),dtype=np.int64)
        for li in range(nlayers) :
            layer_hist,_ = np.histogram(img_array[:,:,li][mask],nbins,(0,nbins))
            layer_hists[:,li]=layer_hist
        return layer_hists
    else :
        layer_hist,_ = np.histogram(img_array[mask],nbins,(0,nbins))
        return layer_hist

#################### THRESHOLDING HELPER FUNCTIONS ####################

#helper function to determine the Otsu threshold given a histogram of pixel values 
#algorithm from python code at https://docs.opencv.org/3.4/d7/d4d/tutorial_py_thresholding.html
#reimplemented here for speed and to increase resolution to 16bit
def getOtsuThreshold(pixel_hist) :
    # normalize histogram
    hist_norm = pixel_hist/pixel_hist.sum()
    # get cumulative distribution function
    Q = hist_norm.cumsum()
    # find the upper limit of the histogram
    max_val = len(pixel_hist)
    # set up the loop to determine the threshold
    bins = np.arange(max_val); fn_min = np.inf; thresh = -1
    # loop over all possible values to find where the function is minimized
    for i in range(1,max_val):
        p1,p2 = np.hsplit(hist_norm,[i]) # probabilities
        q1,q2 = Q[i],Q[max_val-1]-Q[i] # cum sum of classes
        if q1 < 1.e-6 or q2 < 1.e-6:
            continue
        b1,b2 = np.hsplit(bins,[i]) # weights
        # finding means and variances
        m1,m2 = np.sum(p1*b1)/q1, np.sum(p2*b2)/q2
        v1,v2 = np.sum(((b1-m1)**2)*p1)/q1,np.sum(((b2-m2)**2)*p2)/q2
        # calculates the minimization function
        fn = v1*q1 + v2*q2
        if fn < fn_min:
            fn_min = fn
            thresh = i
    # return the threshold
    return thresh

#helper function to calculate the nth moment of a histogram
#used in finding the skewness and kurtosis
def moment(hist,n,standardized=True) :
    norm = 1.*hist.sum()
    #if there are no entries the moments are undefined
    if norm==0. :
        return float('NaN')
    mean = 0.
    for k,p in enumerate(hist) :
        mean+=p*k
    mean/=norm
    var  = 0.
    moment = 0.
    for k,p in enumerate(hist) :
        var+=p*((k-mean)**2)
        moment+=p*((k-mean)**n)
    var/=norm
    moment/=norm
    #if the moment is zero, then the histogram was just one bin and the moment is undefined
    if moment==0 :
        return float('NaN')
    if standardized :
        return moment/(var**(n/2.))
    else :
        return moment

#helper function to get a list of all the Otsu thresholds for a single layer's pixel histogram
#and a corresponding list of weights for which is the best
def getLayerOtsuThresholdsAndWeights(hist) :
    next_it_pixels = hist; skew = 1000.
    test_thresholds=[]; test_weighted_skew_slopes=[]
    #iterate calculating and applying the Otsu threshold values
    while not math.isnan(skew) :
        #get the threshold from OpenCV's Otsu thresholding procedure
        test_threshold = getOtsuThreshold(next_it_pixels)
        #calculate the skew and kurtosis of the pixels that would be background at this threshold
        bg_pixels = hist[:test_threshold+1]
        skew = moment(bg_pixels,3)
        if not math.isnan(skew) :
            test_thresholds.append(test_threshold)
            skewslope=(moment(hist[:test_threshold+2],3) - moment(hist[:test_threshold],3))/2.
            test_weighted_skew_slopes.append(skewslope/skew if not math.isnan(skewslope) else 0)
        #set the next iteration's pixels
        next_it_pixels = bg_pixels
    #sort the lists in order of decreasing weight
    if len(test_weighted_skew_slopes)>0 and len(test_thresholds)>0 :
        weights_thresholds = list(zip(test_weighted_skew_slopes, test_thresholds))
        weights_thresholds.sort(reverse=True)
        test_weighted_skew_slopes, test_thresholds = zip(*weights_thresholds)
    return test_thresholds, test_weighted_skew_slopes

# a helper function to take a list of layer histograms and return the list of optimal thresholds
#can be run in parallel with an index and returndict
def findLayerThresholds(layer_hists,i=None,rdict=None) :
    #first figure out how many layers there are
    nlayers = layer_hists.shape[-1] if len(layer_hists.shape)>1 else 1
    best_thresholds = []
    #for each layer
    for li in range(nlayers) :
        #get the list of thresholds and their weights
        if nlayers>1 :
            test_thresholds, test_weighted_skew_slopes = getLayerOtsuThresholdsAndWeights(layer_hists[:,li])
        else : 
            test_thresholds, test_weighted_skew_slopes = getLayerOtsuThresholdsAndWeights(layer_hists)
        #figure out the best threshold from them and add it to the list
        if len(test_thresholds)<1 :
            best_thresholds.append(0)
        else :
            best_thresholds.append(test_thresholds[0])
    if nlayers==1 :
        best_thresholds = best_thresholds[0]
    if i is not None and rdict is not None :
        #put the list of thresholds in the return dict
        rdict[i]=best_thresholds
    else :
        return best_thresholds

#################### IMAGE I/O HELPER FUNCTIONS ####################

#helper dataclass to use in multithreading some image handling
class FileReadInfo(MyDataClass) :
    rawfile_path     : str                # the path to the raw file
    sequence_print   : str                # a string of "({i} of {N})" to print
    height           : int                # img height
    width            : int                # img width
    nlayers          : int                # number of img layers
    root_dir         : str                # Clinical_Specimen directory
    smooth_sigma     : int = -1           # gaussian sigma for applying smoothing as images are read (-1 = no smoothing)
    med_exp_times    : List[float] = None # a list of the median exposure times in this image's slide by layer 
    corr_offsets     : List[float] = None # a list of the exposure time correction offsets for this image's slide by layer 

#helper function to parallelize calls to getRawAsHWL (plus optional smoothing and normalization)
def getImageArray(fri) :
    flatfield_logger.info(f'  reading file {fri.rawfile_path} {fri.sequence_print}')
    img_arr = getRawAsHWL(fri.rawfile_path,fri.height,fri.width,fri.nlayers)
    if fri.med_exp_times is not None and fri.corr_offsets is not None and fri.corr_offsets[0] is not None :
        try :
            img_arr = correctImageForExposureTime(img_arr,fri.rawfile_path,fri.root_dir,fri.med_exp_times,fri.corr_offsets)
        except (ValueError, RuntimeError) :
            rtd = os.path.dirname(os.path.dirname(os.path.normpath(fri.rawfile_path)))+os.sep
            img_arr = correctImageForExposureTime(img_arr,fri.rawfile_path,rtd,fri.med_exp_times,fri.corr_offsets)
    if fri.smooth_sigma !=-1 :
        img_arr = smoothImageWorker(img_arr,fri.smooth_sigma)
    return img_arr

#helper function to get the result of getImageArray as a histogram of pixel fluxes instead of an array
def getImageLayerHists(fri) :
    img_array = getImageArray(fri)
    return getImageArrayLayerHistograms(img_array)

#helper function to read and return a group of raw images with multithreading
#set 'smooth_sigma' to some positive integer to smooth images as they're read in
#pass in a list of median exposure times and correction offsets by layer to correct image layer flux 
def readImagesMT(slide_image_filereads,smooth_sigma=-1,med_exposure_times_by_layer=None,et_corr_offsets_by_layer=None) :
    for fr in slide_image_filereads :
        fr.smooth_sigma = smooth_sigma
        fr.med_exp_times = med_exposure_times_by_layer
        fr.corr_offsets = et_corr_offsets_by_layer
    e = ThreadPoolExecutor(len(slide_image_filereads))
    new_img_arrays = list(e.map(getImageArray,[fr for fr in slide_image_filereads]))
    e.shutdown()
    return new_img_arrays

#helper function to read and return a group of image pixel histograms with multithreading
#set 'smooth_sigma' to to some numpy array element to smooth images as they're read in
#pass in a list of median exposure times and correction offsets by layer to correct image layer flux 
def getImageLayerHistsMT(slide_image_filereads,smooth_sigma=-1,med_exposure_times_by_layer=None,et_corr_offsets_by_layer=None) :
    for fr in slide_image_filereads :
        fr.smooth_sigma = smooth_sigma
        fr.med_exp_times = med_exposure_times_by_layer
        fr.corr_offsets = et_corr_offsets_by_layer
    e = ThreadPoolExecutor(len(slide_image_filereads))
    new_img_layer_hists = list(e.map(getImageLayerHists,[fr for fr in slide_image_filereads]))
    e.shutdown()
    return new_img_layer_hists

#helper function to split a list of filenames into chunks to be read in in parallel
def chunkListOfFilepaths(fps,dims,root_dir,n_threads) :
    fileread_chunks = [[]]
    for i,fp in enumerate(fps,start=1) :
        if len(fileread_chunks[-1])>=n_threads :
            fileread_chunks.append([])
        fileread_chunks[-1].append(FileReadInfo(fp,f'({i} of {len(fps)})',dims[0],dims[1],dims[2],root_dir))
    return fileread_chunks

#################### MASKING UTILITY FUNCTIONS ####################

#return a binary mask with all of the areas smaller than min_size removed
def getSizeFilteredMask(mask,min_size,both=True,invert=False) :
    """
    both   = True if regions of ones and zeros should both be filtered (just recursively calls the function with !invert)
    invert = True if regions of zeros instead of ones should be filtered
    """
    if invert :
        mask = (np.where(mask==1,0,1)).astype(mask.dtype)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    sizes = stats[1:, -1]; nb_components = nb_components - 1
    new_mask = np.zeros_like(mask)
    for i in range(0, nb_components):
        if sizes[i] >= min_size :
            new_mask[output == i + 1] = 1
    if invert :
        new_mask = (np.where(new_mask==1,0,1)).astype(mask.dtype)
    if both :
        return getSizeFilteredMask(new_mask,min_size,both=False,invert=(not invert))
    return new_mask

#return the fully-determined single tissue mask for a multilayer image
def getImageTissueMask(image_arr,bkg_thresholds) :
    #figure out where the layer groups are
    if image_arr.shape[-1]==35 :
        mask_layer_groups=UNIV_CONST.LAYER_GROUPS_35
        dapi_layer_group_index=UNIV_CONST.DAPI_LAYER_GROUP_INDEX_35
    elif image_arr.shape[-1]==43 :
        mask_layer_groups=UNIV_CONST.LAYER_GROUPS_43
        dapi_layer_group_index=UNIV_CONST.DAPI_LAYER_GROUP_INDEX_43
    else :
        raise RuntimeError(f'ERROR: no defined list of broadband filter breaks for images with {image_arr.shape[-1]} layers!')
    #smooth the entire image
    sm_image_array = smoothImageWorker(image_arr,CONST.TISSUE_MASK_SMOOTHING_SIGMA)
    #mask each layer individually first
    layer_masks = []
    for li in range(image_arr.shape[-1]) :
        #threshold
        layer_mask = (np.where(sm_image_array[:,:,li]>bkg_thresholds[li],1,0)).astype(np.uint8)
        #convert to UMat to use on the GPU
        layer_mask = cv2.UMat(layer_mask)
        #small close/open
        cv2.morphologyEx(layer_mask,cv2.MORPH_CLOSE,CONST.SMALL_CO_EL,layer_mask,borderType=cv2.BORDER_REPLICATE)
        cv2.morphologyEx(layer_mask,cv2.MORPH_OPEN,CONST.SMALL_CO_EL,layer_mask,borderType=cv2.BORDER_REPLICATE)
        layer_masks.append(layer_mask.get())
    #find the well-defined tissue and background in each layer group
    overall_tissue_mask = np.zeros_like(layer_masks[0])
    overall_background_mask = np.zeros_like(layer_masks[0])
    total_stacked_masks = np.zeros_like(layer_masks[0])
    #for each layer group
    for lgi,lgb in enumerate(mask_layer_groups) :
        stacked_masks = np.zeros_like(layer_masks[0])
        for ln in range(lgb[0],lgb[1]+1) :
            stacked_masks+=layer_masks[ln-1]
        total_stacked_masks+=stacked_masks
        #well-defined tissue is anything called tissue in at least all but two layers
        overall_tissue_mask[stacked_masks>(lgb[1]-lgb[0]-1)]+= 10 if lgi==dapi_layer_group_index else 1
        #well-defined background is anything called background in at least half the layers
        overall_background_mask[stacked_masks<(lgb[1]-lgb[0]+1)/2.]+= 10 if lgi==dapi_layer_group_index else 1
    #threshold tissue/background masks to include only those from the DAPI and at least one other layer group
    overall_tissue_mask = (np.where(overall_tissue_mask>10,1,0)).astype(np.uint8)
    overall_background_mask = (np.where(overall_background_mask>10,1,0)).astype(np.uint8)
    #final mask has tissue=1, background=0
    final_mask = np.zeros_like(layer_masks[0])+2
    final_mask[overall_tissue_mask==1] = 1
    final_mask[overall_background_mask==1] = 0
    #anything left over is signal if it's stacked in at least 60% of the total number of layers
    thresholded_stacked_masks = np.where(total_stacked_masks>(0.6*image_arr.shape[-1]),1,0)
    final_mask[final_mask==2] = thresholded_stacked_masks[final_mask==2]
    if np.min(final_mask) != np.max(final_mask) :
        #filter the tissue and background portions to get rid of the small islands
        final_mask = getSizeFilteredMask(final_mask,min_size=CONST.TISSUE_MIN_SIZE)
        #convert to UMat
        final_mask = cv2.UMat(final_mask)
        #medium size close/open to smooth out edges
        cv2.morphologyEx(final_mask,cv2.MORPH_CLOSE,CONST.MEDIUM_CO_EL,final_mask,borderType=cv2.BORDER_REPLICATE)
        cv2.morphologyEx(final_mask,cv2.MORPH_OPEN,CONST.MEDIUM_CO_EL,final_mask,borderType=cv2.BORDER_REPLICATE)
        final_mask = final_mask.get()
    return final_mask

#return a binary mask with any areas that are already flagged in a prior mask removed
def getExclusiveMask(mask_to_check,prior_mask,min_independent_pixel_frac,invert=True) :
    prior_mask_hot = 1
    if invert :
        mask_to_check = (np.where(mask_to_check==1,0,1)).astype(mask_to_check.dtype)
        prior_mask_hot = 0
    n_regions, regions_im = cv2.connectedComponents(mask_to_check)
    new_mask = np.zeros_like(mask_to_check)
    for region_i in range(1,n_regions) :
        total_region_size = np.sum(regions_im==region_i)
        n_already_selected_pixels = np.sum(prior_mask[regions_im==region_i]==prior_mask_hot)
        if (total_region_size-n_already_selected_pixels)/total_region_size < min_independent_pixel_frac :
            continue
        new_mask[regions_im==region_i] = mask_to_check[regions_im==region_i]
    if invert :
        new_mask = (np.where(new_mask==1,0,1)).astype(mask_to_check.dtype)
    return new_mask

#erode a binary mask near a corresponding tissue mask and filter for size
def getMorphedAndFilteredMask(mask,tissue_mask,min_pixels,min_size) :
    if np.min(mask)<1 and np.max(mask)!=np.min(mask) :
        #a window-sized open incorporating the tissue mask to get rid of any remaining thin borders
        mask_to_transform = cv2.UMat(np.where((mask==0) | (tissue_mask==0),0,1).astype(mask.dtype))
        twice_eroded_fold_mask = cv2.UMat(np.empty_like(mask))
        cv2.morphologyEx(mask,cv2.MORPH_ERODE,CONST.WINDOW_EL,twice_eroded_fold_mask,iterations=2,borderType=cv2.BORDER_REPLICATE)
        cv2.morphologyEx(mask_to_transform,cv2.MORPH_ERODE,CONST.MEDIUM_CO_EL,mask_to_transform,borderType=cv2.BORDER_REPLICATE)
        cv2.morphologyEx(mask_to_transform,cv2.MORPH_OPEN,CONST.WINDOW_EL,mask_to_transform,borderType=cv2.BORDER_REPLICATE)
        cv2.morphologyEx(mask_to_transform,cv2.MORPH_DILATE,CONST.MEDIUM_CO_EL,mask_to_transform,borderType=cv2.BORDER_REPLICATE)
        mask_to_transform = mask_to_transform.get()
        twice_eroded_fold_mask = twice_eroded_fold_mask.get()
        mask[(mask==1) & (tissue_mask==1) & (twice_eroded_fold_mask==0)] = mask_to_transform[(mask==1) & (tissue_mask==1) & (twice_eroded_fold_mask==0)]
        #remove any remaining small spots after the tissue mask incorporation
        mask = getSizeFilteredMask(mask,min_size)
        #make sure there are at least the minimum number of pixels selected
        if np.sum(mask==0)<min_pixels :
            return np.ones_like(mask)
    return mask

#function to compute and return the variance of the normalized laplacian for a given image layer
def getImageLayerLocalVarianceOfNormalizedLaplacian(img_layer) :
    #build the laplacian image and normalize it to get the curvature
    img_laplacian = cv2.Laplacian(img_layer,cv2.CV_32F,borderType=cv2.BORDER_REFLECT)
    img_lap_norm = cv2.UMat(np.empty(img_layer.shape,dtype=np.float32))
    cv2.filter2D(img_layer,cv2.CV_32F,CONST.LOCAL_MEAN_KERNEL,img_lap_norm,borderType=cv2.BORDER_REFLECT)
    img_lap_norm = img_lap_norm.get()
    img_norm_lap = img_laplacian
    img_norm_lap[img_lap_norm!=0] /= img_lap_norm[img_lap_norm!=0]
    img_norm_lap[img_lap_norm==0] = 0
    #find the variance of the normalized laplacian in the neighborhood window
    norm_lap_loc_mean = cv2.UMat(np.empty_like(img_norm_lap))
    cv2.filter2D(img_norm_lap,cv2.CV_32F,CONST.WINDOW_EL,norm_lap_loc_mean,borderType=cv2.BORDER_REFLECT)
    norm_lap_2_loc_mean = cv2.UMat(np.empty_like(img_norm_lap))
    cv2.filter2D(np.power(img_norm_lap,2),cv2.CV_32F,CONST.WINDOW_EL,norm_lap_2_loc_mean,borderType=cv2.BORDER_REFLECT)
    npix = np.sum(CONST.WINDOW_EL)
    norm_lap_loc_mean = norm_lap_loc_mean.get()
    norm_lap_2_loc_mean = norm_lap_2_loc_mean.get()
    norm_lap_loc_mean /= npix
    norm_lap_2_loc_mean /= npix
    local_norm_lap_var = np.abs(norm_lap_2_loc_mean-np.power(norm_lap_loc_mean,2))
    #return the local variance of the normalized laplacian
    return local_norm_lap_var

#function to return a blur mask for a given image layer group, along with a dictionary of plots to add to the group for this image
def getImageLayerGroupBlurMask(img_array,exp_times,layer_group_bounds,nlv_cut,n_layers_flag_cut,max_mean,brightest_layer_n,ethistandbins,return_plots=True) :
    #start by making a mask for every layer in the group
    stacked_masks = np.zeros(img_array.shape[:-1],dtype=np.uint8)
    brightest_layer_nlv = None
    for ln in range(layer_group_bounds[0],layer_group_bounds[1]+1) :
        #get the local variance of the normalized laplacian image
        img_nlv = getImageLayerLocalVarianceOfNormalizedLaplacian(img_array[:,:,ln-1])
        if ln==brightest_layer_n :
            brightest_layer_nlv = img_nlv
        #get the mean of those local normalized laplacian variance values in the window size
        img_nlv_loc_mean = cv2.UMat(np.empty_like(img_nlv))
        cv2.filter2D(img_nlv,cv2.CV_32F,CONST.SMALLER_WINDOW_EL,img_nlv_loc_mean,borderType=cv2.BORDER_REFLECT)
        img_nlv_loc_mean=img_nlv_loc_mean.get()
        img_nlv_loc_mean/=np.sum(CONST.SMALLER_WINDOW_EL)
        #threshold on the local variance of the normalized laplacian and the local mean of those values to make a binary mask
        layer_mask = (np.where((img_nlv>nlv_cut) | (img_nlv_loc_mean>max_mean),1,0)).astype(np.uint8)
        if np.min(layer_mask) != np.max(layer_mask) :
            #convert to UMat
            layer_mask = cv2.UMat(layer_mask)
            #small open/close to refine it
            cv2.morphologyEx(layer_mask,cv2.MORPH_OPEN,CONST.SMALL_CO_EL,layer_mask,borderType=cv2.BORDER_REPLICATE)
            cv2.morphologyEx(layer_mask,cv2.MORPH_CLOSE,CONST.SMALL_CO_EL,layer_mask,borderType=cv2.BORDER_REPLICATE)
            #erode by the smaller window element
            cv2.morphologyEx(layer_mask,cv2.MORPH_ERODE,CONST.SMALLER_WINDOW_EL,layer_mask,borderType=cv2.BORDER_REPLICATE)
            layer_mask = layer_mask.get()
        #add it to the stack 
        stacked_masks+=layer_mask
    #determine the final mask for this group by thresholding on how many individual layers contribute
    group_blur_mask = (np.where(stacked_masks>n_layers_flag_cut,1,0)).astype(np.uint8)    
    if np.min(group_blur_mask) != np.max(group_blur_mask) :
        #medium sized open/close to refine it
        group_blur_mask = cv2.UMat(group_blur_mask)
        cv2.morphologyEx(group_blur_mask,cv2.MORPH_OPEN,CONST.MEDIUM_CO_EL,group_blur_mask,borderType=cv2.BORDER_REPLICATE)
        cv2.morphologyEx(group_blur_mask,cv2.MORPH_CLOSE,CONST.MEDIUM_CO_EL,group_blur_mask,borderType=cv2.BORDER_REPLICATE)
        group_blur_mask = group_blur_mask.get()
    #set up the plots to return
    if return_plots :
        plot_img_layer = img_array[:,:,brightest_layer_n-1]
        sorted_pil = np.sort(plot_img_layer[group_blur_mask==1].flatten())
        pil_max = sorted_pil[int(0.95*len(sorted_pil))]; pil_min = sorted_pil[0]
        norm = 255./(pil_max-pil_min)
        im_c = (np.clip(norm*(plot_img_layer-pil_min),0,255)).astype(np.uint8)
        overlay_c = np.array([im_c,im_c*group_blur_mask,im_c*group_blur_mask]).transpose(1,2,0)
        plots = [{'image':plot_img_layer,'title':f'raw IMAGE layer {brightest_layer_n}'},
                 {'image':overlay_c,'title':f'layer {layer_group_bounds[0]}-{layer_group_bounds[1]} blur mask overlay (clipped)'},
                 {'bar':ethistandbins[0],'bins':ethistandbins[1],'xlabel':f'layer {layer_group_bounds[0]}-{layer_group_bounds[1]} exposure times (ms)','line_at':exp_times[layer_group_bounds[0]-1]},
                 {'hist':brightest_layer_nlv.flatten(),'xlabel':'variance of normalized laplacian','line_at':nlv_cut},
                 {'image':stacked_masks,'title':f'stacked layer masks (cut at {n_layers_flag_cut})','cmap':'gist_ncar','vmin':0,'vmax':layer_group_bounds[1]-layer_group_bounds[0]+1},
                ]
    else :
        plots = None
    #return the blur mask for the layer group and the list of plot dictionaries
    return group_blur_mask, plots

#return the tissue fold mask for an image combining information from all layer groups
def getImageTissueFoldMask(sm_img_array,exp_times,tissue_mask,exp_t_hists,mask_layer_groups,brightest_layers,dapi_lgi,rbc_lgi,return_plots=False) :
    #make the list of flag cuts (how many layers can miss being flagged to include the region in the layer group mask)
    fold_flag_cuts = []
    for lgi,lgb in enumerate(mask_layer_groups) :
        if lgi==dapi_lgi or lgi==rbc_lgi :
            fold_flag_cuts.append(3)
        elif lgb[1]-lgb[0]<3 :
            fold_flag_cuts.append(0)
        else :
            fold_flag_cuts.append(1)
    #get the fold masks for each layer group
    fold_masks_by_layer_group = []; fold_mask_plots_by_layer_group = []
    for lgi,lgb in enumerate(mask_layer_groups) :
        lgtfm, lgtfmps = getImageLayerGroupBlurMask(sm_img_array,
                                                    exp_times,
                                                    lgb,
                                                    CONST.FOLD_NLV_CUT,
                                                    fold_flag_cuts[lgi],
                                                    CONST.FOLD_MAX_MEAN,
                                                    brightest_layers[lgi],
                                                    exp_t_hists[lgi],
                                                    return_plots)
        fold_masks_by_layer_group.append(lgtfm)
        fold_mask_plots_by_layer_group.append(lgtfmps)
    #combine the layer group blur masks to get the final mask for all layers
    stacked_fold_masks = np.zeros_like(fold_masks_by_layer_group[0])
    for lgi,layer_group_fold_mask in enumerate(fold_masks_by_layer_group) :
        stacked_fold_masks[layer_group_fold_mask==0]+=10 if lgi in (dapi_lgi,rbc_lgi) else 1
    overall_fold_mask = (np.where(stacked_fold_masks>11,0,1)).astype(np.uint8)
    #morph and filter the mask using the common operations
    overall_fold_mask = getMorphedAndFilteredMask(overall_fold_mask,tissue_mask,CONST.FOLD_MIN_PIXELS,CONST.FOLD_MIN_SIZE)
    if return_plots :
        return overall_fold_mask,fold_mask_plots_by_layer_group
    else :
        return overall_fold_mask,None

#return the single blur mask for an image
def getImageBlurMask(img_array,exp_times,tissue_mask,exp_time_hists,return_plots=False) :
    #figure out where the layer groups are, etc.
    if img_array.shape[-1]==35 :
        mask_layer_groups=UNIV_CONST.LAYER_GROUPS_35
        brightest_layers=UNIV_CONST.BRIGHTEST_LAYERS_35
        dapi_layer_group_index=UNIV_CONST.DAPI_LAYER_GROUP_INDEX_35
        rbc_layer_group_index=UNIV_CONST.RBC_LAYER_GROUP_INDEX_35
    elif img_array.shape[-1]==43 :
        mask_layer_groups=UNIV_CONST.LAYER_GROUPS_43
        brightest_layers=UNIV_CONST.BRIGHTEST_LAYERS_43
        dapi_layer_group_index=UNIV_CONST.DAPI_LAYER_GROUP_INDEX_43
        rbc_layer_group_index=UNIV_CONST.RBC_LAYER_GROUP_INDEX_43
    else :
        raise RuntimeError(f'ERROR: no defined list of broadband filter breaks for images with {img_array.shape[-1]} layers!')
    #smooth the image array for both the tissue fold and DAPI layer blur detection
    sm_img_array = smoothImageWorker(img_array,CONST.BLUR_MASK_SMOOTHING_SIGMA)
    #get the tissue fold mask and its associated plots
    tissue_fold_mask,tissue_fold_plots_by_layer_group = getImageTissueFoldMask(sm_img_array,exp_times,tissue_mask,exp_time_hists,mask_layer_groups,
                                                                               brightest_layers,dapi_layer_group_index,rbc_layer_group_index,return_plots)
    #get masks for the blurriest areas of the DAPI layer group
    dapi_dust_mask,dapi_dust_plots = getImageLayerGroupBlurMask(sm_img_array,
                                                                exp_times,
                                                                mask_layer_groups[dapi_layer_group_index],
                                                                CONST.DUST_NLV_CUT,
                                                                0.5*(mask_layer_groups[dapi_layer_group_index][1]-mask_layer_groups[dapi_layer_group_index][0]+1),
                                                                CONST.DUST_MAX_MEAN,
                                                                brightest_layers[dapi_layer_group_index],
                                                                exp_time_hists[dapi_layer_group_index],
                                                                False)
    #same morphology transformations as for the multilayer blur masks
    dapi_dust_mask = getMorphedAndFilteredMask(dapi_dust_mask,tissue_mask,CONST.DUST_MIN_PIXELS,CONST.DUST_MIN_SIZE)
    #make sure any regions in that mask are sufficiently exclusive w.r.t. what's already flagged
    dapi_dust_mask = getExclusiveMask(dapi_dust_mask,tissue_fold_mask,0.25)
    #combine the multilayer and single layer blur masks into one by multiplying them together
    final_blur_mask = tissue_fold_mask*dapi_dust_mask
    #return the blur mask and the list of plot dicts 
    return final_blur_mask,tissue_fold_plots_by_layer_group

#return the list of saturation masks by layer group for a given image
def getImageSaturationMasks(image_arr,norm_ets) :
    #figure out where the layer groups are, etc.
    if image_arr.shape[-1]==35 :
        mask_layer_groups=UNIV_CONST.LAYER_GROUPS_35
        saturation_intensity_cuts=CONST.SATURATION_INTENSITY_CUTS_35
    elif image_arr.shape[-1]==43 :
        mask_layer_groups=UNIV_CONST.LAYER_GROUPS_43
        saturation_intensity_cuts=CONST.SATURATION_INTENSITY_CUTS_43
    else :
        raise RuntimeError(f'ERROR: no defined list of broadband filter breaks for images with {image_arr.shape[-1]} layers!')
    #normalize the image by its exposure time
    normalized_image_arr = np.zeros(image_arr.shape,dtype=np.float32)
    for li in range(image_arr.shape[-1]) :
        normalized_image_arr[:,:,li] = image_arr[:,:,li]/norm_ets[li]
    #smooth the exposure time-normalized image
    sm_n_image_arr = smoothImageWorker(normalized_image_arr,CONST.TISSUE_MASK_SMOOTHING_SIGMA)
    #make masks for each layer group
    layer_group_saturation_masks = []
    for lgi,lgb in enumerate(mask_layer_groups) :
        stacked_masks = np.zeros(sm_n_image_arr.shape[:-1],dtype=np.uint8)
        for ln in range(lgb[0],lgb[1]+1) :
            #threshold the image layer to make a binary mask
            layer_mask = (np.where(sm_n_image_arr[:,:,ln-1]>saturation_intensity_cuts[lgi],0,1)).astype(np.uint8)
            stacked_masks+=layer_mask
        #the final mask is anything flagged in ANY layer
        group_mask = (np.where(stacked_masks>lgb[1]-lgb[0],1,0)).astype(np.uint8)    
        if np.min(group_mask)!=np.max(group_mask) :
            #convert to UMat
            group_mask = cv2.UMat(group_mask)
            #medium sized open/close to refine it
            cv2.morphologyEx(group_mask,cv2.MORPH_OPEN,CONST.MEDIUM_CO_EL,group_mask,borderType=cv2.BORDER_REPLICATE)
            cv2.morphologyEx(group_mask,cv2.MORPH_CLOSE,CONST.MEDIUM_CO_EL,group_mask,borderType=cv2.BORDER_REPLICATE)
            group_mask = group_mask.get()
            #filter the mask for the total number of pixels and regions by the minimum size
            group_mask = getSizeFilteredMask(group_mask,CONST.SATURATION_MIN_SIZE)
        if np.sum(group_mask==0)<CONST.SATURATION_MIN_PIXELS :
            group_mask = np.ones_like(group_mask)
        layer_group_saturation_masks.append(group_mask)
    return layer_group_saturation_masks

#################### USEFUL PLOTTING FUNCTION ####################

def drawThresholds(img_array, *, layer_index=0, emphasize_mask=None, show_regions=False, saveas=None, plotstyling = lambda fig, ax: None):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1)
    if len(img_array.shape)>2 :
        img_array = img_array[layer_index]
    hist = getImageArrayLayerHistograms(img_array)
    if emphasize_mask is not None:
        hist_emphasize = getImageArrayLayerHistograms(img_array, mask=emphasize_mask)
    thresholds, weights = getLayerOtsuThresholdsAndWeights(hist)
    histmax = np.max(np.argwhere(hist!=0))
    hist = hist[:histmax+1]
    plt.bar(range(len(hist)), hist, width=1)
    if emphasize_mask is not None:
        hist_emphasize = hist_emphasize[:histmax+1]
        plt.bar(range(len(hist)), hist_emphasize, width=1)
    for threshold, weight in zip(thresholds, weights):
        plt.axvline(x=threshold, color="red", alpha=0.5+0.5*(weight-min(weights))/(max(weights)-min(weights)))
    plotstyling(fig, ax)
    if saveas is None:
        plt.show()
    else:
        plt.savefig(saveas)
        plt.close()
    if show_regions:
        for t1, t2 in more_itertools.pairwise([0]+sorted(thresholds)+[float("inf")]):
            if t1 == t2: continue #can happen if 0 is a threshold
            print(t1, t2)
            plt.imshow(img_array)
            lower = np.array(
              [0*img_array+1, 0*img_array, 0*img_array, img_array < t1],
              dtype=float
            ).transpose(1, 2, 0)
            higher = np.array(
              [0*img_array, 0*img_array+1, 0*img_array, img_array > t2],
              dtype=float
            ).transpose(1, 2, 0)
            plt.imshow(lower)
            plt.imshow(higher)
            plt.show()
