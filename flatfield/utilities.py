from .config import CONST
from ..utilities.img_file_io import getRawAsHWL, correctImageForExposureTime, smoothImageWorker
from concurrent.futures import ThreadPoolExecutor
from typing import List
import matplotlib.pyplot as plt
import numpy as np
import os, logging, math, dataclasses, more_itertools

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
@dataclasses.dataclass
class FieldLog :
    sample   : str
    file     : str
    location : str
    use      : str
    stacked_in_layers : List[int] = None

#helper class for inputting samples with their names and raw/metadata directories
@dataclasses.dataclass
class FlatfieldSampleInfo :
    name : str
    rawfile_top_dir : str
    metadata_top_dir : str

#################### GENERAL HELPER FUNCTIONS ####################

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

#helper function to return the sample name from a whole filepath
def sampleNameFromFilepath(fp) :
    return os.path.basename(os.path.dirname(os.path.normpath(fp)))

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
@dataclasses.dataclass(eq=False, repr=False)
class FileReadInfo :
    rawfile_path     : str                # the path to the raw file
    sequence_print   : str                # a string of "({i} of {N})" to print
    height           : int                # img height
    width            : int                # img width
    nlayers          : int                # number of img layers
    metadata_top_dir : str                # top directory with metadata files
    to_smooth        : bool = False       # whether the image should be smoothed
    med_exp_times    : List[float] = None # a list of the median exposure times in this image's sample by layer 
    corr_offsets     : List[float] = None # a list of the exposure time correction offsets for this image's sample by layer 

#helper function to parallelize calls to getRawAsHWL (plus optional smoothing and normalization)
def getImageArray(fri) :
    flatfield_logger.info(f'  reading file {fri.rawfile_path} {fri.sequence_print}')
    img_arr = getRawAsHWL(fri.rawfile_path,fri.height,fri.width,fri.nlayers)
    if fri.med_exp_times is not None and fri.corr_offsets is not None and fri.corr_offsets[0] is not None :
        try :
            img_arr = correctImageForExposureTime(img_arr,fri.rawfile_path,fri.metadata_top_dir,fri.med_exp_times,fri.corr_offsets)
        except ValueError :
            rtd = os.path.dirname(os.path.dirname(os.path.normpath(fri.rawfile_path)))
            img_arr = correctImageForExposureTime(img_arr,fri.rawfile_path,rtd,fri.med_exp_times,fri.corr_offsets)
    if fri.to_smooth :
        img_arr = smoothImageWorker(img_arr,CONST.GENTLE_GAUSSIAN_SMOOTHING_SIGMA)
    return img_arr

#helper function to get the result of getImageArray as a histogram of pixel fluxes instead of an array
def getImageLayerHists(fri) :
    img_array = getImageArray(fri)
    return getImageArrayLayerHistograms(img_array)

#helper function to read and return a group of raw images with multithreading
#set 'smoothed' to True when calling to smooth images with gentle gaussian filter as they're read in
#pass in a list of median exposure times and correction offsets by layer to correct image layer flux 
def readImagesMT(sample_image_filereads,smoothed=False,med_exposure_times_by_layer=None,et_corr_offsets_by_layer=None) :
    for fr in sample_image_filereads :
        fr.to_smooth = smoothed
        fr.med_exp_times = med_exposure_times_by_layer
        fr.corr_offsets = et_corr_offsets_by_layer
    e = ThreadPoolExecutor(len(sample_image_filereads))
    new_img_arrays = list(e.map(getImageArray,[fr for fr in sample_image_filereads]))
    e.shutdown()
    return new_img_arrays

#helper function to read and return a group of image pixel histograms with multithreading
#set 'smoothed' to True when calling to smooth images with gentle gaussian filter as they're read in
#pass in a list of median exposure times and correction offsets by layer to correct image layer flux 
def getImageLayerHistsMT(sample_image_filereads,smoothed=False,med_exposure_times_by_layer=None,et_corr_offsets_by_layer=None) :
    for fr in sample_image_filereads :
        fr.to_smooth = smoothed
        fr.med_exp_times = med_exposure_times_by_layer
        fr.corr_offsets = et_corr_offsets_by_layer
    e = ThreadPoolExecutor(len(sample_image_filereads))
    new_img_layer_hists = list(e.map(getImageLayerHists,[fr for fr in sample_image_filereads]))
    e.shutdown()
    return new_img_layer_hists

#helper function to split a list of filenames into chunks to be read in in parallel
def chunkListOfFilepaths(fps,dims,mtd,n_threads) :
    fileread_chunks = [[]]
    for i,fp in enumerate(fps,start=1) :
        if len(fileread_chunks[-1])>=n_threads :
            fileread_chunks.append([])
        fileread_chunks[-1].append(FileReadInfo(fp,f'({i} of {len(fps)})',dims[0],dims[1],dims[2],mtd))
    return fileread_chunks

#################### USEFUL PLOTTING FUNCTION ####################

def drawThresholds(img_array, *, layer_index=0, emphasize_mask=None, show_regions=False, saveas=None, plotstyling = lambda fig, ax: None):
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
