#imports
from ..image_masking.config import CONST as MASKING_CONST
from ...utilities.dataclasses import MyDataClass
from ...utilities.img_file_io import smooth_image_worker
import numpy as np
import math

#A small dataclass to hold entries in the background threshold datatable
class RectangleThresholdTableEntry(MyDataClass) :
    rect_n                  : int
    layer_n                 : int
    counts_threshold        : int
    counts_per_ms_threshold : float

#helper class for logging included/excluded fields
class FieldLog(MyDataClass) :
    slide   : str
    file     : str
    location : str
    use      : str
    stacked_in_layers : str = ''

def calculate_statistics_for_image(image) :
    """
    Return the maximum, minimum, 5th-95th percentile spread, and standard deviation 
    of the numbers in a given image in the entire image region and in the central "primary region"
    """
    yclip = int(image.shape[0]*0.1)
    xclip = int(image.shape[1]*0.1)
    flatfield_image_clipped=image[yclip:-yclip,xclip:-xclip,:]
    overall_max = np.max(image)
    overall_min = np.min(image)
    central_max = np.max(flatfield_image_clipped)
    central_min = np.min(flatfield_image_clipped)
    overall_spreads_by_layer = []; overall_stddevs_by_layer = []
    central_spreads_by_layer = []; central_stddevs_by_layer = []
    for li in range(image.shape[-1]) :
        sorted_u_layer = np.sort((image[:,:,li]).flatten())/np.mean(image[:,:,li])
        sorted_c_layer = np.sort((flatfield_image_clipped[:,:,li]).flatten())/np.mean(image[:,:,li])
        overall_spreads_by_layer.append(sorted_u_layer[int(0.95*len(sorted_u_layer))]-sorted_u_layer[int(0.05*len(sorted_u_layer))])
        overall_stddevs_by_layer.append(np.std(sorted_u_layer))
        central_spreads_by_layer.append(sorted_c_layer[int(0.95*len(sorted_c_layer))]-sorted_c_layer[int(0.05*len(sorted_c_layer))])
        central_stddevs_by_layer.append(np.std(sorted_c_layer))
    overall_spread = np.mean(np.array(overall_spreads_by_layer))
    overall_stddev = np.mean(np.array(overall_stddevs_by_layer))
    central_spread = np.mean(np.array(central_spreads_by_layer))
    central_stddev = np.mean(np.array(central_stddevs_by_layer))
    return overall_max, overall_min, overall_spread, overall_stddev, central_max, central_min, central_spread, central_stddev

#################### THRESHOLDING HELPER FUNCTIONS ####################

#helper function to determine the Otsu threshold given a histogram of pixel values 
#algorithm from python code at https://docs.opencv.org/3.4/d7/d4d/tutorial_py_thresholding.html
#reimplemented here for speed and to increase resolution to 16bit
def get_otsu_threshold(pixel_hist) :
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
def get_layer_otsu_thresholds_and_weights(hist) :
    next_it_pixels = hist; skew = 1000.
    test_thresholds=[]; test_weighted_skew_slopes=[]
    #iterate calculating and applying the Otsu threshold values
    while not math.isnan(skew) :
        #get the threshold from OpenCV's Otsu thresholding procedure
        test_threshold = get_otsu_threshold(next_it_pixels)
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
def find_layer_thresholds(layer_hists) :
    #first figure out how many layers there are
    nlayers = layer_hists.shape[-1] if len(layer_hists.shape)>1 else 1
    best_thresholds = []
    #for each layer
    for li in range(nlayers) :
        #get the list of thresholds and their weights
        if nlayers>1 :
            test_thresholds, test_weighted_skew_slopes = get_layer_otsu_thresholds_and_weights(layer_hists[:,li])
        else : 
            test_thresholds, test_weighted_skew_slopes = get_layer_otsu_thresholds_and_weights(layer_hists)
        #figure out the best threshold from them and add it to the list
        if len(test_thresholds)<1 :
            best_thresholds.append(0)
        else :
            best_thresholds.append(test_thresholds[0])
    if nlayers==1 :
        best_thresholds = best_thresholds[0]
    return best_thresholds

def get_background_thresholds_and_pixel_hists_for_rectangle_image(rimg) :
    """
    Return the optimal background thresholds and pixel histograms by layer for a given rectangle image
    (this function is in utilities and not a function of a rectangle so several iterations of it can be run in parallel)
    """
    smoothed_im = smooth_image_worker(rimg,MASKING_CONST.TISSUE_MASK_SMOOTHING_SIGMA,gpu=True)
    nbins = np.iinfo(rimg.dtype).max+1
    layer_hists = np.zeros((nbins,rimg.shape[-1]),dtype=np.uint64)
    for li in range(rimg.shape[-1]) :
        layer_hists[:,li],_ = np.histogram(smoothed_im[:,:,li],nbins,(0,nbins))
    thresholds = find_layer_thresholds(layer_hists)
    return thresholds, layer_hists
