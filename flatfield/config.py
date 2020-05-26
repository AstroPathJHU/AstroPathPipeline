import numpy as np
import cv2, logging

#Class for errors encountered during flatfielding
class FlatFieldError(Exception) :
    pass

#logger
flatfield_logger = logging.getLogger("flatfield")
flatfield_logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(message)s    [%(funcName)s, %(asctime)s]"))
flatfield_logger.addHandler(handler)

#flatield_producer globals
FILEPATH_TEXT_FILE_NAME='filepath_log.txt'
THRESHOLDING_PLOT_DIR_NAME='thresholding_info'
THRESHOLD_TEXT_FILE_NAME_STEM='background_thresholds.txt'

#flatfield_sample globals
RECTANGLE_LOCATION_PLOT_STEM='rectangle_locations'
UPPER_THRESHOLD_KURTOSIS_CUT=2.0
MIN_POINTS_TO_SEARCH=50

#mean_image constants
IMG_DTYPE_OUT=np.float64
FILE_EXT='.bin'

#parameters for creating the image masks
GENTLE_GAUSSIAN_SMOOTHING_SIGMA = 5
ERODE1_EL  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
DILATE1_EL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(61,61))
DILATE2_EL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(25,25))
ERODE2_EL  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(64,64))
ERODE3_EL  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(37,37))
DILATE3_EL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
DILATE3_ITERATIONS = 3
MAX_BG_PIXEL_MEAN_NORM_STD_DEV = 0.15
MASKING_PLOT_DIR_NAME = 'masking_plots'
MASKING_PLOT_FIG_SIZE = (17,24)
#ADD_IF_SHARED_IN_AT_LEAST  = 0.80
#DROP_IF_ABSENT_IN_AT_LEAST = 0.90
#info for figures that get created
IMAGE_LAYER_PLOT_DIRECTORY_NAME='image_layer_pngs'
POSTRUN_PLOT_DIRECTORY_NAME='postrun_plots'
IMG_LAYER_FIG_WIDTH=7.5 #width of image layer figures created in inches
INTENSITY_FIG_WIDTH=11.0 #width of the intensity plot figure
LAST_FILTER_LAYERS = [9,18,25,32] #last image layers of each broadband filter
