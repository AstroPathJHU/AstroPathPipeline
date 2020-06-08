#imports for this file and other that are shared
from ..utilities.misc import cd
import numpy as np
import os, cv2, logging

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
GRAYSCALE_MAX_VALUE=np.iinfo(np.uint8).max
UPPER_THRESHOLD_KURTOSIS_CUT=2.0
MIN_POINTS_TO_SEARCH=50
MAX_POINTS_TO_SEARCH=300
ALLOW_NEGATIVE_SKEW_FROM=30

#mean_image constants
IMG_DTYPE_OUT=np.float64
FILE_EXT='.bin'
PIXEL_INTENSITY_PLOT_NAME='pixel_intensity_plot.png'
N_IMAGES_STACKED_PER_LAYER_PLOT_NAME='n_images_stacked_per_layer.png'
N_IMAGES_STACKED_PER_LAYER_TEXT_FILE_NAME='n_images_stacked_per_layer.txt'

#parameters for creating the image masks
GENTLE_GAUSSIAN_SMOOTHING_SIGMA = 5
CO1_EL  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
CO2_EL  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(16,16))
C3_EL  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(45,45))
OPEN_3_ITERATIONS = 3
MASKING_PLOT_DIR_NAME = 'masking_plots'
MASKING_PLOT_FIG_SIZE = (12.8,18.4)

#info for figures that get created
IMAGE_LAYER_PLOT_DIRECTORY_NAME='image_layer_pngs'
POSTRUN_PLOT_DIRECTORY_NAME='postrun_plots'
IMG_LAYER_FIG_WIDTH=6.4 #width of image layer figures created in inches
INTENSITY_FIG_WIDTH=14.0 #width of the intensity plot figure
LAST_FILTER_LAYERS = [9,18,25,32] #last image layers of each broadband filter
