import numpy as np
import logging

#run_flatield globals
IMG_DTYPE_IN = np.uint16
FILEPATH_TEXT_FILE_NAME='filepath_log.txt'
PLOT_DIRECTORY_NAME='plots'

#mean_image constants
IMG_DTYPE_OUT=np.float64
FILE_EXT='.bin'
IMG_LAYER_FIG_WIDTH=7.5 #width of image layer figures created in inches
INTENSITY_FIG_WIDTH=11.0 #width of the intensity plot figure
LAST_FILTER_LAYERS = [9,18,25,32] #last image layers of each broadband filter

#logger
flatfield_logger = logging.getLogger("flatfield")
flatfield_logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(message)s    [%(funcName)s, %(asctime)s]"))
flatfield_logger.addHandler(handler)