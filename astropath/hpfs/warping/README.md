# Warping

The "warping" portion of the code provides routines to determine the best-fit warping model in a given image layer. The model is found by minimizing the mean squared error between aligned overlapping image regions. Several fits are performed to find the best patterns for particular "octets" (sets of 8 overlaps surrounding a single HPF), and the final pattern is the average of many individual octet results, weighted by the fractional cost reduction from the unwarped alignment cost. Warping models are defined using camera matrices and radial distortion parameters [as described in OpenCV's Camera Calibration methods](https://docs.opencv.org/master/dc/dbb/tutorial_py_calibration.html), and implemented as instances of [the CameraWarp class](./warp.py#L296-621).

To run the code in the most common use case, enter the following command and arguments:

`run_warping_fits warp_fit [slide_IDs] [rawfile_directory] [root_directory] [working_directory] [n_threads] --exposure_time_offset_file [path_to_exposure_time_offset_file] --flatfield_file [path_to_flatfield_file] --threshold_file_dir [path_to_threshold_file_directory] --layer [layer_number]`

where:
-`[slide_IDs]` is a comma-separated list of slide IDs to use (i.e., "`ZW1,ZW2,ZW3,ZW4,ZW5,ZW6,ZW7,ZW8,ZW9,ZW10`")
-`[rawfile_directory]` is the path to a directory containing a `[slide_ID]` subdirectory for each `[slide_ID]` in `[slide_IDs]` containing the slide's ".Data.dat" files (i.e. ``\\bki07\dat``)
-`[root_directory]` is the path to the usual "Clinical_Specimen" directory containing a `[slide_ID]` subdirectory for each `[slide_ID]` in `[slide_IDs]` (i.e. `\\bki02\E\Clinical_Specimen`)
-`[working_directory]` is the path to the directory that should contain all of the output of the run (it will be created if it doesn't exist).
-`[n_threads]` is the maximum number of CPUs to use simultaneously in running the groups of fits (the independent fits for each octet are handled by a multiprocessing pool; this is the number of workers allowed in that pool). This should be adjusted depending on running conditions, GPU availability, etc., but on BKI06 a single invocation of the above command can usually be run with 10 threads.
-`[path_to_exposure_time_offset_file]` is the path to a .csv file holding the exposure time correction "dark current" offsets as a list of ["LayerOffset" objects](https://github.com/AstroPathJHU/AstroPathPipeline/blob/master/astropath/utilities/img_file_io.py#L28-L34) (i.e. the file [here](https://github.com/AstroPathJHU/alignmentjenkinsdata/blob/master/corrections/best_exposure_time_offsets_Vectra_9_8_2020.csv)), which is output by the code that does the exposure time correction fits
-`[path_to_flatfield_file]` is the path to a `flatfield_BatchID_[batch_ID].bin` file containing the flatfield correction factors to apply to each slide in `[slide_IDs]` 
-`[path_to_threshold_file_directory]` is the path to a directory containing `[slideID]_background_thresholds.txt` files for each `[slide_ID]` in `[slide_IDs]`
-`[layer_number]` is the image layer to find the patterns for (indexed starting at 1)

Under these conditions, the following things will happen:
1. The set of valid overlap octets to use will be found for each slide in `[slide_IDs]`. Octets are "valid" if, after correction for exposure time and flatfielding, every overlap shows at least 85% tissue (as opposed to background) determined by a simple thresholding at the value in the corresponding `[slideID]_background_thresholds.txt` file.
1. The full set of all valid octets will be randomly divided into three subsets of 50, 50, and 100 octets each
1. 50 independent fits to the first subset of octets will be performed to find an initial warping model to use as a starting point for subsequent fits.
1. 50 more independent fits to the second subset of octets will be performed with the radial distortion parameters fixed to the average of the previous set of fit results, weighted by their fractional reduction in alignment cost.
 