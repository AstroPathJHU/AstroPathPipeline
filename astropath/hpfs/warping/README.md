# Warping

The "warping" portion of the code provides routines to determine the best-fit warping model in a given image layer. The model is found by minimizing the mean squared error between aligned overlapping image regions. Several fits are performed to find the best patterns for particular "octets" (sets of 8 overlaps surrounding a single HPF), and the final pattern is the average of many individual octet results, weighted by the fractional cost reduction from the unwarped alignment cost. Warping models are defined using camera matrices and radial distortion parameters [as described in OpenCV's Camera Calibration methods](https://docs.opencv.org/master/dc/dbb/tutorial_py_calibration.html), and implemented as instances of [the CameraWarp class](./warp.py#L296-L621).

To run the code in the most common use case, enter the following command and arguments:

`run_warping_fits warp_fit [slide_IDs] [rawfile_directory] [root_directory] [working_directory] [n_threads] --exposure_time_offset_file [path_to_exposure_time_offset_file] --flatfield_file [path_to_flatfield_file] --threshold_file_dir [path_to_threshold_file_directory] --layer [layer_number]`

where:
- `[slide_IDs]` is a comma-separated list of slide IDs to use (i.e., "`ZW1,ZW2,ZW3,ZW4,ZW5,ZW6,ZW7,ZW8,ZW9,ZW10`")
- `[rawfile_directory]` is the path to a directory containing a `[slide_ID]` subdirectory for each `[slide_ID]` in `[slide_IDs]` containing the slide's ".Data.dat" files (i.e. ``\\bki07\dat``)
- `[root_directory]` is the path to the usual "Clinical_Specimen" directory containing a `[slide_ID]` subdirectory for each `[slide_ID]` in `[slide_IDs]` (i.e. `\\bki02\E\Clinical_Specimen`)
- `[working_directory]` is the path to the directory that should contain all of the output of the run (it will be created if it doesn't exist).
- `[n_threads]` is the maximum number of CPUs to use simultaneously in running the groups of fits (the independent fits for each octet are handled by a multiprocessing pool; this is the number of workers allowed in that pool). This should be adjusted depending on running conditions, GPU availability, etc., but on BKI06 a single invocation of the above command can usually be run with 10 threads.
- `[path_to_exposure_time_offset_file]` is the path to a .csv file holding the exposure time correction "dark current" offsets as a list of [`LayerOffset` objects](../../utilities/img_file_io.py#L31-L36), which is output by the code that does the exposure time correction fits
- `[path_to_flatfield_file]` is the path to a `flatfield_BatchID_[batch_ID].bin` file containing the flatfield correction factors to apply to each slide in `[slide_IDs]` 
- `[path_to_threshold_file_directory]` is the path to a directory containing `[slideID]_background_thresholds.txt` files for each `[slide_ID]` in `[slide_IDs]`
- `[layer_number]` is the image layer to find the patterns for (indexed starting at 1)

Under these conditions, the following things will happen:
1. The set of valid overlap octets to use will be found for each slide in `[slide_IDs]`. Octets are considered valid if every overlap in the octet shows at least 85% tissue as determined by a simple thresholding at the value in the corresponding `[slideID]_background_thresholds.txt` file (after correction for exposure time and flatfielding).
1. The full set of all valid octets will be randomly divided into three subsets of 50, 50, and 100 octets each
1. 50 independent fits to the first subset of octets will be performed to find an initial warping model to use as a starting point for subsequent fits.
1. 50 additional independent fits to the second subset of octets will be performed with the radial distortion parameters fixed to the weighted average of the previous set of fit results.
1. 100 additional independent fits to the third subset of octets will be performed with all parameters floating, but with the center principal point location constrained to within 2.5 sigma of the weighted average of the previous set of fit results. The final model is the weighted average of this last set of fit results.
This multistep procedure is used to prevent tunneling into local minima around inaccurate center point locations, without imposing any external constraints on where the actual center point location is. 

More concretely, running the above command will produce a directory at `[working_directory]` containing:
1. **a weighted average fit result file** called `[working_directory_name]_weighted_average_warp.csv`. This file contains the weighted average warping parameters and metadata details about the slides used. It is stored as a [`WarpingSummary` object](./utilities.py#L114-L132). This is the main output of the entire routine, and it can be used to apply corrections for warping in subsequent processing steps.
1. **weighted average warp field .bin files** called `dx_warp_field_[working_directory_name].bin` and `dy_warp_field_[working_directory_name].bin`.
1. **subdirectories for each of the three fit groups** called `warping_initial_pattern_50_octets`, `warping_center_principal_point_50_octets`, and `warping_final_pattern_100_octets`. Each of these subdirectories contains:
    - **a list of all the individual fit results** called `all_results_[subdirectory_name].csv` stored as [`WarpFitResult` objects](./utilities.py#L75-L100).
    - **a list of all the HPFs used** in the fits called `field_log_[subdirectory_name].csv` stored as [`FieldLog` objects](./utilities.py#L102-L106).
    - **a list of the octets used** in the fits called `[subdirectory_name]_overlap_octets.csv` stored as [`OverlapOctet` objects](./utilities.py#L38-L73).
    - **a visualization of the weighted average warping model** at this step called `warp_fields_[subdirectory_name].png`
    - **a subdirectory of plots** and text files called `batch_plots` containing several visualizations and details of the individual results in the group (plots of radial warping distortion parameters, fractional cost reductions, center principal point locations, etc.). These plots can all be remade if necessary using the `all_results_*.csv` file.
    - **subdirectories for each individual fit** in the group called `warping_batch_octets_[slide_ID]_[octet_center_rectangle_n]` which in turn contain:
        1. a field log file that is a portion of the consolidated `field_log_*.csv` file
        1. a metadata summary file called `metadata_summary_[dirname].csv` listing information about the HPFs used as [`MetadataSummary` objects](../../utilities/misc.py#L128-L135).
        1. several more visualizations of the individual fit result, the fit progression, and comparisons of the warped/unwarped raw/aligned octet overlays
    - For the "`initial_pattern`" and "`center_principal_point`" subdirectories, **weighted average fit result and warp field .bin** files like those detailed above, just at these intermediate steps.
1. **a .txt file of the commands run** for each of the three sets of fits, called `fit_group_commands.txt` (helpful in restarting a stalled run)
1. **lists of all valid octets found** for every slide, called `[slide_ID]_overlap_octets.csv`, stored as [`OverlapOctet` objects](./utilities.py#L38-L73).
1. **a main log file** in `[root_directory]/logfiles` called `warp_fit_layer_[layer_n].log` showing that the code was run
1. **more detailed sample log files** for every slide used in `[root_directory]/[slide_ID]/logfiles` called `[slide_ID]-warp_fit_layer_[layer_n].log` 
1. **an even more detailed "global" log file** called `global-warp_fit_layer_[layer_n].log` 

The background fraction allowed in valid overlaps, the numbers of octets used in each stage of the fitting procedure, and the maximum number of minimization iterations allowed at each stage of the fitting procedure are all user-configurable, but the default values have been found to produce consistent results. The user can also use command line options to specify which distortion parameters are fixed or floating in the fits, but the chosen model with fixed focal lengths and no tangential warping has again been found to produce consistent results.

