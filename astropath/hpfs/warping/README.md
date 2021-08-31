# Warping Corrections

The "warping" module contains code to measure the warping affecting raw data files in a cohort. To run it for an entire cohort in the most common use case, enter the following command and arguments:

`warpingcohort <Dpath>\<Dname> <Rpath> --flatfield_file [path_to_flatfield_bin_file]`

where `[path_to_flatfield_bin_file]` is the path to the ".bin" file specifying the flatfield corrections to apply, or the name of a file located in the `<Dpath>\<Dname>\Flatfield` directory

See [here](../../scans/docs/Definitions.md#43-definitions) for definitions of the terms in `<angle brackets>`.

Running the above command will produce:
1. **a new "Warping" directory** at `<Dpath>\<Dname>\Warping`
1. **a "weighted_average_warp.csv" file** in the Warping directory summarizing the final warping pattern determined by the fits, stored as a [WarpingSummary object](./utilities.py#L43-L60). This file can be used as an input to the image correction code to define the warping corrections that should be applied.
1. **an "octets" subdirectory** inside the "Warping" directory, with files listing the overlap octets found for each sample in the cohort and files listing the overlap octets chosen to run each of three stages of fits
1. three **fit result files**, one for each stage of the fit, listing all of the individual octet fit results found, stored as a datatable of [WarpFitResult objects](./utilities.py#L63-L83). The results in these files are used to define the initial parameter values and bounds for later stages of fits, or the final weighted average result.
1. three **field log files**, one for each stage of the fit, listing which raw images were used, stored as [FieldLog objects](./utilities.py#L86-L88).
1. three **metadata summary files**, one for each stage of the fit, summarizing the samples used in the fits including date ranges of image files, stored as [`MetadataSummary` objects](../../shared/samplemetadata.py#L95-L104)
1. **a main log file** called "`warping.log`" in `<Dpath>\<Dname>\logfiles` with just a single line showing that `warpingcohort` was run 
1. **more detailed sample log files** called "`<SlideID>-warping.log`" in `<Dpath>\<Dname>\<SlideID>\logfiles` for each slide used in fitting for the warping patterns.

Other options for how the correction should be done include:
1. Running only the octet finding portion of the code for all samples in the cohort: add the `--octets_only` flag. When running with this flag the octets for each sample will be found and written out and nothing else will happen. This is useful for running octet finding for subsets of the cohort samples in multiple processes instead of waiting for them all to happen sequentially.
1. Putting the output in a different location: add the `--workingdir [workingdir_path]` argument where `[workingdir_path]` is the path to the directory where the output should go
1. Skipping corrections for differences in exposure time: add the `--skip_exposure_time_corrections` argument
1. Using exposure time dark current offsets that are different from what's stored in each sample's Full.xml file: add the `--exposure_time_offset_file [path_to_exposure_time_offset_file]` argument where `[path_to_exposure_time_offset_file]` is the path to a .csv file holding a list of [`LayerOffset` objects](../../utilities/img_file_io.py#L30-L35)

To see more command line arguments available, run `warpingcohort --help`.
