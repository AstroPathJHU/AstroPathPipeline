# 5.8.5. Additional Tools
# Applying the Image Correction

The "image correction" portion of the code corrects raw ".Data.dat" files based on a given flatfield and warping model and writes out their contents as ".fw" files. To run it for a single sample in the most common use case, enter the following command and arguments:

`imagecorrectionsample <Dpath>\<Dname> <Rpath> <SlideID> --flatfield_file [path_to_flatfield_bin_file] --warping_file [path_to_warping_summary_csv_file] --njobs [njobs]`

where:
- `[path_to_flatfield_bin_file]` is the path to the ".bin" file specifying the flatfield corrections to apply, or the name of a file located in the `<Dpath>\<Dname>\Flatfield` directory
- `[path_to_warping_summary_csv_file]` is the path to a .csv file detailing the warping model parameters as a [`WarpingSummary` object](../warping/utilities.py#L43-L60). This file can have several `WarpingSummary` entries specifying different warping patterns to apply in differen raw image layers.
- `[njobs]` is the maximum number of parallel processes allowed to run at once (many parallel processes can be used; each process corrects and writes out one file at a time)

See [here](../../scans/docs/Definitions.md#43-definitions) for definitions of the terms in `<angle brackets>`.

Running the above command will produce:
1. **corrected ".fw" files** in `<Dpath>\flatw\<SlideID>`
1. **a main log file** called "`imagecorrection.log`" in `<Dpath>\<Dname>\logfiles` with just a single line showing that `imagecorrectionsample` was run 
1. **a more detailed sample log file** called "`<SlideID>-imagecorrection.log`" in `<Dpath>\<Dname>\<SlideID>\logfiles`

Other options for how the correction should be done include:
1. Putting the output in a different location: add the `--workingdir [workingdir_path]` argument where `[workingdir_path]` is the path to the directory where the output should go
1. Writing out corrected files for single image layers as well as multilayer images: add the `--layers [layers]` argument where `[layers]` is any number of arguments specifying which layer numbers to use (starting from 1). In the single image layer case, the corresponding output files are named ".fwxx" where "xx" is the two-digit layer number (i.e. ".fw01"). The special number -1 can also be given as an argument in `[layers]`, in which case the multilayer files will be written out in addition to any single layer files requested. Using this argument one could, for example, simultaneously write out the corrected multilayer .fw files and the corrected single layer .fw01 files.
1. Skipping the flatfield and/or warping corrections: run without specifying the `--flatfield_file` and/or `--warping_file` arguments

The image correction routine can be run for an entire cohort of samples at once using the following command:

`imagecorrectioncohort <Dpath>\<Dname> <Rpath>`

To see more command line arguments available for both routines, run `imagecorrectionsample --help` or `imagecorrectioncohort --help`.
