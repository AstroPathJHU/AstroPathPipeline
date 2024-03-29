# 5.7.5. Additional Tools

Different coding tools that can be run outside of the *AstroPath Pipeline* are described here. Note that, the code still workers under the assumption that the ```<Mpath>``` exists with all its configuration files and the samples to be process are in the *AstroPath* format.

# 5.7.5.1. Instructions to Run Standalone via *AstroPath Pipeline* Workflow
The entire image correction workflow can be run for a single slide outside of the *AstroPath Pipeline* by running the following commands in Powershell:

```
Import-Module '*.\astropath'; launchmodule -slideid:<slideid> -mpath:<mpath> -module:'imagecorrection' -stringin:<Project>-<slideid>-<ProcessingLocation>
```
- replace '\*' with the location up to and including the *AstroPathPipeline* repository
- ```<SlideID>```: the names for the specimens in the astropath processing pipeline
- ```<mpath>```: the main path for all the astropath processing .csv configuration files; the current location of this path is *\\bki04\astropath_processing*
- ```<Project>```: Project Number
- ```<ProcessingLocation>```: The fully qualified path to a location where slides should be processed, use `'*'` if the slide should be processed in place

This workflow is described in more detail [here](OverviewWorkflow.md#576-overview-workflow "Title").

# 5.7.5.2. Instructions to Apply Image Correction Standalone via Python Package
The "applyflatw" portion of the code corrects raw ".Data.dat" files based on a given flatfield and warping model and writes out their contents, either overwriting the original raw image files, or as new ".fw" files. To run it for a single sample in the most common use case, enter the following command and arguments:

`applyflatwsample <Dpath>\<Dname> <SlideID> --shardedim3root <Rpath> --njobs [njobs]`

where `[njobs]` is the maximum number of parallel processes allowed to run at once (many parallel processes can be used; each process corrects and writes out one file at a time).

See [here](../../../scans/docs/Definitions.md#43-definitions) for definitions of the terms in `<angle brackets>`.

Successfully running the command above requires that the file at `\\bki04\astropath_processing\AstroPathCorrectionModels.csv` contain exactly one entry for the `<SlideID>` sample listing the flatfield version and the name of a warping file to use in correcting the image contents as a [`CorrectionModelTableEntry` object](../utilities.py#L4-L11). If the flatfield version and/or warping filename listed in the file for the `<SlideID>` sample are "none" then corrections of that type are NOT applied.

Running the above command will produce:
1. **corrected image files** that **overwrite** those in `<Dpath>\<Dname>\<SlideID>`
1. **a main log file** called "`applyflatw.log`" in `<Dpath>\<Dname>\logfiles` with just a single line showing that `applyflatwsample` was run 
1. **a more detailed sample log file** called "`<SlideID>-applyflatw.log`" in `<Dpath>\<Dname>\<SlideID>\logfiles`

Other options for how the correction should be done include:
1. Putting the output in a different location: add the `--workingdir [workingdir_path]` argument where `[workingdir_path]` is the path to the directory where the output should go. In this case the file extension is ".fw" and not the original raw image file extension.
1. Writing out corrected files for single image layers as well as multilayer images: add the `--layers [layers]` argument where `[layers]` is any number of arguments specifying which layer numbers to use (starting from 1). In the single image layer case, the corresponding output files are named ".fwxx" where "xx" is the two-digit layer number (i.e. ".fw01"). The special number -1 can also be given as an argument in `[layers]`, in which case the multilayer files will be written out in addition to any single layer files requested. Using this argument one could, for example, simultaneously write out the corrected multilayer .fw files and the corrected single layer .fw01 files. When using this argument, a `[workingdir_path]` must be specified.
1. Using different flatfield and/or warping corrections than those defined in the file at ``\\bki04\astropath_processing\AstroPathCorrectionModels.csv``: run with the `--flatfield-file [path_to_flatfield_bin_file]` and/or `--warping-file [path_to_warping_summary_csv_file]` arguments, where `[path_to_flatfield_bin_file]` is the path to the ".bin" file specifying the flatfield corrections to apply, or the name of a file located in the `<Dpath>\<Dname>\flatfield` or `\\bki04\astropath_processing\flatfield` directories, and `[path_to_warping_summary_csv_file]` is the path to a .csv file detailing the warping model parameters as a [`WarpingSummary` object](../../warping/utilities.py#L43-L61), or the name of a file in the same format located in the `<Dpath>\<Dname>\warping` or `\\bki04\astropath_processing\warping` directories. This warping .csv file can have several `WarpingSummary` entries specifying different warping patterns to apply in different raw image layers. 

The routine can be run for an entire cohort of samples at once using the following command:

`applyflatwcohort <Dpath>\<Dname> --shardedim3root <Rpath>`

To see more command line arguments available for both routines, run `applyflatwsample --help` or `applyflatwcohort --help`.

# 5.7.5.3. Instructions Apply Image Correction Standalone *version 0.0.1*

