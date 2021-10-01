## 5.5.2. Mean Image
The `meanimage` submodule runs on raw (".Data.dat") image files. It finds the optimal background thresholds for a given sample, produces masks to remove empty background and blur/saturation artifacts from raw images, and stacks the masked images together to find an overall mean image for the sample with units of average counts/ms. This routine can be run for a single sample with the "[meanimagesample.py](./meanimagesample.py)" code, or for an entire cohort of samples using the "[meanimagecohort.py](./meanimagecohort.py)" code. 

## 5.5.2.1. Instructions to Run via AstroPath Pipeline Workflow

## 5.5.2.2. Instructions to Run Standalone

To run the routine for a single sample in the most common use case, enter the following command and arguments:

`meanimagesample <Dpath>\<Dname> <Rpath> <SlideID> --njobs [njobs]`

where `[njobs]` is the maximum number of parallel processes allowed to run at once during the parallelized portions of the code running

Running the above command will produce a "`meanimage`" directory in `<Dpath>\<Dname>\SlideID\im3` that contains the following:
1. **a `<SlideID>-mean_image.bin` file** that is the mean of the counts/ms in all of the selected HPFs' tissue regions, stored as 64-bit floats
1. **a `<SlideID>-std_error_of_mean_image.bin` file** that is the standard error on the mean counts/ms in all of the selected HPFs' tissue regions, stored as 64-bit floats
1. **a `<SlideID>-sum_images_squared.bin` file** that is the sum of the square of all the selected HPFs' tissue regions, stored as 64-bit floats. (This file is used to combine multiple samples' mean images into a flatfield model.)
1. **a `<SlideID>-mask_stack.bin` file** that is the stack of the binary image masks from every selected HPF, stored as 64-bit unsigned integers
1. **a list of every field used** in finding the background thresholds and in making the mean image, called "`fields_used.csv`", stored as [`FieldLog` objects](./utilities.py#L28-L34)
1. **an overview of the image files used to determine the background thresholds**, including the name of the miscroscope and date ranges, called "`<SlideID>-metadata_summary_thresholding_images.csv`", stored as [`MetadataSummary` objects](../../shared/samplemetadata.py#L96-L105)
1. **an overview of the image files stacked to create the mean image** in the same format as the above, called "`<SlideID>-metadata_summary_stacked_images.csv`"
1. **a "`<SlideID>_background_thresholds.txt`" file**, which lists the optimal background thresholds in each image layer, stored as [`ThresholdTableEntry` objects](utilities.py#L15-L19).
1. **a "`<SlideID>-thresholding_data_table.csv`" file**, which lists the background thresholds found for each individual tissue edge HPF image, stored as [`RectangleThresholdTableEntry` objects](./utilities.py#L21-L26)
1. Three summary .pdf files:
    - **`<SlideID>-thresholding_summary.pdf`** containing plots describing which images were used to find the optimal background thresholds, how the optimal background thresholds found vary as a function of image layer, and the distributions of the optimal thresholds found for each tissue edge HPF with signal vs. background histograms summed over all images used.
    - **`<SlideID>-masking_summary.pdf`** containing a plot describing the locations of HPFs that had blur and/or saturation artifacts masked out, and several sheets of example masking plots for the individual images that had the largest numbers of pixels masked out due to blur and/or saturation.
    - **`<SlideID>-meanimage_summary.pdf`** containing plots of each layer of the final mean image, the standard error on the mean image, and the mask stack, as quick reference for the contents of the .bin files listed above.
1. An `image_masking` subdirectory that contains the following:
    - "`[image_key]_tissue_mask.bin`" mask files for every HPF in the slide. These image masks are compressed to save space, and must be unpacked using [the "`ImageMask.unpack_tissue_mask`" function](../../shared/image_masking/image_mask.py#L150-L156) to work with them. That function takes as inputs the path to the mask file and the dimensions of the output mask (same as the original HPF layers, i.e. (1004,1344)), and returns a numpy array of zeroes and ones where 0 = background and 1 = tissue. There is only one tissue mask per HPF; it is the same for all image layers.
    - multilayer "`[image_key]_full_mask.bin`" mask files for any images that had blur or saturation flagged in them, stored as single column arrays of unsigned 64-bit integers. The first layer of these files contains a mask where 0=background, 1=good tissue, and anything >1 is flagged due to blur. The other layers contain the same mask as in the first layer, except potentially with additional regions with values >1 showing where saturation is flagged in each layer group. The blur masks are the same for every image layer, but saturation is flagged independently for each layer group.
    - a "`labelled_mask_regions.csv`" file listing every region masked due to blur or saturation as [`LabelledMaskRegion` objects](../../shared/image_masking/utilities.py#L7-L13). This file can be used to understand the contents of the `*_full_mask.bin` files: any region with any index >1 in a `*_full_mask.bin` file has a corresponding line in the `labelled_mask_regions.csv` file stating which layers it should be flagged in, the size of the region, and why it was flagged.
1. **a main log file** called "`meanimage.log`" in `<Dpath>\<Dname>\logfiles` with just a single line showing that meanimage was run
1. **a more detailed sample log file** called "`<SlideID>-meanimage.log`" in `<Dpath>\<Dname>\<SlideID>\logfiles`

Other options for running the code include:
- skipping corrections for differences in exposure time: add the `--skip_exposure_time_corrections` argument
- using exposure time dark current offsets that are different from what's stored in the sample's Full.xml file: add the `--exposure_time_offset_file [path_to_exposure_time_offset_file]` argument where `[path_to_exposure_time_offset_file]` is the path to a .csv file holding a list of [`LayerOffset` objects](../../utilities/img_file_io.py#L21-L26)
- skipping determining background thresholds and creating masks: add the `--skip_masking` flag 
- changing the output location: add the `--workingdir [workingdir_path]` argument where `[workingdir_path]` is the path to the directory where the output should go (the default is `<Dpath>\<Dname>\SlideID\im3\meanimage` as detailed above)
- using pre-created mask/threshold files: If the routine has already been run and background thresholds and/or masking files have already been created in the expected location, those portions of the code will not be run again unless the existing files are deleted. If you would like to force recreation of the files, add the `--maskroot [mask_root_path]` argument, where `[mask_root_path]` is a path to a directory other than `<Dpath>\<Dname>`. This same argument can be used to reference pre-created threshold/masking files in any other location as well, and the sample log file will list details of which subroutines have been skipped or run and where the data they're using are coming from. 

The meanimage routine can be run for an entire cohort of samples at once using the command:

`meanimagecohort <Dpath>\<Dname> <Rpath> --exposure_time_offset_file [path_to_exposure_time_offset_file] --njobs [njobs]`

where the arguments are the same as those listed above for `meanimagesample`. To see more command line arguments available for both routines, run `meanimagesample --help` or `meanimagecohort --help`.

## 5.5.2.3. Instructions to Run Standalone Version *0.0.1*
After the intial AstroPath publication, significant changes to the mean image processing took place. To keep the code backward compatible both the older version of the code and newer version to apply the mean image are housed here. The older version of the code is run via the pipeline in matlab by specifying a version number of *0.0.1* in the [*AstroPathConfig.csv*](../../scans/docs/AstroPathProcessingDirectoryandInitializingProjects.md#451-astropath_processing-directory). The older version of the code has significant drawbacks and should only be used for backwards compatibility. We also include the matlab version as a standalone tool that can be run as follows. 

Download the repository to a working location. Open a new session of matlab and add the ```AstroPathPipline``` to the matlab path. Then use the following to launch:
   ``` meanimages(<basepath>, <slideid>) ``` 

Output files for this version are:
   - ```<meanimage-output-flt>```: the output flt file should be named *```<SlideID>```-mean.flt*. This file contains the *total* image for all im3s in a slide in as a single column vector. The order is layer, width, height. 
   - ```<meanimage-output-csv>```: the output csv file should be named *```<SlideID>```-mean.csv*. This file contains four numbers: the number of images used, number of image layers, image width, and image height.
