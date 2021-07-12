# Flatfielding

The "flatfielding" portion of the code provides routines to create meanimages for single samples, find the sets of samples that should be combined into flatfield correction models, and create those flatfield correction models. It also provides some code to double-check the efficacy of applying the determined flatfield corrections within a cohort. 

## The meanimage routine

The "meanimage" routine runs on raw (".Data.dat") image files. It finds the optimal background thresholds for a given sample, produces masks to remove empty background and blur/saturation artifacts from raw images, and stacks the masked images together to find an overall mean image for the sample with units of average counts/ms. This routine can be run for a single sample with the "[meanimagesample.py](./meanimagesample.py)" code, or for an entire cohort of samples using the "[meanimagecohort.py](./meanimagecohort.py)" code. 

To run the routine for a single sample in the most common use case, enter the following command and arguments:

`meanimagesample <Dpath>\<Dname> <Rpath> <SlideID> --exposure_time_offset_file [path_to_exposure_time_offset_file] --njobs [njobs]`

where:
- `[path_to_exposure_time_offset_file]` is the path to a .csv file holding the exposure time correction "dark current" offsets as a list of [`LayerOffset` objects](../../utilities/img_file_io.py#L30-L35), which is output by the code that does the exposure time correction fits.
- `[njobs]` is the maximum number of parallel processes allowed to run at once during the parallelized portions of the code running
See [here](../../scans/docs/Definitions.md#43-definitions) for definitions of the terms in `<angle brackets>`.

Running the above command will produce a "`meanimage`" directory in `<Dpath>\<Dname>\SlideID\im3` that contains the following:
1. **a `<SlideID>-mean_image.bin` file** that is the mean of the counts/ms in all of the selected HPFs' tissue regions, stored as 64-bit floats
1. **a `<SlideID>-std_error_of_mean_image.bin` file** that is the standard error on the mean counts/ms in all of the selected HPFs' tissue regions, stored as 64-bit floats
1. **a `<SlideID>-sum_images_squared.bin` file** that is the sum of the square of all the selected HPFs' tissue regions, stored as 64-bit floats. (This file is used to combine multiple samples' mean images into a flatfield model.)
1. **a `<SlideID>-mask_stack.bin` file** that is the stack of the binary image masks from every selected HPF, stored as 64-bit unsigned integers
1. **a list of every field used** in finding the background thresholds and in making the mean image, called "`fields_used.csv`", stored as [`FieldLog` objects](./utilities.py#L22-L28)
1. **an overview of the image files used to determine the background thresholds**, including the name of the miscroscope and date ranges, called "`<SlideID>-metadata_summary_thresholding_images.csv`", stored as [`MetadataSummary` objects](../../utilities/misc.py#L145-L152)
1. **an overview of the image files stacked to create the mean image** in the same format as the above, called "`<SlideID>-metadata_summary_stacked_images.csv`"
1. **a "`<SlideID>_background_thresholds.txt`" file**, which lists the optimal background thresholds in each image layer, stored as [`ThresholdTableEntry` objects](../../utilities/misc.py#L154-L158).
1. **a "`<SlideID>-thresholding_data_table.csv`" file**, which lists the background thresholds found for each individual tissue edge HPF image, stored as [`RectangleThresholdTableEntry` objects](./utilities.py#L15-L20)
1. Three summary .pdf files:
    - **`<SlideID>-thresholding_summary.pdf`** containing plots describing which images were used to find the optimal background thresholds, how the optimal background thresholds found vary as a function of image layer, and the distributions of the optimal thresholds found for each tissue edge HPF with signal vs. background histograms summed over all images used.
    - **`<SlideID>-masking_summary.pdf`** containing a plot describing the locations of HPFs that had blur and/or saturation artifacts masked out, and several sheets of example masking plots for the individual images that had the largest numbers of pixels masked out due to blur and/or saturation.
    - **`<SlideID>-meanimage_summary.pdf`** containing plots of each layer of the final mean image, the standard error on the mean image, and the mask stack, as quick reference for the contents of the .bin files listed above.
1. An `image_masking` subdirectory that contains the following:
    - "`[image_key]_tissue_mask.bin`" mask files for every HPF in the slide. These image masks are compressed to save space, and must be unpacked using [the "`ImageMask.unpack_tissue_mask`" function](../image_masking/image_mask.py#L150-L156) to work with them. That function takes as inputs the path to the mask file and the dimensions of the output mask (same as the original HPF layers, i.e. (1004,1344)), and returns a numpy array of zeroes and ones where 0 = background and 1 = tissue. There is only one tissue mask per HPF; it is the same for all image layers.
    - multilayer "`[image_key]_full_mask.bin`" mask files for any images that had blur or saturation flagged in them, stored as single column arrays of unsigned 64-bit integers. The first layer of these files contains a mask where 0=background, 1=good tissue, and anything >1 is flagged due to blur. The other layers contain the same mask as in the first layer, except potentially with additional regions with values >1 showing where saturation is flagged in each layer group. The blur masks are the same for every image layer, but saturation is flagged independently for each layer group.
    - a "`labelled_mask_regions.csv`" file listing every region masked due to blur or saturation as [`LabelledMaskRegion` objects](../image_masking/utilities.py#L7-L13). This file can be used to understand the contents of the `*_full_mask.bin` files: any region with any index >1 in a `*_full_mask.bin` file has a corresponding line in the `labelled_mask_regions.csv` file stating which layers it should be flagged in, the size of the region, and why it was flagged.
1. **a main log file** called "`meanimage.log`" in `<Dpath>\<Dname>\logfiles` with just a single line showing that meanimage was run
1. **a more detailed sample log file** called "`<SlideID>-meanimage.log`" in `<Dpath>\<Dname>\<SlideID>\logfiles`

Other options for running the code include:
- skipping corrections for differences in exposure time: replace the `--exposure_time_offset_file` argument with the `--skip_exposure_time_correction` flag
- skipping determining background thresholds and creating masks: add the `--skip_masking` flag 
- changing the output location: add the `--workingdir [workingdir_path]` argument where `[workingdir_path]` is the path to the directory where the output should go (the default is `<Dpath>\<Dname>\SlideID\im3\meanimage` as detailed above)
- using pre-created mask/threshold files: If the routine has already been run and background thresholds and/or masking files have already been created in the expected location, those portions of the code will not be run again unless the existing files are deleted. If you would like to force recreation of the files, add the `--maskroot [mask_root_path]` argument, where `[mask_root_path]` is a path to a directory other than `<Dpath>\<Dname>`. This same argument can be used to reference pre-created threshold/masking files in any other location as well, and the sample log file will list details of which subroutines have been skipped or run and where the data they're using are coming from. 

The meanimage routine can be run for an entire cohort of samples at once using the command:

`meanimagecohort <Dpath>\<Dname> <Rpath> --exposure_time_offset_file [path_to_exposure_time_offset_file] --njobs [njobs]`

where the arguments are the same as those listed above for `meanimagesample`. To see more command line arguments available for both routines, run `meanimagesample --help` or `meanimagecohort --help`.

## Finding the set of samples to combine into a flatfield model

After running the meanimage routine for a whole cohort, the set of samples whose mean images should be used to determine a single flatfield correction model can be found using the plot(s)/datatable(s) created by the "[meanimage_comparison_plot.py](./meanimage_comparison_plot.py)" script. The comparison between any two samples' mean images is determined using the standard deviation of the distribution of the pixel-wise differences between the two mean images divided by their uncertainties. This comparison statistic is calculated for every image layer and every pair of samples, and the average over all image layers is plotted for each pair in a grid. The resulting plot shows values near one for samples whose mean images are comparable, and values far from one for samples whose mean images are very different. The plot can be run several times (more quickly, after the initial data table is produced) to find the best grouping of slides to use for a single flatfield model. It can also be used to check a new cohort of samples' mean images against previously-run cohorts. To run the script in the most common use case, enter the following command and arguments:

`meanimagecomparison --root_dirs [Dpaths] --workingdir [workingdir_path]`

where:
- `[Dpaths]` is a comma-separated list of `<Dpath>\<Dname>` directory paths whose cohorts' mean images should be compared
- `[workingdir_path]` is the path to the directory where the output should be located

Running the above command will create a new directory at `[workingdir_path]` that contains the following:
1. **a `meanimage_comparison_table.csv` file** whose entries list the comparisons between each layer of every pair of slide mean images, stored as [`TableEntry` objects](./meanimage_comparison_plot.py#L27-L34). This datatable can be used to recreate the plot with different specifications by giving the path to it as `[input_file_path]` in the argument `--input_file [input_file_path]`.
1. **a `meanimage_comparison_average_over_all_layers.png` plot** showing a grid of comparison statistic values for each combination of slides in the inputted cohort(s)
1. **a log file** called `meanimage_comparison.log`

Other options for running the code include:
- Running from a previously-created input file: use the `--input_file [input_file_path]` argument as described above; this allows remaking the plot with a subset of slides without needing to recalculate all of the comparison statistic values
- Skipping plotting certain slides: add the `--skip_slides [slides_to_skip]` argument where `[slides_to_skip]` is a comma-separated list of `<SlideID>`s that should be excluded
- Reordering the slides within the grid plot: use the `--sort_by` argument to sort slides either by the order in which they are listed in the cohort(s)'s sampledef.csv file(s) (`--sort_by order`) or by the project number, then by cohort, and then by batch number (`--sort_by project_cohort_batch`). (The latter is the default.)
- Changing where the lines are on the plot: By default, there are dividing lines on the plot between every batch of slides. You can put these dividing lines in different locations by adding the `--lines_after [slide_ids]` argument, where `[slide_ids]` is a comma-separated list of `<SlideID>`s below which the dividing lines should appear.
- Changing the limits on the z-axis of the plot: Add the `--bounds [bounds]` argument where `[bounds]` is two values for the lower and upper bound of the z-axis. The default value for this argument is `0.85,1.15`.
- Save plots for every layer individually instead of the average over all layers: add the `--save_all_layers` flag

## Creating a flatfield model from a group of slides' mean image files

After the mean images for each slide have been run and a suitable set of slides has been determined, the mean images from all of those slides are combined together to produce a single flatfield correction model using the "[batchflatfieldcohort.py](./batchflatfieldcohort.py)" code. To run this code in the most common use case, enter the following command and arguments:

`batchflatfieldcohort <Dpath>\<Dname> <Rpath> --sampleregex [sample_regex] --batchID [batch_ID]`

where:
- `[sample_regex]` is a regular expression that matches all of the `<SlideID>`s in the cohort that you'd like to combine
- `[batch_ID]` is an integer between 0 and 99 representing the unique batch ID code for the created flatfield model

Running the above command will create a `flatfield_BatchID_[batch_ID].bin` file, and a new directory called `flatfield_BatchID_[batch_ID]`, both in `<Dpath>\<Dname>\Flatfield`. The file will contain the flatfield correction model itself, and the new directory will contain the following:
1. **a `flatfield_BatchID_[batch_ID]_uncertainty.bin` file** that contains the uncertainties on the flatfield correction factors that were calculated
1. **a list of every field used** in finding the background thresholds and in making the flatfield model, called "`fields_used.csv`", stored as [`FieldLog` objects](./utilities.py#L22-L28). This is a combination of the files with the same name created for all the samples used.
1. **an overview of the image files used to create the flatfield model** in the same "metadata summary" format as the above, called "`metadata_summary_stacked_images.csv`"
1. **a summary .pdf file** called `flatfield_summary_BatchID_[batch_ID].pdf` containing a plot of the flatfield correction factor distributions as a function of image layer, a datatable with some statistics about the same, and plots of the correction factors, their uncertainties, and the combined stack of image masks in every layer to serve as a quick reference for the contents of the outputted `.bin` files.

To see more command line arguments available, run `batchflatfieldcohort --help`.

## Testing the effect of applying a set of flatfield corrections

One last routine in this portion of the code can be used to test the effect of applying a set of flatfield corrections. The "[appliedflatfieldcohort.py](./appliedflatfieldcohort.py)" code splits the images in each slide in a given set randomly into two equally-sized subsamples. It creates a mean image from each subsample using the same methods as in `meanimagesample`, and uses one subsample to calculate a set of flatfield correction factors to apply to the mean image created using the other subsample. It outputs the test flatfield model and the corrected mean image. To run the code in the most common use case, enter the following command and arguments:

`appliedflatfieldcohort <Dpath>\<Dname> <Rpath> [workingdir_path] --exposure_time_offset_file [path_to_exposure_time_offset_file]`

where:
- `[workingdir_path]` is the path to the directory where the output should be located
- `[path_to_exposure_time_offset_file]` is the path to a .csv file holding the exposure time correction "dark current" offsets as a list of [`LayerOffset` objects](../../utilities/img_file_io.py#L30-L35), like above.

Running the above command will create a new directory at `[workingdir_path]` that contains the following:
1. **a `flatfield.bin` file** containing the test flatfield correction model
1. **a `flatfield_uncertainty.bin` file** containing the uncertainties on the test corrections
1. **a `corrected_mean_image.bin` file** containing the mean image computed using the orthogonal subsample of images, after application of the flatfield correction factors in `flatfield.bin`
1. **a `corrected_mean_image_uncertainty.bin` file** containing the uncertainties on the above `corrected_mean_image`
1. **a `corrected_mean_image_mask_stack.bin` file** containing the stack of all image masks used to create the corrected mean image
1. **lists of every field used** in finding the corrected mean image and test flatfield model, called "`fields_used_corrected_mean_image.csv`" and "`fields_used_flatfield.csv`", respectively, stored as [`FieldLog` objects](./utilities.py#L22-L28).
1. **overviews of the image files used to create the corrected mean image and test flatfield model** in the same "metadata summary" format as the above, called "`metadata_summary_corrected_mean_image_stacked_images.csv`" and "`metadata_summary_flatfield_stacked_images.csv`", respectively.
1. **a summary .pdf file** called `applied_flatfield_summary.pdf` containing plots of each layer of the mean image before application of the correction factors along with their uncertainties and the layers of the mask stack, a plot and datatable detailing the distribution of the flatfield correction factors, plots of the layers of the flatfield correction image and their uncertainties, layers of the corrected mean image with its uncertainties, plots of the pre- and post-correction mean-relative pixel intensities in each layer, plots of the change in illumination variation in each image layer caused by applying the corrections, and a datatable summarizing the effects of applying the corrections. 

To see more command line arguments available, run `appliedflatfieldcohort --help`.

## Note

Producing the summary PDF file requires running on a system that recognizes `pdflatex` as a command. If the runtime environment doesn't have LaTeX installed (along with the `graphicx` and `geometry` packages), the template .tex files for the PDFs described above are still created but the output PDFs are not, and a low-level warning is output to the log files. The plots that would be in the summary PDFs will exist in the output directories as individual image files, and the .tex file can be compiled at a later time.

