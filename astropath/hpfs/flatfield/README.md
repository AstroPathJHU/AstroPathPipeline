# 5.5. Flatfield

## 5.5.1. Description
The `flatfield` subworkflow provides routines to create meanimages for single samples, find the sets of samples that should be combined into flatfield correction models (`meanimagecomparison`), and create those flatfield correction models (`batchflatfield`). It also provides some code to double-check the efficacy of applying the determined flatfield corrections within a cohort. 

See [here](../../scans/docs/Definitions.md#43-definitions) for definitions of the terms in `<angle brackets>`.



## 5.5.3. Mean Image Comparison

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
1. **a list of every field used** in finding the background thresholds and in making the flatfield model, called "`fields_used.csv`", stored as [`FieldLog` objects](./utilities.py#L28-L34). This is a combination of the files with the same name created for all the samples used.
1. **an overview of the image files used to create the flatfield model** in the same "metadata summary" format as the above, called "`metadata_summary_stacked_images.csv`"
1. **a summary .pdf file** called `flatfield_summary_BatchID_[batch_ID].pdf` containing a plot of the flatfield correction factor distributions as a function of image layer, a datatable with some statistics about the same, and plots of the correction factors, their uncertainties, and the combined stack of image masks in every layer to serve as a quick reference for the contents of the outputted `.bin` files.

To see more command line arguments available, run `batchflatfieldcohort --help`.

## 5.5.4. Batch Flatfield

One last routine in this portion of the code can be used to test the effect of applying a set of flatfield corrections. The "[appliedflatfieldcohort.py](./appliedflatfieldcohort.py)" code splits the images in each slide in a given set randomly into two equally-sized subsamples. It creates a mean image from each subsample using the same methods as in `meanimagesample`, and uses one subsample to calculate a set of flatfield correction factors to apply to the mean image created using the other subsample. It outputs the test flatfield model and the corrected mean image. To run the code in the most common use case, enter the following command and arguments:

`appliedflatfieldcohort <Dpath>\<Dname> <Rpath> [workingdir_path]`

where `[workingdir_path]` is the path to the directory where the output should be located

Running the above command will create a new directory at `[workingdir_path]` that contains the following:
1. **a `flatfield.bin` file** containing the test flatfield correction model
1. **a `flatfield_uncertainty.bin` file** containing the uncertainties on the test corrections
1. **a `corrected_mean_image.bin` file** containing the mean image computed using the orthogonal subsample of images, after application of the flatfield correction factors in `flatfield.bin`
1. **a `corrected_mean_image_uncertainty.bin` file** containing the uncertainties on the above `corrected_mean_image`
1. **a `corrected_mean_image_mask_stack.bin` file** containing the stack of all image masks used to create the corrected mean image
1. **lists of every field used** in finding the corrected mean image and test flatfield model, called "`fields_used_corrected_mean_image.csv`" and "`fields_used_flatfield.csv`", respectively, stored as [`FieldLog` objects](./utilities.py#L28-L34).
1. **overviews of the image files used to create the corrected mean image and test flatfield model** in the same "metadata summary" format as the above, called "`metadata_summary_corrected_mean_image_stacked_images.csv`" and "`metadata_summary_flatfield_stacked_images.csv`", respectively.
1. **a summary .pdf file** called `applied_flatfield_summary.pdf` containing plots of each layer of the mean image before application of the correction factors along with their uncertainties and the layers of the mask stack, a plot and datatable detailing the distribution of the flatfield correction factors, plots of the layers of the flatfield correction image and their uncertainties, layers of the corrected mean image with its uncertainties, plots of the pre- and post-correction mean-relative pixel intensities in each layer, plots of the change in illumination variation in each image layer caused by applying the corrections, and a datatable summarizing the effects of applying the corrections. 

To see more command line arguments available, run `appliedflatfieldcohort --help`.

## Note

Producing the summary PDF files requires running on a system that recognizes `pdflatex` as a command. If the runtime environment doesn't have LaTeX installed (along with the `graphicx` and `geometry` packages), the template .tex files for the PDFs described above are still created but the output PDFs are not, and a low-level warning is output to the log files. The plots that would be in the summary PDFs will exist in the output directories as individual image files, and the .tex file can be compiled at a later time.

