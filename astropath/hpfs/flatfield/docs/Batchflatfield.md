# 5.5.5. Batch Flatfield

## 5.5.5.1. Instructions to Run via *AstroPath Pipeline* Workflow


## 5.5.5.2. Instructions to Run Standalone Via Python Package

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

# 5.5.6. Batch Flatfield Tests

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
