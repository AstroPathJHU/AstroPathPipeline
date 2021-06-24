# Flatfielding

The "flatfielding" portion of the code provides routines to determine background thresholds, make mean tissue images for individual slides or slide groups, make flatfield correction factor images from batches of slides, and to apply a previously-run flatfield model to an orthogonal set of images. The two most relevant use cases in the context of production are: making mean images from individual slides (using the `slide_mean_image` run mode), and combining those mean images, with their stacks of tissue masks, into a flatfield correction model for a batch of slides (using the `batch_flatfield`) mode.

## Making mean images for individual slides

The `slide_mean_image` run mode finds the background flux thresholds for a given slide, uses those thresholds to mask background out of HPFs, and stacks the tissue regions, normalized by exposure time, to create a mean image that can later be combined with those of other slides to make the flatfield model for a batch of slides. To run it in the most common use case, enter the following command and arguments:

`run_flatfield slide_mean_image --slides [slide_ID] --rawfile_top_dir [rawfile_directory] --root_dir [root_directory] --exposure_time_offset_file [path_to_exposure_time_offset_file] --n_threads [n_threads]`

where:
- `[slide_ID]` is the name of the slide whose meanimage should be created (i.e. "`M21_1`")
- `[rawfile_directory]` is the path to a directory containing a `[slide_ID]` subdirectory with the slide's ".Data.dat" files (i.e. ``\\bki07\dat``)
- `[root_directory]` is the path to the usual "Clinical_Specimen" directory containing a `[slide_ID]` subdirectory (i.e. `\\bki02\E\Clinical_Specimen`)
- `[path_to_exposure_time_offset_file]` is the path to a .csv file holding the exposure time correction "dark current" offsets as a list of [`LayerOffset` objects](../../utilities/img_file_io.py#L31-L36), which is output by the code that does the exposure time correction fits
- `[n_threads]` is the integer number of threads to use in the portions of the code that are parallelized, namely, reading the raw HPF images to get their histograms for finding background thresholds or to stack them, and finding the image masks for each HPF. Neither of those processes is particularly CPU-intensive (and determining the image masks explicitly occurs on the GPU, also) so many threads can be used at once. I have run this code with up to 64 threads on BKI06 without taxing CPU or memory hardly at all; you may want to find a number that works well to balance the overhead of copying the data back from the subprocesses. The default is 10.

Running the above command will produce:
1. **a "`meanimage`" directory** in `[root_directory]\[slide_ID]\im3` that contains the following:
    - **a `[slideID]-mean_image.bin` file** that is the mean of the counts/ms in all of the selected HPFs' tissue regions, stored as 64-bit floats
    - **a `[slideID]-std_error_of_mean_image.bin` file** that is the standard error on the mean counts/ms in all of the selected HPFs' tissue regions, stored as 64-bit floats
    - **a `[slideID]-mask_stack.bin` file** that is the stack of the binary image masks from every selected HPF, stored as 64-bit unsigned integers
    - **a very detailed "global" log file** called "`global-slide_mean_image.log`"
    - **a list of every field used** in finding the background thresholds and in making the mean image, called "`fields_used_meanimage.csv`", stored as [`FieldLog` objects](./utilities.py#L23-L29)
    - **a list of every slide used** in stacking images, including date ranges, called "`metadata_summary_stacked_images_meanimage.csv`" and stored as [`MetadataSummary` objects](../../utilities/misc.py#L123-L130)
    - **a "`thresholding_info`" subdirectory** containing plots/details about how the background thresholding proceeded. Most important in this subdirectory is the `[slideID]_background_thresholds.txt` file, which lists the background thresholds in each image layer (after exposure time corrections are applied).
    - **a "`postrun_info`" subdirectory** containing information about how many HPFs were stacked in each layer of the meanimage, how many raw HPFs were read, and a subdirectory of .png images of the individual mean image, mask stack, and standard error of the mean image layers.
    - **a "`image_masking`" subdirectory** containing:
        1. "`[image_key]_tissue_mask.bin`" mask files for every HPF in the slide. These image masks are compressed to save space, and must be unpacked using [the "`unpackTissueMask`" function](../image_masking/utilities.py#L15-L20) to work with them. That function takes as inputs the path to the mask file and the dimensions of the output mask (same as the original HPF layers, i.e. (1004,1344)), and returns a numpy array of zeroes and ones where 0 = background and 1 = tissue. There is only one tissue mask per HPF; it is the same for all image layers.
        1. multilayer "`[image_key]_full_mask.bin`" mask files for any stacked images that had blur or saturation flagged in them. The first layer of these files contains a mask where 0=background, 1=good tissue, and anything >1 is flagged due to blur. The other images layers contain the same mask as in the first layer, except potentially with additional regions >1 showing where saturation is flagged in each layer group. The blur masks are the same for every image layer, but saturation is flagged independently for each layer group.
        1. a "`labelled_mask_regions.csv`" file listing every region masked due to blur or saturation as [`LabelledMaskRegion` objects](../image_masking/utilities.py#L7-L13). This file can be used to understand the contents of the `*_full_mask.bin` files: any region with any index >1 in a `*_full_mask.bin` file has a corresponding line in the `labelled_mask_regions.csv` file stating which layers it should be flagged in, the size of the region, why it was flagged, etc.
        1. a useful plot called `"[slideID]_flagged_HPF_locations.png"` showing which HPFs within the slide were flagged for blur, saturation, or both, and where those HPFs are located spatially within the slide.
1. **a main log file** called "`slide_mean_image.log`" in `[root_directory]\logfiles` with just a single line showing that slide_mean_image was run
1. **a more detailed sample log file** called "`[slideID]-slide_mean_image.log`" in `[root_directory]\[slide_ID]\logfiles`

The number of threads is currently the ONLY option the user can change, in order to ensure consistency between mean images created for different slides.

## Combining slide mean images into a flatfield model for a batch

The `batch_flatfield` mode reads the `[slideID]-mean_image.bin` and `[slideID]-mask_stack.bin` files created for a batch of slides and combines them to produce a flatfield correction model. To run it in the most common use case, after running `slide_mean_image` for a batch of slides, enter the following command and arguments :

`run_flatfield batch_flatfield --slides [comma_separated_list_of_slide_IDs] --rawfile_top_dir [rawfile_directory] --root_dir [root_directory] --batchID [batch_ID]`

where:
- `[comma_separated_list_of_slide_IDs]` is a comma-separated list of slide IDs whose mean images should be combined (i.e. "`M107_1,M109_1,M110_1,M111_1,M112_1`")
- `[rawfile_directory]` and `[root_directory]` are the same as in the previous run mode
- `[batch_ID]` is the integer identifier of which batch the group of slides represents (i.e. "14"). Currently only IDs between 0 and 99 are supported, since the IDs are written out as zero-padded two-digit numbers in several output contexts.

Running the above command will produce:
1. **a flatfield correction model .bin file** called `flatfield_BatchID_[batch_ID].bin` in `[root_directory]\flatfield`
2. **a main log file** called "`batch_flatfield.log`" in `[root_directory]\logfiles` with just a single line showing that batch_flatfield was run 
3. **more detailed sample log files** called "`[slide_ID]-batch_flatfield.log`" in `[root_directory]\[slide_ID]\logfiles` for each `[slide_ID]` in `[comma_separated_list_of_slide_IDs]`
4. **a `flatfield_BatchID_[batch_ID]` subdirectory** in `[root_directory]\flatfield` that contains the following:
    - **a summary PDF file** called `flatfield_BatchID_[batch_ID]_summary.pdf` that shows the layers of the flatfield image, the relative spread in the correction factors in each layer, how many images were stacked from all slides in each layer, and the layers of the combined mask stack in one document for reference
    - **a very detailed "global" log file** called "`global-batch_flatfield.log`"
    - **field log** and **metadata summary** files like in the above run mode, combined for every slide in the batch
    - **a "`postrun_info`" subdirectory** containing similar low-level info to that of the previous run mode

There are currently NO OPTIONS for the user to change in this run mode, again to ensure consistency between the mean images used in making the flatfield model. Also please note that producing the summary PDF file requires running on a system that recognizes `pdflatex` as a command. If the runtime environment doesn't have LaTeX installed (along with the `graphicx` and `geometry` packages), the template .tex file for the PDF is still created but the output PDF is not, and a low-level warning is output to the log files.
