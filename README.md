# Introduction

This is the main repository for the Astropath group's code that has been ported to Python. 

## Installation

To install the code, first check out the repository, enter its directory, and run
```bash
pip install .
```
If you want to continue developing the code after installing, run instead
```bash
pip install --editable .
```

Once the code is installed using either of those lines, you can run
```python
import astropath
```
from any directory.

## Dependencies

In addition to 64-bit Python 3, The following packages are required to run code in this repository:
- NumPy  ([homepage](https://numpy.org/))
- SciPy ([homepage](https://www.scipy.org/))
- scikit-learn ([homepage](https://scikit-learn.org/stable/))
- scikit-image ([homepage](https://scikit-image.org/))
- opencv-python ([homepage](https://opencv.org/), [python version PyPi page](https://pypi.org/project/opencv-python/))
- Numba ([homepage](https://numba.pydata.org/))
- PyOpenCL ([homepage](https://documen.tician.de/pyopencl/)) 
- Reikna ([homepage](http://reikna.publicfields.net/en/latest/))
- cvxpy ([homepage](https://www.cvxpy.org/))
- Matplotlib ([homepage](https://matplotlib.org/))
- seaborn ([homepage](https://seaborn.pydata.org/))
- pyvips ([homepage](https://libvips.github.io/libvips/), [python binding homepage](https://libvips.github.io/pyvips/intro.html))
- imagecodecs ([PyPi page](https://pypi.org/project/imagecodecs/))
- NetworkX ([homepage](https://networkx.org/)) 
- uncertainties ([homepage](https://uncertainties-python-package.readthedocs.io/en/latest/))
- methodtools ([PyPi page](https://pypi.org/project/methodtools/))
- more_itertools ([homepage](https://more-itertools.readthedocs.io/en/stable/))
- jxmlease ([GitHub page](https://github.com/Juniper/jxmlease))

These packages will all be installed automatically when you install this repository.

A testing environment exists for this code, with a corresponding minimal environment inside a Docker container. The current environment setup for that container can be found in a sister repository [here](https://github.com/AstroPathJHU/astropathtest/blob/master/Dockerfile).

# Running the code

The code in this repository serves several functions, and most of them are still in development. Below we detail how to run the portions of the code that are more static.

## Flatfielding

The "flatfielding" portion of the code provides routines to determine background thresholds, make mean tissue images for individual slides or slide groups, make flatfield correction factor images from batches of slides, and to apply a previously-run flatfield model to an orthogonal set of images. The two most relevant use cases in the context of production are: making mean images from individual slides (using the `slide_mean_image` run mode), and combining those mean images, with their stacks of tissue masks, into a flatfield correction model for a batch of slides (using the `batch_flatfield`) mode.

### Making mean images for individual slides

The `slide_mean_image` run mode finds the background flux thresholds for a given slide, uses those thresholds to mask background out of HPFs, and stacks the tissue regions, normalized by exposure time, to create a mean image that can later be combined with those of other slides to make the flatfield model for a batch of slides. To run it in the most common use case, enter the following command and arguments:

`run_flatfield slide_mean_image --slides [slide_ID] --rawfile_top_dir [rawfile_directory] --root_dir [root_directory] --exposure_time_offset_file [path_to_exposure_time_offset_file] --n_threads [n_threads]`

where:
- `[slide_ID]` is the name of the slide whose meanimage should be created (i.e. "`M21_1`")
- `[rawfile_directory]` is the path to a directory containing a `[slide_ID]` subdirectory with the slide's ".Data.dat" files (i.e. ``\\bki07\dat``)
- `[root_directory]` is the path to the usual "Clinical_Specimen" directory containing a `[slide_ID]` subdirectory (i.e. `\\bki02\E\Clinical_Specimen`)
- `[path_to_exposure_time_offset_file]` is the path to a .csv file holding the exposure time correction "dark current" offsets as a ["LayerOffset" object](https://github.com/AstroPathJHU/AstroPathPipeline/blob/master/astropath/utilities/img_file_io.py#L28-L34) (i.e. the file [here](https://github.com/AstroPathJHU/alignmentjenkinsdata/blob/master/corrections/best_exposure_time_offsets_Vectra_9_8_2020.csv)), which is output by the code that does the exposure time correction fits
- `[n_threads]` is the integer number of threads to use in the portions of the code that are parallelized, namely, reading the raw HPF images to get their histograms for finding background thresholds or to stack them, and finding the image masks for each HPF. Neither of those processes is particularly CPU-intensive (and determining the image masks explicitly occurs on the GPU, also) so many threads can be used at once. I have run this code with up to 64 threads on BKI06 without taxing CPU or memory hardly at all; you may want to find a number that works well to balance the overhead of copying the data back from the subprocesses. The default is 10.

Running the above command will produce:
1. **a "`meanimage`" directory** in `[root_directory]\[slide_ID]\im3` that contains the following:
    - **a `[slideID]-mean_image.bin` file** that is the mean of the counts/ms in all of the selected HPFs' tissue regions, stored as 64-bit floats
    - **a `[slideID]-std_error_of_mean_image.bin` file** that is the standard error on the mean counts/ms in all of the selected HPFs' tissue regions, stored as 64-bit floats
    - **a `[slideID]-mask_stack.bin` file** that is the stack of the binary image masks from every selected HPF, stored as 64-bit unsigned integers
    - **a very detailed "global" log file** called "`global-slide_mean_image.log`"
    - **a list of every field used** in finding the background thresholds and in making the mean image, called "`fields_used_meanimage.csv`", stored as a ["FieldLog" object](https://github.com/AstroPathJHU/AstroPathPipeline/blob/master/astropath/flatfield/utilities.py#L22-L29)
    - **a list of every slide used** in stacking images, including date ranges, called "`metadata_summary_stacked_images_meanimage.csv`" and stored as a ["MetadataSummary" object](https://github.com/AstroPathJHU/AstroPathPipeline/blob/master/astropath/utilities/misc.py#L141-L149)
    - **a "`thresholding_info`" subdirectory** containing plots/details about how the background thresholding proceeded
    - **a "`postrun_info`" subdirectory** containing information about how many HPFs were stacked in each layer of the meanimage, how many raw HPFs were read, and .png images of the individual mean image and mask stack layers
    - **a "`image_masking`" subdirectory** containing some example plots of the image masks that were produced, multilayer "`[image_key]_mask.bin`" mask files for any stacked images that had blur or saturation flagged in them, a "`labelled_mask_regions.csv`" file listing every region masked due to blur or saturation as ["LabelledMaskRegion" objects](https://github.com/AstroPathJHU/AstroPathPipeline/blob/flagging_HPF_regions/astropath/flatfield/utilities.py#L38-L44), and a plot of where the HPFs that were read and flagged are located within the slide.
2. **a main log file** called "`slide_mean_image.log`" in `[root_directory]\logfiles` with just a single line showing that slide_mean_image was run
3. **a more detailed sample log file** called "`[slideID]-slide_mean_image.log`" in `[root_directory]\[slide_ID]\logfiles`

The number of threads is currently the ONLY option the user can change, in order to ensure consistency between mean images created for different slides.

### Combining slide mean images into a flatfield model for a batch

The `batch_flatfield` mode reads the `[slideID]-mean_image.bin` and `[slideID]-mask_stack.bin` files created for a batch of slides and combines them to produce a flatfield correction model. To run it in the most common use case, after running `slide_mean_image` for a batch of slides, enter the following command and arguments :

`run_flatfield batch_flatfield --slides [comma_separated_list_of_slide_IDs] --rawfile_top_dir [rawfile_directory] --root_dir [root_directory] --batchID [batch_ID]`

where:
- `[comma_separated_list_of_slide_IDs]` is a comma-separated list of slide IDs whose mean images should be combined (i.e. "`M107_1,M109_1,M110_1,M111_1,M112_1`")
- `[rawfile_directory]` and `[root_directory]` are the same as in the previous run mode
- `[batch_ID]` is the integer identifier of which batch the group of slides represents (i.e. "14"). Currently only IDs between 0 and 99 are supported, since the IDs are written out as zero-padded two-digit numbers in several output contexts.

Running the above command will produce:
1. **a flatfield correction model .bin file** called `flatfield_BatchID_[batch_ID].bin` in `[root_directory]\Flatfield`
2. **a main log file** called "`batch_flatfield.log`" in `[root_directory]\logfiles` with just a single line showing that batch_flatfield was run 
3. **more detailed sample log files** called "`[slide_ID]-batch_flatfield.log`" in `[root_directory]\[slide_ID]\logfiles` for each `[slide_ID]` in `[comma_separated_list_of_slide_IDs]`
4. **a `flatfield_BatchID_[batch_ID]` subdirectory** in `[root_directory]\Flatfield` that contains the following:
    - **a summary PDF file** called `flatfield_BatchID_[batch_ID]_summary.pdf` that shows the layers of the flatfield image, the relative spread in the correction factors in each layer, how many images were stacked from all slides in each layer, and the layers of the combined mask stack all in one quick little document for reference
    - **a very detailed "global" log file** called "`global-batch_flatfield.log`"
    - **field log** and **metadata summary** files like in the above run mode, combined for every slide in the batch
    - **a "`postrun_info`" subdirectory** containing similar low-level info to that of the previous run mode

There are currently NO OPTIONS for the user to change in this run mode, again to ensure consistency between the mean images used in making the flatfield model. Also please note that producing the summary PDF file requires running on a system that recognizes `pdflatex` as a command. If the runtime environment doesn't have LaTeX installed (along with the `graphicx` and `geometry` packages), the template .tex file for the PDF is still created but the output PDF is not, and a low-level warning is output to the log files.

## Image Correction

The "image correction" portion of the code corrects raw ".Data.dat" files based on a given flatfield and warping model and writes out their contents as ".fw" files. It runs for one slide at a time. To run it in the most common use case, enter the following command and arguments:

`run_image_correction [slide_ID] [rawfile_directory] [root_directory] [working_directory] --flatfield_file [path_to_flatfield_bin_file] --warp_def [path_to_warp_csv_file]`

where:
- `[slide_ID]` is the name of the slide whose files should be corrected and re-written out (i.e. "`M21_1`")
- `[rawfile_directory]` is the path to a directory containing a `[slide_ID]` subdirectory with the slide's ".Data.dat" files (i.e. ``\\bki07\dat``)
- `[root_directory]` is the path to the usual "Clinical_Specimen" directory containing a `[slide_ID]` subdirectory (i.e. `\\bki02\E\Clinical_Specimen`)
- `[working_directory]` is the path to the directory in which the corrected ".fw" files should be written out (it will be created if it doesn't already exist, and it can be anywhere)
- `[path_to_flatfield_bin_file]` is the path to the ".bin" file specifying the flatfield corrections to apply (i.e. `\\bki02\E\Clinical_Specimen\Flatfield\flatfield.bin`, or one of the files created by the python version of the flatfielding code)
- `[path_to_warp_csv_file]` is the path to a .csv file detailing the warping model parameters as a ["WarpingSummary" object](https://github.com/AstroPathJHU/AstroPathPipeline/blob/master/astropath/warping/utilities.py#L130-L149) (i.e. the file [here](https://github.com/AstroPathJHU/alignmentjenkinsdata/blob/master/corrections/TEST_WARPING_weighted_average_warp.csv), which was output by the python version of the warping code)

Running the above command will produce:
1. **corrected ".fw" files** in the `[working_directory]`
2. **some plots** and details about the correction models that were applied in `[working_directory]\applied_correction_plots`
3. **a main log file** called "`image_correction.log`" in `[root_directory]\logfiles` with just a single line showing that image_correction was run 
4. **a more detailed sample log file** called "`[slide_ID]-image_correction.log`" in `[root_directory]\[slide_ID]\logfiles` and
5. **a very detailed "image level" log file** called "`[slide_ID]_images-image_correction.log`" in `[working_directory]` 

Other options for how the correction should be done include:
1. Skipping the flatfield corrections: run with the `--skip_flatfielding` flag instead of the `--flatfield_file` argument (exactly one of them must be given)
2. Skipping the warping corrections: run with the `--skip_warping` flag instead of the `--warp_def` argument (exactly one of them must be given)
3. Using only one image layer instead of all image layers at once: add the `--layer [n]` argument where `[n]` is the integer layer number to use (starting from 1). In this case, the output files are named ".fwxx" where "xx" is the two-digit layer number (i.e. ".fw01"). (The default `--layer` argument is `-1`, which runs all the image layers at once.)
4. Shifting the principal point of the warping pattern, in one of two ways:
    - Shifting the pattern by a different amount for each image layer: add the `--warp_shift_file [path_to_warp_shift_csv_file]` argument where `[path_to_warp_shift_csv_file]` is the path to a .csv file detailing a ["WarpShift" object](https://github.com/AstroPathJHU/AstroPathPipeline/blob/master/astropath/warping/utilities.py#L123-L128) like the example file [here](https://github.com/AstroPathJHU/alignmentjenkinsdata/blob/master/corrections/random_warp_shifts_for_testing.csv), which can be made either by hand or using [this "writetable" function](https://github.com/AstroPathJHU/AstroPathPipeline/blob/master/astropath/utilities/tableio.py#L84-L132)
    - Shifting the pattern by the same amount for every layer: add the `--warp_shift [cx_shift,cy_shift]` argument where `[cx_shift,cy_shift]` is something like "5.2,-3.6" (i.e. parseable as a tuple of floats)
In both cases the `cx_shift` and `cy_shift` fields are how far to move the principal point of the pattern in pixel units (floats are accepted)
5. Scaling the strength of the warping model by a single factor: add the `--warping_scalefactor [wsf]` argument where `[wsf]` is the scalefactor to multiply the relevant parameters by (the default argument is 1.0)
6. Only running some files instead of all of them: (usually for testing) add the `--max_files [max_files]` argument where `[max_files]` is the integer # of files to run (the default argument is -1, which runs all the files)
7. Using different input or output file extensions: add the `--input_file_extension` (default=".Data.dat") or `--output_file_extension` (default is ".fw") arguments, respectively
8. Specifying a warping pattern using dx and dy warp fields instead of model parameters: in this case the `--warp_def [warp_def_dir]` argument is the path to a directory holding the dx/dy warping factor .bin files. The files must be in the `[warp_def_dir]`, named `dx_warp_field_[warp_def_dirname]` and `dy_warp_field_[warp_def_dirname]`. Note that a pattern defined in this way cannot be shifted, but it can be scaled. 
