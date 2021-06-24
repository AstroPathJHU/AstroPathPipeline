# Image Correction

The "image correction" portion of the code corrects raw ".Data.dat" files based on a given flatfield and warping model and writes out their contents as ".fw" files. It runs for one slide at a time. To run it in the most common use case, enter the following command and arguments:

`run_image_correction [slide_ID] [rawfile_directory] [root_directory] [working_directory] --flatfield_file [path_to_flatfield_bin_file] --warp_def [path_to_warp_csv_file]`

where:
- `[slide_ID]` is the name of the slide whose files should be corrected and re-written out (i.e. "`M21_1`")
- `[rawfile_directory]` is the path to a directory containing a `[slide_ID]` subdirectory with the slide's ".Data.dat" files (i.e. ``\\bki07\dat``)
- `[root_directory]` is the path to the usual "Clinical_Specimen" directory containing a `[slide_ID]` subdirectory (i.e. `\\bki02\E\Clinical_Specimen`)
- `[working_directory]` is the path to the directory in which the corrected ".fw" files should be written out (it will be created if it doesn't already exist, and it can be anywhere)
- `[path_to_flatfield_bin_file]` is the path to the ".bin" file specifying the flatfield corrections to apply (i.e. `\\bki02\E\Clinical_Specimen\Flatfield\flatfield.bin`, or one of the files created by the python version of the flatfielding code)
- `[path_to_warp_csv_file]` is the path to a .csv file detailing the warping model parameters as a [`WarpingSummary` object](../warping/utilities.py#L114-L132)

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
    - Shifting the pattern by a different amount for each image layer: add the `--warp_shift_file [path_to_warp_shift_csv_file]` argument where `[path_to_warp_shift_csv_file]` is the path to a .csv file detailing a [`WarpShift` object](../warping/utilities.py#L108-L112) like the example file [here](https://github.com/AstroPathJHU/alignmentjenkinsdata/blob/master/corrections/random_warp_shifts_for_testing.csv), which can be made either by hand or using [this "writetable" function](../../utilities/tableio.py#L82)
    - Shifting the pattern by the same amount for every layer: add the `--warp_shift [cx_shift,cy_shift]` argument where `[cx_shift,cy_shift]` is something like "5.2,-3.6" (i.e. parseable as a tuple of floats)
In both cases the `cx_shift` and `cy_shift` fields are how far to move the principal point of the pattern in pixel units (floats are accepted)
5. Scaling the strength of the warping model by a single factor: add the `--warping_scalefactor [wsf]` argument where `[wsf]` is the scalefactor to multiply the relevant parameters by (the default argument is 1.0)
6. Only running some files instead of all of them: (usually for testing) add the `--max_files [max_files]` argument where `[max_files]` is the integer # of files to run (the default argument is -1, which runs all the files)
7. Using different input or output file extensions: add the `--input_file_extension` (default=".Data.dat") or `--output_file_extension` (default is ".fw") arguments, respectively
8. Specifying a warping pattern using dx and dy warp fields instead of model parameters: in this case the `--warp_def [warp_def_dir]` argument is the path to a directory holding the dx/dy warping factor .bin files. The files must be in the `[warp_def_dir]`, named `dx_warp_field_[warp_def_dirname]` and `dy_warp_field_[warp_def_dirname]`. Note that a pattern defined in this way cannot be shifted, but it can be scaled. 
