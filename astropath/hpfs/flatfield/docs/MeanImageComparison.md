## 5.5.4. Mean Image Comparison

## 5.5.4.1. Instructions to Run via *AstroPath Pipeline* Workflow

## 5.5.4.2. Instructions to Run Standalone Via Python Package

After running the meanimage routine for a whole cohort, the set of samples whose mean images should be used to determine a single flatfield correction model can be found using the plot(s)/datatable(s) created by the "[meanimagecomparison](../meanimagecomparison.py)" script. The comparison between any two samples' mean images is determined using the standard deviation of the distribution of the pixel-wise differences between the two mean images divided by their uncertainties (standard deviation of the delta/sigma distribution). This comparison statistic is calculated for every image layer and every pair of samples, and the average over all image layers is plotted for each pair in a grid. The resulting plot shows values near one for samples whose mean images are comparable, and values far from one for samples whose mean images are very different. The plot can be run several times (much more quickly, after the initial datatable is produced) to find the best grouping of slides to use for a single flatfield model. It can also be used to check a new cohort of samples' mean images against previously-run cohorts. To run the script in the most common use case, enter the following command and arguments:

`meanimagecomparison <Dpath>\<Dname>... --sampleregex [sample_regex]`

where:
- `<Dpath>\<Dname>...` is any number of `<Dpath>\<Dname>` paths, one for every root directory containing the slides whose mean images should be compared
- `[sample_regex]` is a regular expression that will match any slide IDs to be added from any of the paths in `[Dpaths]`

Running the above command will compare slides in any/all of the directories in `[Dpaths]` that match the given `[sample_regex]`, producing the following output by default at `//bki04/astropath_processing/meanimagecomparison`:
1. **a `meanimagecomparison_table.csv` file** whose entries list the comparisons between each layer of every pair of slide mean images, stored as [`ComparisonTableEntry` objects](../utilities.py#L37-L44). If that file already exists with some entries in it, invoking the command above will automatically perform additional comparisons with any samples listed in the file that match `[sample_regex]`.
1. **a `meanimage_comparison_average_over_all_layers.png` plot** showing a grid of comparison statistic values for each pair of slides. Again, if the datatable already exists when the command is run, this plot will include comparisons with the slides that were already listed and match `[sample_regex]`.
1. **a log file** called `meanimage_comparison.log`

The goal is that this script can be used to add a new group of slides to an ever-growing table of mean image comparisons. When new slides are received and meanimages for them are produced, this script can be run with the root directory of the new slides given as `[Dpaths]`, and a regular expression matching all of the new slides as well as any desired slides that already exist in the datatable. The plot that is outputted will then include comparisons for the new slides and older slides all together, to help in making determinations of which slides should be combined into a flatfield model. When a group of slides is determined (probably by running the script several times with different groups of slides), a new group of slides for a flatfield model can be registered by running:

`meanimagecomparison <Dpath>\<Dname>... --sampleregex [sample_regex] --store-as [version_tag]`

where `[version_tag]` is a string used to identify the particular group of slides for the flatfield model (i.e. 'v1.1' or similar). Running the command will add lines to the `\\bki04\astropath_processing\AstroPathFlatfieldModels.csv` file listing each of the matched slides as part of a model with the given `[version_tag]` (entries in this file are stored as [`ModelTableEntry` objects](../utilities.py#L29-L35)). It will also create the flatfield model with that version tag in the default location by running BatchFlatfieldCohort (standalone instructions for using that module on its own are [here](./Batchflatfield.md#5552-instructions-to-run-standalone-via-python-package "Title")).

Other options for running the code include:
- Putting output in a different location, including reading existing output from that location: add the `--workingdir [path_to_output_dir]` argument, where `[path_to_output_dir]` is the path to the desired output directory 
- Sorting the slide IDs in the plot in sequential order rather than by Project, Cohort, and Batch: add the `--sort-by order` argument
- Saving a different set of plots: By default, the plot that gets created shows the average of the comparison statistics over all image layers, but plots of individual layers can be save instead by adding the `--plot (all / brightest / none)` argument. "All" saves plots for every image layer individually. "Brightest" saves plots for the brightest layer in each filter group. "None" skips saving plots entirely (useful when defining/saving a group of slides as a new model).
- Changing where the red line breaks are on the outputted plot: Add the `--lines-after [slide_ids]` argument where `[slide_ids]` is a comma-separated list of slide IDs after which a dividing line should be plotted. By default, dividing lines are places between slides in different Projects, Cohorts, and/or Batches.
- Changing the upper/lower bounds for the scale of the outputted plot: Add the `--bounds lower_bound,upper_bound` argument where `lower_bound` and `upper_bound` are the new bounds for the z-axis scale.
- Registering a new version of the flatfield in the flatfield models .csv file, without actually creating the model by running BatchFlatfieldCohort: add the `--skip-creation` flag (BatchFlatfieldCohort can be run on its own further down the line).
