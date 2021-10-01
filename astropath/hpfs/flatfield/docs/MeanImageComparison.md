## 5.5.3. Mean Image Comparison

## 5.5.3.1. Instructions to Run via *AstroPath Pipeline* Workflow

## 5.5.3.2. Instructions to Run Standalone Via Python Package

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
