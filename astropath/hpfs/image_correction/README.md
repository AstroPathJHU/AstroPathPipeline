
# 5.8. Flatw
## 5.8.1. Description
This workflow serves to create a directory of flat field and warping corrected *.im3* images files for each slide. In addition, this workflow saves the full metadata for the first *.im3* (*.full.im3*) for a slide, the single column bitmap of each corrected *.im3* (*.fw* files), some shape parameters for the first *.im3* (*.Parameters.xml*), as well as relevant image metadata (*.SpectralBasisInfo.Exposure.Protocol.DarkCurrentSettings.xml*) of each im3 image. We assume that the directory format of the input image files is in the ```AstroPathPipeline``` processing file stucture described [here](../../scans/docs/DirectoryOrganization.md#46-directory-organization) and in [5.8.3.1.](docs/ImportantDefinitions.md#5831-flatw-expected-directory-structure). 

There are multiple parts of this module provided which can be run collectively as a workflow through MATLAB functions ([5.8.4.](docs/WorkflowInstructions.md#584-workflow-instructions)) or as separate tools ([5.8.5.](docs/AdditionalTools.md#585-additional-tools)). The code is maintained and updated to be used through the workflow and uses outside of this workflow will not be supported by the *AstroPath* group at this time. The workflow loops through all ```Project```s in the [*AstropathCohortsProgress.csv*](../../scans/docs/AstroPathProcessingDirectoryandInitializingProjects.md#451-astropath_processing-directory) and processes slides if a *flatfield_BatchID_NN.bin* file is created for that batch (see [5.8.3.1.](docs/ImportantDefinitions.md#5831-flatw-expected-directory-structure)).

*NOTE*: The warping corrections are only applied in this version of the code to images with the same shape parameters (defined in the *Parameters.xml* extracted from the image metadata) as the JHU Vectra 3.0 machine. A warping model could be estimated and added to the *mkwarp* function in the *runflatw.m* file [here](flatw_matlab/mtools) for a different microscope. That functionality is not included in this processing. 

## 5.8.2. Contents
- [5.8.3. Important Definitions](docs/ImportantDefinitions.md#583-important-definitions)
  - [5.8.3.1. Flatw Expected Directory Structure](docs/ImportantDefinitions.md#5831-flatw-expected-directory-structure)
  - [5.8.3.2. Output Formatting](docs/ImportantDefinitions.md#5832-output-formatting)
- [5.8.4. Workflow Instructions](docs/WorkflowInstructions.md#584-workflow-instructions)
  - [5.8.4.1. flatw queue](docs/WorkflowInstructions.md#5841-flatw_queue)
  - [5.8.4.2. flatw worker](docs/WorkflowInstructions.md#5842-flatw_worker)
- [5.8.5. Additional Tools](docs/AdditionalTools.md#585-additional-tools)
  - [5.8.5.1. Im3Tools](docs/AdditionalTools.md#5851-im3tools)
  - [5.8.5.2. ConvertIm3](docs/AdditionalTools.md#5852-convertim3)
  - [5.8.5.3. ConvertIm3Path & ConvertIm3Cohort](docs/AdditionalTools.md#5853-convertim3path--convertim3cohort)
- [5.8.6. Overview Workflow of Im3Tools](docs/OverviewWorkflowofIm3Tools.md#586-overview-workflow-of-im3tools)
# Image Correction

The "image correction" portion of the code corrects raw ".Data.dat" files based on a given flatfield and warping model and writes out their contents as ".fw" files. To run it for a single sample in the most common use case, enter the following command and arguments:

`imagecorrectionsample <Dpath>\<Dname> <Rpath> <SlideID> --flatfield_file [path_to_flatfield_bin_file] --warping_file [path_to_warping_summary_csv_file] --njobs [njobs]`

where:
- `[path_to_flatfield_bin_file]` is the path to the ".bin" file specifying the flatfield corrections to apply, or the name of a file located in the `<Dpath>\<Dname>\Flatfield` directory
- `[path_to_warping_summary_csv_file]` is the path to a .csv file detailing the warping model parameters as a [`WarpingSummary` object](../warping/utilities.py#L43-L60). This file can have several `WarpingSummary` entries specifying different warping patterns to apply in differen raw image layers.
- `[njobs]` is the maximum number of parallel processes allowed to run at once (many parallel processes can be used; each process corrects and writes out one file at a time)

See [here](../../scans/docs/Definitions.md#43-definitions) for definitions of the terms in `<angle brackets>`.

Running the above command will produce:
1. **corrected ".fw" files** in `<Dpath>\flatw\<SlideID>`
1. **a main log file** called "`imagecorrection.log`" in `<Dpath>\<Dname>\logfiles` with just a single line showing that `imagecorrectionsample` was run 
1. **a more detailed sample log file** called "`<SlideID>-imagecorrection.log`" in `<Dpath>\<Dname>\<SlideID>\logfiles`

Other options for how the correction should be done include:
1. Putting the output in a different location: add the `--workingdir [workingdir_path]` argument where `[workingdir_path]` is the path to the directory where the output should go
1. Writing out corrected files for single image layers as well as multilayer images: add the `--layers [layers]` argument where `[layers]` is any number of arguments specifying which layer numbers to use (starting from 1). In the single image layer case, the corresponding output files are named ".fwxx" where "xx" is the two-digit layer number (i.e. ".fw01"). The special number -1 can also be given as an argument in `[layers]`, in which case the multilayer files will be written out in addition to any single layer files requested. Using this argument one could, for example, simultaneously write out the corrected multilayer .fw files and the corrected single layer .fw01 files.
1. Skipping the flatfield and/or warping corrections: run without specifying the `--flatfield_file` and/or `--warping_file` arguments

The image correction routine can be run for an entire cohort of samples at once using the following command:

`imagecorrectioncohort <Dpath>\<Dname> <Rpath>`

To see more command line arguments available for both routines, run `imagecorrectionsample --help` or `imagecorrectioncohort --help`.

