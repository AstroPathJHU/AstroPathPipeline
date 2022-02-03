## Machine Learning-based Nuclear Segmentation

The `segmentation` submodule runs on unmixed component .tif image files. It uses pre-trained neural network models to segment cellular nuclei based on the DAPI layers of images. The two models implemented are a custom version of [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) trained on a small amount of AstroPath data, and nuclear-only [DeepCell](https://deepcell.com/) (GitHub resources [here](https://github.com/vanvalenlab/intro-to-deepcell)) trained on the entire TissueNet dataset.

To run the routines for a single sample in the most common use case, enter the following commands and arguments:

`segmentationsamplennunet <Dpath>\<Dname> <SlideID> --njobs [njobs]`

OR

`segmentationsampledeepcell <Dpath>\<Dname> <SlideID>`

where `[njobs]` is the maximum number of parallel processes allowed to run at once during the parallelized portions of the code running. 

See [here](../../../scans/docs/Definitions.md#43-definitions) for definitions of the terms in `<angle brackets>`.

Two main differences between the algorithms are: only the `nnunet` algorithm is capable of processing in parallel, and the `nnunet` algorithm takes significantly longer to run than the `deepcell` algorithm.

Running the above commands will produce a "`segmentation\nnunet`" or "`segmentation\deepcell`" directory in `<Dpath>\<Dname>\SlideID\im3` containing a single .npz file for each HPF image in the slide. The .npz files are compressed single-layer image arrays where 0=background, 1=boundaries between different nuclei or between nuclei and background, and 2=individual cellular nuclei. Running the code also produces `segmentationnnunet` or `segmentationdeepcell` main and sample log files.

The `segmentation` routines can be run for an entire cohort of samples at once using the commands:

`segmentationcohortnnunet <Dpath>\<Dname> --njobs [njobs]`

OR

`segmentationcohortdeepcell <Dpath>\<Dname>`

where the arguments are the same as those listed above for `segmentationsample*`. To see more command line arguments available for all routines, run `segmentationsample* --help` or `segmentationcohort* --help`.
