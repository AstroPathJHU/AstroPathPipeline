## Machine Learning-based Nuclear Segmentation

The `segmentation` submodule runs on unmixed component .tif image files. It uses pre-trained neural network models to segment cellular nuclei based on the DAPI layers of images. The two models implemented are a custom version of [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) trained on a small amount of AstroPath data, and nuclear-only [DeepCell](https://deepcell.com/) (GitHub resources [here](https://github.com/vanvalenlab/intro-to-deepcell)) trained on the entire TissueNet dataset.

To run the routine for a single sample in the most common use case, enter the following command and arguments:

`segmentationsample <Dpath>\<Dname> <SlideID> --algorithm [algorithm] --njobs [njobs]`

where `[algorithm]` is the algorithm to use (either `nnunet` or `deepcell`), and `[njobs]` is the maximum number of parallel processes allowed to run at once during the parallelized portions of the code running. 

See [here](../../../scans/docs/Definitions.md#43-definitions) for definitions of the terms in `<angle brackets>`.

Two main differences between the algorithms are: only the `nnunet` algorithm is capable of processing in parallel, and the `nnunet` algorithm takes significantly longer to run than the `deepcell` algorithm.

Running the above command will produce a "`segmentation\[algorithm]`" directory in `<Dpath>\<Dname>\SlideID\im3` containing a single .npz file for each HPF image in the slide. The .npz files are compressed single-layer image arrays where 0=background, 1=boundaries between different nuclei or between nuclei and background, and 2=individual cellular nuclei. Running the code also produces `segmentation` main and sample log files.

The `segmentation` routine can be run for an entire cohort of samples at once using the command:

`segmentationcohort <Dpath>\<Dname> --algorithm [algorithm] --njobs [njobs]`

where the arguments are the same as those listed above for `segmentationsample`. To see more command line arguments available for both routines, run `segmentationsample --help` or `segmentationcohort --help`.
