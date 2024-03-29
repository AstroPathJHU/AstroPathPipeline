# <div align="center"> AstroPath Pipeline </div>
#### <div align="center"> The *AstroPath Pipeline* was developed to process whole slide multiplex immunofluorescence data from microscope to database </div>
#### <div align="center">Correspondence to: bgreen42@jhu.edu</div>

## 1. Description
The *AstroPath Pipeline* was designed to automate the processing of whole slide multiplex immunoflourecence histopathology image data, taken by Akoya Biosciences’ Vectra imaging platform, from the microscope to database. The automated process begins after whole slide scans have been captured by the microscope and manually verified complete. Code is divided into three main stages; defined as ```hpf```, ```slide```, and ```sample``` level processing. In the ```hpf``` (or high powered field) processing stage, images are reorganized, corrected for camera\ imaging effects, and segmented\ phenotyped. Here images are mainly processed individually. In the next processing stage, aptly named ```slide``` the data is stiched together into a whole slide and the slides are annotated by a pathologist. Finally, slides across a cohort are corrected for batch to batch variation and loaded into a database. Here the image, cell, and annotation data of each whole slide image is linked its clinical information thus providing a completed ```sample```. Code for each stage is organized into its own folder under ```astropath```, with each folder containing a particular set of modules. Each module is organized separately in subfolders and described with linked documenation. An overview of the current pipeline can be seen below.

![Figure1](documents/PipelineOverview.PNG)

## 2. Getting Started
## 2.1. Prerequisites
- [Windows 10](https://www.microsoft.com/en-us/software-download/windows10)
- [Python 3.6 or higher](https://www.python.org/)
- [MATLAB 2020a](https://www.mathworks.com/products/matlab.html)

## 2.2. Instructions
### 2.2.1. Python Instructions
#### 2.2.1.1. Environment setup
Especially on Windows, it is recommended to run python using an Anaconda distribution, which helps
with installing dependencies.  While most of the dependencies can just be
installed with pip, others have C++ requirements that are significantly easier
to set up with Anaconda.

Our recommendation is to download a [Miniconda distribution](https://docs.conda.io/en/latest/miniconda.html).
Once you install it, open the Anaconda powershell prompt and create an environment
```pwsh
conda create --name astropath python=3.8
conda activate astropath
```
You should activate the `astropath` environment in every new session before
installing packages (whether through conda or pip) or before running code.

At least the following dependencies should be installed through Anaconda.
```pwsh
conda install -c conda-forge pyopencl gdal cvxpy numba 'ecos!=2.0.8' git jupyter
```
(`pyopencl`, `gdal`, and `cvxpy` have C++ dependencies.
`numba` requires a specific numpy version, and installing it here
avoids unpleasant interactions between conda and pip.
`ecos!=2.0.8` is a workaround for a [bug](https://github.com/embotech/ecos/issues/201)
in the ecos distribution on conda.
`git` may or may not be needed, depending if you
have it installed separately on your computer.
`jupyter` is needed for deepcell.)

Many of the other dependencies can also be installed through Anaconda if you want,
but we have found that they work just as well when installing with pip.

Note: GPU computation is supported in some Python modules through PyOpenCL. You will need to have third-party OpenCL drivers installed for any GPU you want to use. Any GPU built in 2011 or later supports OpenCL. OpenCL drivers can be downloaded [here for Intel GPUs](https://www.intel.com/content/www/us/en/developer/articles/tool/opencl-drivers.html), [here for AMD GPUs](https://www.amd.com/en/support), and [here for NVIDIA GPUs](https://www.nvidia.com/Download/index.aspx).

#### 2.2.1.2. Code installation
To install the code, first check out the repository, enter its directory,
activate the conda envorinment if you are using conda, and run
```bash
pip install .
```
If you want to continue developing the code after installing, run instead
```bash
pip install --editable .
```
You can also add optional dependencies by specifying them in brackets,
as in `pip install (--editable) .[gdal,deepcell]`.
The optional dependencies include:
* `deepcell` - needed to run the DeepCell segmentation algorithm.
* `gdal` - needed for polygon handling, which is used in the `geom`, `geomcell`, `stitchmask`, and `csvscan` steps.
* `nnunet` - needed to run the nnU-Net segmentation algorithm
* `test` - these packages are not needed for the actual AstroPath workflow but are used in various unit tests
* `vips` - used in the `zoom` and `deepzoom` steps of the pipeline
To install all optional dependencies, just specify `[all]`.

Once the code is installed, you can run
```python
import astropath
```
from any directory.

### 2.2.2. PowerShell Instructions
#### 2.2.2.1. Launch using batch files
Most of the code written into powershell was designed to run automated as a background process launched by double clicking a batch file. 
The code monitors all projects defined in the astropath processing files and starts new tasks for slides when appropriate triggers take place.
The set of batch files for modules launched this way can be found in the [*\*\astropath\launch*](astropath/launch) directory. Assuming slides are set up in the [astropath format](astropath/scans/docs/DirectoryOrganization.md) and the [AstroPath processing directory](astropath/scans/docs/AstroPathProcessingDirectoryandInitializingProjects.md) is set up correctly, double clicking the file with the appropriate module name will initiate it. 

#### 2.2.2.2. Starting in Powershell
To run a module on particular slide, check out the repository and in a powershell console enter:
``` 
import-module *\astropath 
``` 
replacing the '`*' with the path to the repository. 

Next use the launchmodule function to start a module as follows: 
```
LaunchModule -mpath:<mpath> -module:<module name> -stringin:<module input>
```
- ```<mapth>```: the astropath processing directory
- ```<module name>```: module name to be launched, most modules launched in powershell are located in the hpfs or scans directories
- ```<stringin>```: dash separated list of arguements for a particular module
For simplicity (understanding that most users will not have a great deal of comfort in powershell), 
one could launch a module such as vminform by invoking the following from a command line:
```
powershell -noprofile -command import-module *\astropath; LaunchModule -mpath:*\astropath_processing -module:vminform -stringin:<dpath>-<slideid>-<antibody>-<algorithm>-<inform version>
```

### 2.2.3. MATLAB Instructions
Check out\ download the github repository. In MATLAB, add the entire *AstroPath Pipeline* to the MATLAB path. The *AstroPath Pipeline* commands should then be available in MATLAB. 

*NOTE*: For specific Python, MATLAB, cmd, or PowerShell commands of a particular module check the module or workflow instructions.

## 3. Contents
- [1. Description](#1-description "Title")
- [2. Getting Started](#2-getting-started "Title")
   - [2.1. Prerequisites](#21-prerequisites)
   - [2.2. Instructions](#22-instructions)
      - [2.2.1. Python Instructions](#221-python-instructions)
      - [2.2.2. PowerShell Instructions](#222-powershell-instructions)
      - [2.2.3. MATLAB Instructions](#223-matlab-instructions)
- [3. Contents](#3-contents "Title")
- [4. Scanning Slides (scans)](astropath/scans#4-scans "Title")
   - [4.1. Description](astropath/scans#41-description "Title")
   - [4.2. Contents](astropath/scans#42-contents "Title")
   - [4.3. Definitions](astropath/scans/docs/Definitions.md/#43-definitions)
   - [4.4. Scanning Instructions](astropath/scans/docs/ScanningInstructionsIntro.md)
   - [4.5. AstroPath Processing Directory and Initalizing Projects](astropath/scans/docs/AstroPathProcessingDirectoryandInitializingProjects.md)
   - [4.6. Directory Organization](astropath/scans/docs/DirectoryOrganization.md)
   - [4.7. AstroIDGen](astropath/scans/astroidgen#47-astroid-generation-v0000001 "Title")
   - [4.8. Transfer Daemon](astropath/scans/transferdaemon#48-transfer-daemon "Title")
- [5. HPF Processing (hpfs)](astropath/hpfs#5-hpf-processing-hpfs "Title")
   - [5.1. Description](astropath/hpfs#51-description "Title")
   - [5.2. Contents](astropath/hpfs#52-contents "Title")
   - [5.3. Instructions](astropath/hpfs/docs/Instructions.md)
   - [5.4. Workflow Overview](astropath/hpfs/docs/WorkflowOverview.md)
   - [5.5. Flatfield](astropath/hpfs/flatfield#55-flatfield "Title")
   - [5.6 Warping](astropath/hpfs/warping#56-warping "Title")
   - [5.7. Image Correction](astropath/hpfs/imagecorrection#57-image-correction "Title")
   - [5.8. VMinForm](astropath/hpfs/vminform#58-vminform "Title")
   - [5.9. Merge](astropath/hpfs/merge#59-merge "Title")
   - [5.10. Segmaps](astropath/hpfs/segmaps#510-seg-maps "Title")
   - [5.11. Create & Transfer Annotations](astropath/hpfs/transferanno#511-transfer-annotations "Title")
   - [5.12. Validate Data](astropath/hpfs/validatedata#512-validate-data)
- [6. Slide Processing (slides)](astropath/slides/#6-slide-processing)
   - [6.1 Description](astropath/slides/#61-description)
   - [6.2 Prepdb](astropath/slides/prepdb/#62-prepdb)
   - [6.3 Align](astropath/slides/align/#63-align)
   - [6.4 Stitch mask](astropath/slides/stitchmask/#64-stitch-mask)
   - [6.5 Zoom](astropath/slides/zoom/#65-zoom)
   - [6.6 Deepzoom](astropath/slides/deepzoom/#66-deepzoom)
   - [6.7 Annotation info](astropath/slides/annotationinfo/#67-annotation-info)
      - [6.7.1 Write annotation info](astropath/slides/annotationinfo/#671-write-annotation-info)
      - [6.7.2 Copy annotation info](astropath/slides/annotationinfo/#672-copy-annotation-info)
   - [6.8 Warp annotations](astropath/slides/annowarp/#68-warp-annotations)
   - [6.9 Cell Geometries](astropath/slides/geomcell/#69-cell-geometries)
   - [6.10 Csvscan](astropath/slides/csvscan/#610-csvscan)
- [7. Sample Processing (samples)](astropath/samples/)
   - [7.1 Description](astropath/samples/)
   - [7.2 Control TMAs & Calibration](astropath/samples/ctrl/)
   - [7.3. Load Data for Database](astropath/samples/loaddb/)
   - [7.4. Load Zoomed Images for CellView](astropath/samples/loadzoom/)
   - [7.5. Prepare the Merge for Database](astropath/samples/prepmerge/)
   - [7.6. Merge the Sample Databases](astropath/samples/mergedb/)
- [8. Powershell](documents/AstropathPSWorkflow.md#81-astropath-powershell-workflow)
   - [8.1 AstroPath Powershell Workflow](documents/AstropathPSWorkflow.md#81-astropath-powershell-workflow)
      - [8.1.1. PS Workflow](documents/AstropathPSWorkflow.md#811-ps-workflow)
	  - [8.1.2. AstroPath Task Distribution](documents/AstropathPSWorkflow.md#812-astropath-task-distribution)
	  - [8.1.3. AstroPath File Change Monitors](documents/AstropathPSWorkflow.md#813-astropath-file-change-monitors)
	  - [8.1.4. Adding Modules](documents/AstropathPSWorkflow.md#814-adding-modules)
	  - [8.1.5. AstroPath Requirements](documents/AstropathPSWorkflow.md#815-astropath-requirements)
	  - [8.1.6. AstroPath Dependencies](documents/AstropathPSWorkflow.md#816-astropath-dependencies)

## Credits
#### <div align="center">Created by: Benjamin Green<sup>1</sup>, Jeffrey S. Roskes<sup>4</sup>, Margaret Eminizer<sup>4</sup>, Richard Wilton<sup>4</sup>, Sigfredo Soto-Diaz<sup>2</sup>, Andrew Jorquera<sup>1</sup>, Sneha Berry<sup>2</sup>, Elizabeth Engle<sup>2</sup>, Nicolas Giraldo<sup>3</sup>, Peter Nguyen<sup>2</sup>, Tricia Cottrell<sup>3</sup>, Janis Taube<sup>1,2,3</sup>, and Alex Szalay<sup>4</sup></div>

 <div align="center">Departments of <sup>1</sup>Dermatology, <sup>2</sup>Oncology, <sup>3</sup>Pathology at Johns Hopkins University SOM, the Mark Center for Advanced Genomics and Imaging, the Sidney Kimmel Comprehensive Cancer Center, and the Bloomberg~Kimmel Institute for Cancer Immunotherapy at Johns Hopkins, Baltimore, MD, USA</div>
 <div align="center"> Departments of <sup>4</sup>Astronomy and Physics at Johns Hopkins University and IDIES, Baltimore, MD, USA</div> 
 <br>
 
Individual Contributions: **Benjamin Green**: Conceptualization, Methodology, Software, Writing – Original Draft, Visualization **Jeffrey S. Roskes**: Conceptualization, Methodology, Software, Writing – Original Draft **Margaret Eminizer**: Conceptualization, Methodology, Software, Writing – Original Draft, Visualization **Richard Wilton**: Methodology, Software **Sigfredo Soto-Diaz**: Methodology, Software, Writing – Original Draft **Andrew Jorquera**: Methodology, Software, Writing – Original Draft **Sneha Berry**: Conceptualization, Validation, Visualization **Liz Engle**: Conceptualization, Resources, Validation **Nicolas Giraldo-Castillo**: Conceptualization **Peter Nguyen**: Conceptualization, Methodology **Tricia Cottrell**: Conceptualization, Validation, Writing – Review & Editing **Janis Taube**: Conceptualization, Resources, Supervision **Alex Szalay**: Conceptualization, Methodology, Validation, Software, Supervision
