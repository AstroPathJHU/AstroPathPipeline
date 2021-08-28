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
Especially on Windows,
it is recommended to run python using an Anaconda distribution, which helps
with installing dependencies.  While most of the dependencies can just be
installed with pip, others have C++ requirements that are significantly easier
to set up with Anaconda.

Our recommendation is to download a [Miniconda distribution](https://docs.conda.io/en/latest/miniconda.html).
Once you install it, open the Anaconda powershell prompt and create an environment
```
conda create --name astropath python=3.8
conda activate astropath
```

At least the following dependencies should be installed through Anaconda.
```
conda install -c conda-forge pyopencl gdal cvxpy
```
Many of the other dependencies can also be installed through Anaconda if you want,
but we have found that they work just as well when installing with pip.

#### 2.2.1.2. Code installation
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

### 2.2.2. MATLAB Instructions
Check out\ download the github repository. In MATLAB add the entire *AstroPath Pipeline* to the MATLAB path. The *AstroPath Pipeline* commands should then be available in MATLAB. 

*NOTE*: For specific Python, MATLAB, cmd, or PowerShell commands of a particular module check the module or workflow instructions.

## 3. Contents
- [1. Description](#1-description "Title")
- [2. Getting Started](#2-getting-started "Title")
   - [2.1. Prerequisites](#21-prerequisites)
   - [2.2. Instructions](#22-instructions)
      - [2.2.1. Python Instructions](#221-python-instructions)
         - [2.2.1.1. Environment setup](#2211-environment-setup)
         - [2.2.1.2. Code installation](#2212-code-installation)
      - [2.2.2. MATLAB Instructions](#222-matlab-instructions)
- [3. Contents](#3-contents "Title")
- [4. Scanning Slides (scans)](astropath/scans#4-scans "Title")
   - [4.1. Description](astropath/scans#41-description "Title")
   - [4.2. Contents](astropath/scans#42-contents "Title")
   - [4.3. Definitions](astropath/scans/docs/Definitions.md/#43-definitions)
   - [4.4. Scanning Instructions](astropath/scans/docs/ScanningInstructionsIntro.md)
   - [4.5. AstroPath Processing Directory and Initalizing Projects](astropath/scans/docs/AstroPathProcessingDirectoryandInitializingProjects.md)
   - [4.6. Directory Organization](astropath/scans/docs/DirectoryOrganization.md)
- [5. HPF Processing (hpfs)](astropath/hpfs#5-hpf-processing-hpfs "Title")
  - [5.1. Description](astropath/hpfs#51-description "Title")
  - [5.2. Contents](astropath/hpfs#52-contents "Title")
  - [5.3. Instructions](astropath/hpfs/docs/Instructions.md)
  - [5.4. Workflow Overview](astropath/hpfs/docs/WorkflowOverview.md)
  - [5.5. AstroIDGen](astropath/hpfs/astroidgen#55-astroid-generation-v0000001 "Title")
  - [5.6. Transfer Daemon](astropath/hpfs/transferdaemon#56-transfer-daemon "Title")
  - [5.7. Meanimages](astropath/hpfs/meanimages#57-meanimages "Title")
  - [5.8. Image Correction](astropath/hpfs/image_correction#58-image-correction "Title")
  - [5.9. Mergeloop](astropath/hpfs/mergeloop#59-mergeloop "Title")
  - [5.10. Inform Processing (VMinForm)](astropath/hpfs/vminform#510-vminform "Title")
  - [5.11. Segmaps](astropath/hpfs/segmaps#511-seg-maps "Title")
  - [5.12. Create & Transfer Annotations](astropath/hpfs/transferanno#512-transfer-annotations "Title")
  - [5.13. ValidateData](astropath/hpfs/validatedata#513-validate-data)
 - [6. Slide Processing (slides)](astropath/slides/#6-slide-processing)
   - [6.1 Description](astropath/slides/#61-description)
   - [6.2 Prepdb](astropath/slides/prepdb/#62-prepdb)
   - [6.3 Align](astropath/slides/align/#63-align)
   - [6.4 Zoom](astropath/slides/zoom/#64-zoom)
   - [6.5 Deepzoom](astropath/slides/deepzoom/#65-deepzoom)
   - [6.6 Stitch mask](astropath/slides/stitchmask/#66-stitch-mask)
   - [6.7 Warp annotations](astropath/slides/annowarp/#67-warp-annotations)
   - [6.8 Tumor and field geometries](astropath/slides/geom/#68-tumor-and-field-geometries)
   - [6.9 Cell Geometries](astropath/slides/geomcell/#69-cell-geometries)
   - [6.10 Csvscan](astropath/slides/csvscan/#610-csvscan)
 - [7. Sample Processing (samples)](astropath/samples/)
   - [7.1 Description](astropath/samples/)
   - [7.2 Control TMAs & Calibration](astropath/samples/ctrl/)
   - [7.3. Load Data for Database](astropath/samples/loaddb/)
   - [7.4. Load Zoomed Images for CellView](astropath/samples/loadzoom/)
   - [7.5. Prepare the Merge for Database](astropath/samples/prepmerge/)
   - [7.6. Merge the Sample Databases](astropath/samples/mergedb/)

## Credits
#### <div align="center">Created by: Benjamin Green<sup>1</sup>, Jeffrey S. Roskes<sup>4</sup>, Margaret Eminizer<sup>4</sup>, Richard Wilton<sup>4</sup>, Sigfredo Soto-Diaz<sup>2</sup>, Sneha Berry<sup>2</sup>, Elizabeth Engle<sup>2</sup>, Nicolas Giraldo<sup>3</sup>, Peter Nguyen<sup>2</sup>, Tricia Cottrell<sup>3</sup>, Janis Taube<sup>1,2,3</sup>, and Alex Szalay<sup>4</sup></div>

 <div align="center">Departments of <sup>1</sup>Dermatology, <sup>2</sup>Oncology, <sup>3</sup>Pathology at Johns Hopkins University SOM, the Mark Center for Advanced Genomics and Imaging, the Sidney Kimmel Comprehensive Cancer Center, and the Bloomberg~Kimmel Institute for Cancer Immunotherapy at Johns Hopkins, Baltimore, MD, USA</div>
 <div align="center"> Departments of <sup>4</sup>Astronomy and Physics at Johns Hopkins University and IDIES, Baltimore, MD, USA</div> 
 <br>
 
Individual Contributions: **Benjamin Green**: Conceptualization, Methodology, Software, Writing – Original Draft, Visualization **Jeffrey S. Roskes**: Conceptualization, Methodology, Software, Writing – Original Draft **Margaret Eminizer**: Conceptualization, Methodology, Software, Writing – Original Draft, Visualization **Richard Wilton**: Methodology, Software **Sigfredo Soto-Diaz**: Methodology, Software, Writing – Original Draft **Sneha Berry**: Conceptualization, Validation, Visualization **Liz Engle**: Conceptualization, Resources, Validation **Nicolas Giraldo-Castillo**: Conceptualization **Peter Nguyen**: Conceptualization, Methodology **Tricia Cottrell**: Conceptualization, Validation, Writing – Review & Editing **Janis Taube**: Conceptualization, Resources, Supervision **Alex Szalay**: Conceptualization, Methodology, Validation, Software, Supervision
