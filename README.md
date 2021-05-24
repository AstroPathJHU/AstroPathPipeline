# <div align="center"> AstroPath Pipeline </div>
#### <div align="center"> The *AstroPath Pipeline* was developed to process whole slide multiplex immunofluorescence data from microscope to database </div>
#### <div align="center">Correspondence to: bgreen42@jhu.edu</div>

## 1. Description
The *AstroPath Pipeline* was designed to automate the processing of whole slide multiplex immunoflourecence histopathology image data, taken by Akoya Biosciences’ Vectra imaging platform, from the microscope to database. The automated process begins after whole slide scans have been captured by the microscope and manually verified complete. Code is divided into three main stages; defined as ```hpf```, ```slide```, and ```sample``` level processing. In the ```hpf``` (or high powered field) processing stage, images are reorganized, corrected for camera\ imaging effects, and segmented\ phenotyped. Here images are mainly processed individually. In the next processing stage, aptly named ```slide``` the data is stiched together into a whole slide and the slides are annotated by a pathologist. Finally, slides across a cohort are corrected for batch to batch variation and loaded into a database. Here the image, cell, and annotation data of each whole slide image is linked its clinical information thus providing a completed ```sample```. Code for each stage is organized into its own folder under ```astropath```, with each folder containing a particular set of modules. Each module is organized separately in subfolders and described with linked documenation. An overview of the current pipeline can be seen [here](documents/AstroPathPipeline.pdf).

## 2. Getting Started
## 2.1. Prerequisites
- [Windows 10](https://www.microsoft.com/en-us/software-download/windows10)
- [Python 3.6/3.7](https://www.python.org/)
- [MATLAB 2020a](https://www.mathworks.com/products/matlab.html)

## 2.2. Instructions
### 2.2.1. Python Instructions
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
  - [5.5. AstroIDGen](astropath/hpfs/AstroidGen#55-astroid-generation-v0000001 "Title")
  - [5.6. Transfer Daemon](astropath/hpfs/TransferDaemon#56-transfer-daemon "Title")
  - [5.7. Meanimages](astropath/hpfs/meanimages#57-meanimages "Title")
  - [5.8. Flatw](astropath/hpfs/flatw#58-flatw "Title")
  - [5.9. Mergeloop](astropath/hpfs/mergeloop#59-mergeloop "Title")
  - [5.10. Inform Processing](astropath/hpfs/inform_processing#510-inform-processing "Title")
  - [5.11. Segmaps](astropath/hpfs/segmaps#511-seg-maps "Title")
  - [5.12. Create & Transfer Annotations](astropath/hpfs/transferanno#512-transfer-annotations "Title")
 - [6. Slide Processing](astropath/slides/#6-slide-processing)
   - [6.1 Description](astropath/slides/#61-description)
   - [6.2 Prepdb](astropath/slides/prepdb/#62-prepdb)
   - [6.3 Align](astropath/slides/align/#63-align)
   - [6.4 Annotations]
   - [6.5 Cell Geometries]
 - [7. Sample Processing]
   - [7.1 Description]
   - [7.2 Control TMAs]
   - [7.3 Calibration]
   - [7.4 Upload to Database]

## Credits
#### <div align="center">Created by: Benjamin Green<sup>1</sup>, Jeffrey S. Roskes<sup>4</sup>, Margaret Eminizer<sup>4</sup>, Richard Wilton<sup>4</sup>, Sigfredo Soto-Diaz<sup>2</sup>, Sneha Berry<sup>2</sup>, Elizabeth Engle<sup>2</sup>, Nicolas Giraldo<sup>3</sup>, Peter Nguyen<sup>2</sup>, Tricia Cottrell<sup>3</sup>, Janis Taube<sup>1,2,3</sup>, and Alex Szalay<sup>4</sup></div>

 <div align="center">Departments of <sup>1</sup>Dermatology, <sup>2</sup>Oncology, <sup>3</sup>Pathology at Johns Hopkins University SOM, the Mark Center for Advanced Genomics and Imaging, the Sidney Kimmel Comprehensive Cancer Center, and the Bloomberg~Kimmel Institute for Cancer Immunotherapy at Johns Hopkins, Baltimore, MD, USA</div>
 <div align="center"> Departments of <sup>4</sup>Astronomy and Physics at Johns Hopkins University and IDIES, Baltimore, MD, USA</div> 
 <br>
 
Individual Contributions: **Benjamin Green**: Conceptualization, Methodology, Software, Writing – Original Draft, Visualization **Jeffrey S. Roskes**: Conceptualization, Methodology, Software, Writing – Original Draft **Margaret Eminizer**: Conceptualization, Methodology, Software, Writing – Original Draft, Visualization **Richard Wilton**: Methodology, Software **Sigfredo Soto-Diaz**: Methodology, Software, Writing – Original Draft **Sneha Berry**: Conceptualization, Validation, Visualization **Liz Engle**: Conceptualization, Resources, Validation **Nicolas Giraldo-Castillo**: Conceptualization **Peter Nguyen**: Conceptualization, Methodology **Tricia Cottrell**: Conceptualization, Validation, Writing – Review & Editing **Janis Taube**: Conceptualization, Resources, Supervision **Alex Szalay**: Conceptualization, Methodology, Validation, Software, Supervision
