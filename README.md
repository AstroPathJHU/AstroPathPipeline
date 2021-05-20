# <div align="center"> AstroPathPipeline </div>
#### <div align="center"> The AstroPathPipeline was developed to process whole slide multiplex immunofluorescence data from microscope to database </div>
#### <div align="center">Correspondence to: bgreen42@jhu.edu</div>

## 1. Description
The AstroPath pipeline was designed to automate the processing of whole slide multiplex immunoflourecence histopathology image data, taken by Akoya’s Vectra imaging platform, from the microscope to database. The automated process begins after whole slide scans have been captured by the microscope and manually verified complete. Code is divided into three main stages; defined as ```hpf```, ```slide```, and ```sample``` level processing. In the ```hpf``` (or high powered field) processing stage, images are reorganized, corrected for camera\ imaging effects, and segmented\ phenotyped. Here images are mainly processed individually. In the next processing stage, aptly named ```slide``` the data is stiched together into a whole slide and the slides are annotated by a pathologist. Finally, slides across a cohort are corrected for batch to batch variation and loaded into a database. Here the image, cell, and annotation data of each whole slide image is linked its clinical information thus providing a completed ```sample```. Code for each stage is organized into its own folder under ```astropath```, with each folder containing a particular set of modules. Each module is organized separately in subfolders and described with linked documenation. An overview of the current pipeline can be seen [here](https://github.com/AstropathJHU/AstroPathPipeline/blob/main/AstroPathPipeline.pdf).

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
Check out\ download the github repository. In MATLAB add the entire *AstroPathPipeline* to the MATLAB path. The *AstroPathPipeline* commands should then be available in MATLAB. 

*NOTE*: For specific Python, MATLAB, cmd, or PowerShell commands of a particular module check the module or workflow instructions.

## 3. Contents
- [1. Description](#1-description "Title")
- [2. Instructions](#2-instructions "Title")
- [3. Contents](#3-contents "Title")
- [4. Scanning Slides (scans)](astropath/scans#4-scans "Title")
   - [4.1. Description](astropath/scans#41-description "Title")
   - [4.2. Contents](astropath/scans#42-contents "Title")
   - [4.3. Definitions](astropath/scans/docs/Definitions.md/#43-definitions)
     - [4.3.1. Indentification Definitions](astropath/scans/docs/Definitions.md/#431-identification-definitions)
     - [4.3.2. Path Definitions](astropath/scans/docs/Definitions.md/#432-path-definitions)
   - [4.4. Scanning Instructions Introduction](astropath/scans/docs/ScanningInstructionsIntro.md)
     - [4.4.1. Contents](astropath/scans/docs/ScanningInstructionsIntro.md/#441-contents) 
     - [4.4.2. Specimen Table](astropath/scans/docs/scanning/SpecimenTable.md)
     - [4.4.3. SampleNames (Patient # or M Numbers)](astropath/scans/docs/scanning/SampleNames.md)
     - [4.4.4. Control TMA Conventions](astropath/scans/docs/scanning/ControlTMAConventions.md)
     - [4.4.5. Whole Slide Scanning Introduction](astropath/scans/docs/scanning/WholeSlideScanning.md)
       - [4.4.5.1. Contents](astropath/scans/docs/scanning/WholeSlideScanning.md/#4451-contents)
       - [4.4.5.2. Preparing 20% Overlap](astropath/scans/docs/scanning/Preparing20Overlap.md)
       - [4.4.5.3. Scan the *Control TMA*](astropath/scans/docs/scanning/ScantheControlTMA.md)
       - [4.4.5.4. Scan the Whole Slide Specimens](astropath/scans/docs/scanning/ScantheWholeSlideSpecimens.md)
       - [4.4.5.5. Creating a Whole Slide Scanning Plan](astropath/scans/docs/scanning/CreatingaWholeSlideScanningPlan.md)
       - [4.4.5.6. Important Scanning Notes](astropath/scans/docs/scanning/ImportantScanningNotes.md)
     - [4.4.6. BatchIDs](astropath/scans/docs/scanning/BatchIDs.md)
     - [4.4.7. Batch Tables](astropath/scans/docs/scanning/BatchTables.md)
     - [4.4.8. Merge Configuration Tables](astropath/scans/docs/scanning/MergeConfigTables.md)
  - [4.5. AstroPath Processing Directory and Initalizing Projects](astropath/scans/docs/AstroPathProcessingDirectoryandInitializingProjects.md)
    - [4.5.1. AstroPath Processing Directory](astropath/scans/docs/AstroPathProcessingDirectoryandInitializingProjects.md/#451-astropath-processing-directory)
    - [4.5.2. Initializing Projects](astropath/scans/docs/AstroPathProcessingDirectoryandInitializingProjects.md/#452-initializing-projects)
  - [4.6. Directory Organization](astropath/scans/docs/DirectoryOrganization.md)
    - [4.6.1. Directory Subfolders](astropath/scans/docs/DirectoryOrganization.md#461-directory-subfolders)
    - [4.6.2. SlideID Subfolders](astropath/scans/docs/DirectoryOrganization.md#462-slideid-subfolders)
- [5. HPF Processing (hpfs)](astropath/hpfs#5-hpf-processing-hpfs "Title")
  - [5.1. Description](astropath/hpfs#51-description "Title")
  - [5.2. Instructions](astropath/hpfs#52-instructions "Title")
  - [5.3. Workflow Overview](astropath/hpfs#53-workflow-overview "Title")
  - [5.4. Contents](astropath/hpfs#54-contents "Title")
  - [5.5. AstroIDGen](astropath/hpfs/AstroidGen#55-astroid-generation "Title")
    - [5.5.1. Description](astropath/hpfs/AstroidGen#551-description "Title")
    - [5.5.2. Important Definitions](astropath/hpfs/AstroidGen#552-important-definitions "Title")
    - [5.5.3. Workflow](astropath/hpfs/AstroidGen#553-workflow "Title")
  - [5.6. Transfer Daemon](astropath/hpfs/TransferDaemon#56-transfer-daemon "Title")
    - [5.6.1. Description](astropath/hpfs/TransferDaemon#561-description "Title")
    - [5.6.2. Important Definitions](astropath/hpfs/TransferDaemon#562-important-definitions "Title")
    - [5.6.3. Code Input](astropath/hpfs/TransferDaemon#563-code-input "Title")
    - [5.6.4. Workflow](astropath/hpfs/TransferDaemon#564-workflow "Title")
      - [5.6.4.1. Initial Transfer](astropath/hpfs/TransferDaemon#5641-initial-transfer "Title")
      - [5.6.4.2. MD5 Check](astropath/hpfs/TransferDaemon#5642-md5-check "Title")
      - [5.6.4.3. Compression Into Backup](astropath/hpfs/TransferDaemon#5643-compression-into-backup "Title")
      - [5.6.4.4. Source File Handling](astropath/hpfs/TransferDaemon#5644-source-file-handling "Title")
    - [5.6.5. Notes](astropath/hpfs/TransferDaemon#565-notes "Title") 
  - [5.7. Meanimages](astropath/hpfs/meanimages#57-meanimages "Title")
    - [5.7.1. Description](astropath/hpfs/meanimages#571-description "Title")
    - [5.7.2. Important Definitions](astropath/hpfs/meanimages#572-important-definitions "Title")
    - [5.7.3. Instructions](astropath/hpfs/meanimages#573-instructions "Title")
    - [5.7.4. Workflow](astropath/hpfs/meanimages#574-workflow "Title")
      - [5.7.4.1. Checking for Tasks](astropath/hpfs/meanimages#5741-checking-for-tasks "Title")
	  - [5.7.4.2. Shred Im3s](astropath/hpfs/meanimages#5742-shred-im3s "Title")
	  - [5.7.4.3. raw2mean](astropath/hpfs/meanimages#5743-raw2mean "Title")
  - [5.8. Flatfield](astropath/hpfs/Flatfield#58-flatfield "Title")
    - [5.8.1. Description](astropath/hpfs/Flatfield#581-description "Title")
    - [5.8.2. Important Definitions](astropath/hpfs/Flatfield#582-important-definitions "Title")
      - [5.8.2.1. Flatw Expected Directory Structure](astropath/hpfs/Flatfield#5821-flatw-expected-directory-structure "Title")
	  - [5.8.2.2. Output Formatting](astropath/hpfs/Flatfield#5822-output-formatting "Title")
    - [5.8.3. Instructions](astropath/hpfs/Flatfield#583-instructions "Title")
      - [5.8.3.1. flatw_queue](astropath/hpfs/Flatfield#5831-flatw_queue "Title")
	  - [5.8.3.2. flatw_worker](astropath/hpfs/Flatfield#5832-flatw_worker "Title")
	  - [5.8.3.3. Im3tools](astropath/hpfs/Flatfield#5833-im3tools "Title")
    - [5.8.4. Overview Workflow of Im3Tools](astropath/hpfs/Flatfield#584-overview-workflow-of-im3tools "Title")
  - [5.9. Mergeloop](mergeloop#59-mergeloop "Title")
  - [5.10. Inform_processing](inform_processing#510-inform_processing "Title")
  - [5.11. Segmaps](segmaps#511-segmaps "Title")
  - [5.12. Transferanno](transferanno#512-transferanno "Title")
 - [6. Slide_Processing]
   - [6.1 Description]
   - [6.2 Prepdb](astropath/slides/prepdb/#62-prepdb)
   - [6.3 Align](astropath/slides/align/#63-align)
   - [6.4 Annotations]
   - [6.5 Cell Geometries]
 - [7. Sample_Processing]
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
