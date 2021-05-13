# Introduction

This is the main repository for the Astropath group's code that has been ported to Python. 

# <div align="center"> AstropathPipeline </div>
#### <div align="center">***v0.05.0000***</div>

#### <div align="center">Created by: Benjamin Green<sup>1</sup>, Jeffrey S. Roskes<sup>4</sup>, Margaret Eminizer<sup>4</sup>, Richard Wilton<sup>4</sup>, Sigfredo Soto-Diaz<sup>2</sup>, Sneha Berry<sup>2</sup>, Elizabeth Engle<sup>2</sup>, Nicolas Giraldo<sup>3</sup>, Peter Nguyen<sup>2</sup>, Tricia Cottrell<sup>3</sup>, Janis Taube1<sup>2,3</sup>, and Alex Szalay<sup>4</sup></div>

 <div align="center">Departments of <sup>1</sup>Dermatology, <sup>2</sup>Oncology, <sup>3</sup>Pathology at Johns Hopkins University SOM, the Mark Center for Advanced Genomics and Imaging, the Sidney Kimmel Comprehensive Cancer Center, and the Bloomberg~Kimmel Institute for Cancer Immunotherapy at Johns Hopkins, Baltimore, MD, USA</div>
 <div align="center"> Departments of <sup>4</sup>Astronomy and Physics at Johns Hopkins University and IDIES, Baltimore, MD, USA</div> 
 <br>
 
Individual Contributions: **Benjamin Green**: Conceptualization, Methodology, Software, Writing – Original Draft, Visualization **Jeffrey S. Roskes**: Conceptualization, Methodology, Software, Writing – Original Draft **Margaret Eminizer**: Conceptualization, Methodology, Software, Writing – Original Draft, Visualization **Richard Wilton**: Methodology, Software **Sigfredo Soto-Diaz**: Methodology, Software, Writing – Original Draft **Sneha Berry**: Conceptualization, Validation, Visualization **Liz Engle**: Conceptualization, Resources, Validation **Nicolas Giraldo-Castillo**: Conceptualization **Peter Nguyen**: Conceptualization, Methodology **Tricia Cottrell**: Conceptualization, Validation, Writing – Review & Editing **Janis Taube**: Conceptualization, Resources, Supervision **Alex Szalay**: Conceptualization, Methodology, Validation, Software, Supervision

#### <div align="center">Correspondence to: bgreen42@jhu.edu</div>

## 1. Description
The Astropath pipeline was designed to automate the processing of whole slide multiplex immunoflourecence histopathology image data, taken by Akoya’s Vectra imaging platform, from the microscope to database. The automated process begins after whole slide scans have been captured by the microscope and manually verified complete. Code is divided into three main stages; defined as ```hpf```, ```slide```, and ```sample``` level processing. In the ```hpf``` (or high powered field) processing stage, images are reorganized, corrected for camera\ imaging effects, and segmented\ phenotyped. Here images are mainly processed individually. In the next processing stage, aptly named ```slide``` the data is stiched together into a whole slide and the slides are annotated by a pathologist. Next, slides across a cohort are corrected for batch to batch and loaded into a database. Here the image, cell, and annotation data of each whole slide image is linked its clinical information thus providing a completed sample. Code for each stage is organized into its own folder, with each folder containing a particular set of modules. Each module is organized separately in subfolders and described in this documnetation. An overview of the current pipeline can be seen [here](documents/AstroPathPipeline.pdf).

## 2. Instructions

## 3. Contents
- [1. Description](#1-description "Title")
- [2. Instructions](#2-instructions "Title")
- [3. Contents](#3-contents "Title")
- [4. Scans](astropath/scans/#4-scans "Title")
   - [4.1. Description](astropath/scans/#41-description "Title")
   - [4.2. Definitions](astropath/scans/#42-definitions "Title")
     - [4.2.1. Identification Definitions](astropath/scans/#421-identification-definitions "Title")
     - [4.2.2. Path Definitions](astropath/scans/#422-path-definitions "Title")  
   - [4.3. Scanning, Verifying Complete, and Adding BatchIDs](astropath/scans/#43-scanning-verifying-complete-and-adding-batchids "Title")
     - [4.3.1. Specimen_Table](astropath/scans/#431-specimen_table "Title")
     - [4.3.2. SampleNames (Patient # or M Numbers)](astropath/scans/#432-samplenames-patient--or-m-numbers "Title")
     - [4.3.3. Control TMA Conventions](astropath/scans/#433-control-tma-conventions "Title")
     - [4.3.4. Whole Slide Scanning](astropath/scans/#434-whole-slide-scanning "Title")
       - [4.3.4.1. Preparing 20% Overlap](astropath/scans/#4341-preparing-20-overlap "Title")
       - [4.3.4.2. Scan the Control TMA](astropath/scans/#4342-scan-the-control-tma "Title")
       - [4.3.4.3. Scan the Whole Slide Specimens](astropath/scans/#4343-scan-the-whole-slide-specimens "Title")
       - [4.3.4.4. Creating a Whole Slide Scanning Plan](astropath/scans/#4344-creating-a-whole-slide-scanning-plan "Title")
       - [4.3.4.5. Important Scanning Notes](astropath/scans/#4345-important-scanning-notes "Title")
     - [4.3.5. BatchIDs](astropath/scans/#435-batchids "Title")
     - [4.3.6. Batch Tables](astropath/scans/#436-batch-tables "Title")
     - [4.3.7. MergeConfig Tables](astropath/scans/#437-mergeconfig-tables "Title") 
   - [4.4. AstroPath_Processing Directory and Initializing Projects](astropath/scans/#44-astropath_processing-directory-and-initializing-projects "Title")
     - [4.4.1. AstroPath_Processing Directory](astropath/scans/#441-astropath_processing-directory "Title")
     - [4.4.2. Initializing Projects](astropath/scans/#442-initializing-projects "Title")
   - [4.5. Directory Organization](astropath/scans/#45-directory-organization "Title")
- [5. HPF Processing](astropath/hpfs/#5-hpfs "Title")
  - [5.1. Description](astropath/hpfs/#51-description "Title")
  - [5.2. Contents](astropath/hpfs/#52-contents "Title")
  - [5.3. AstroIDGen](astropath/hpfs/AstroidGen/#53-astroid-generation "Title")
    - [5.3.1. Description](astropath/hpfs/AstroidGen/#531-description "Title")
    - [5.3.2. Important Definitions](astropath/hpfs/AstroidGen/#532-important-definitions "Title")
    - [5.3.3. Workflow](astropath/hpfs/AstroidGen/#533-workflow "Title")
  - [5.4. Transfer Daemon](astropath/hpfs/TransferDaemon/#54-transfer-daemon "Title")
    - [5.4.1. Description](astropath/hpfs/TransferDaemon/#541-description "Title")
    - [5.4.2. Important Definitions](astropath/hpfs/TransferDaemon/#542-important-definitions "Title")
    - [5.4.3. Code Input](astropath/hpfs/TransferDaemon/#543-code-input "Title")
    - [5.4.4. Workflow](astropath/hpfs/TransferDaemon/#544-workflow "Title")
      - [5.4.4.1. Initial Transfer](astropath/hpfs/TransferDaemon/#5441-initial-transfer "Title")
      - [5.4.4.2. MD5 Check](astropath/hpfs/TransferDaemon/#5442-md5-check "Title")
      - [5.4.4.3. Compression Into Backup](astropath/hpfs/TransferDaemon/#5443-compression-into-backup "Title")
      - [5.4.4.4. Source File Handling](astropath/hpfs/TransferDaemon/#5444-source-file-handling "Title")
    - [5.4.5. Notes](astropath/hpfs/TransferDaemon/#545-notes "Title") 
   - [5.5 Flatfield]
   - [5.6 Segmentation \ Classification]
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
