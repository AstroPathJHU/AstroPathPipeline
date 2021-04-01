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
The Astropath pipeline was designed to automate the processing of whole slide multiplex immunoflourecence histopathology image data, taken by Akoya’s Vectra imaging platform, from the microscope to database. The automated process begins after whole slide scans have been captured by the microscope and manually verified complete. Code is divided into three main stages; defined as ```hpf```, ```slide```, and ```sample``` level processing. In the ```hpf``` (or high powered field) processing stage, images are reorganized, corrected for camera\ imaging effects, and segmented\ phenotyped. Here images are mainly processed individually. In the next processing stage, aptly named ```slide``` the data is stiched together into a whole slide and the slides are annotated by a pathologist. Next, slides across a cohort are corrected for batch to batch and loaded into a database. Here the image, cell, and annotation data of each whole slide image is linked its clinical information thus providing a completed sample. Code for each stage is organized into its own folder, with each folder containing a particular set of modules. Each module is organized separately in subfolders and described in this documnetation. An overview of the current pipeline can be seen [here](https://github.com/AstropathJHU/AstroPathPipeline/blob/main/AstroPathPipeline.pdf).

## 2. Installation

To install the python parts of the code, first check out the repository, enter its directory, and run
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

## 3. Contents
- [1. Description](#1-description "Title")
- [2. Contents](#2-contents "Title")
- [3. Setting Up, Organization, and Scanning](astropath/scans/#3-setting-up-organization-and-scanning "Title")
   - [3.1. Description](astropath/scans/#31-description "Title")
   - [3.2. Definitions](astropath/scans/#32-definitions "Title")
     - [3.2.1. Identification Definitions](astropath/scans/#321-identification-definitions "Title")
     - [3.2.2. Path Definitions](astropath/scans/#322-path-definitions "Title")  
   - [3.3. Scanning, Verifying Complete, and Adding BatchIDs](astropath/scans/#33-scanning-verifying-complete-and-adding-batchids "Title")
     - [3.3.1. Specimen_Table](astropath/scans/#331-specimen_table "Title")
     - [3.3.2. SampleNames (Patient # or M Numbers)](astropath/scans/#332-samplenames-patient--or-m-numbers "Title")
     - [3.3.3. Control TMA Conventions](astropath/scans/#333-control-tma-conventions "Title")
     - [3.3.4. Whole Slide Scanning](astropath/scans/#334-whole-slide-scanning "Title")
       - [3.3.4.1. Preparing 20% Overlap](astropath/scans/#3341-preparing-20-overlap "Title")
       - [3.3.4.2. Scan the Control TMA](astropath/scans/#3342-scan-the-control-tma "Title")
       - [3.3.4.3. Scan the Whole Slide Specimens](astropath/scans/#3343-scan-the-whole-slide-specimens "Title")
       - [3.3.4.4. Creating a Whole Slide Scanning Plan](astropath/scans/#3344-creating-a-whole-slide-scanning-plan "Title")
       - [3.3.4.5. Important Scanning Notes](astropath/scans/#3345-important-scanning-notes "Title")
     - [3.3.5. BatchIDs](astropath/scans/#335-batchids "Title")
     - [3.3.6. Batch Tables](astropath/scans/#336-batch-tables "Title")
     - [3.3.7. MergeConfig Tables](astropath/scans/#337-mergeconfig-tables "Title") 
   - [3.4. AstroPathProcessing Directory and Initializing Projects](astropath/scans/#34-astropathprocessing-directory-and-initializing-projects "Title")
     - [3.4.1. AstroPath_Processing Directory](astropath/scans/#341-astropath_processing-directory "Title")
     - [3.4.2. Initializing Projects](astropath/scans/#342-initializing-projects "Title")
   - [3.5. Directory Organization](astropath/scans/#35-directory-organization "Title")
 - [4. HPF Processing]
   - [4.1 Description]
   - [4.2 Transfer]
   - [4.3 Flatfield](astropath/hpfs/flatfielding/)
   - [4.4 Image correction](astropath/hpfs/image_correction/)
   - [4.4 Segmentation \ Classification]
 - [5. Slide_Processing]
   - [5.1 Description]
   - [5.2 Align]
   - [5.3 Annotations]
   - [5.4 Cell Geometries]
 - [6. Sample_Processing]
   - [6.1 Description]
   - [6.2 Control TMAs]
   - [6.3 Calibration]
   - [6.4 Upload to Database]
