# <div align="center"> AstropathPipeline </div>
#### <div align="center">***v0.05.0000***</div>
#### <div align="center">Created by: Benjamin Green, Jeffrey S. Roskes, Margaret Eminizer, Richard Wilton, Sneha Berry, Nicolas Giraldo-Castillo, Tricia Cottrell, Elizabeth Engle, Janis Taube, Alex Szalay </div>
#### <div align="center">The Johns Hopkins University Bloomberg~Kimmel Institute for Cancer Immunotherapy</div>
#### <div align="center">Correspondence to: bgreen42@jhu.edu</div>

## ***Description***
The Astropath pipeline was designed to automate the processing of whole slide multiplex immunoflourecence histopathology image data, taken by Akoya’s Vectra imaging platform, from the microscope to database. The automated process begins after whole slide scans have been captured by the microscope and manually verified complete. Code is divided into three main stages; defined as ```hpf```, ```slide```, and ```sample``` level processing. In the ```hpf``` (or high powered field) processing stage, images are reorganized, corrected for camera\ imaging effects, and segmented\ phenotyped. Here images are mainly processed individually. In the next processing stage, aptly named ```slide``` the data is stiched together into a whole slide and the slides are annotated by a pathologist. Next, slides across a cohort are corrected for batch to batch and loaded into a database. Here the image, cell, and annotation data of each whole slide image is linked its clinical information thus providing a completed sample. Code for each stage is organized into its own folder, with each folder containing a particular set of modules. Each module is organized separately in subfolders and described in this documnetation. An overview of the current pipeline can be seen [here](https://github.com/AstropathJHU/AstroPathPipeline/blob/main/AstroPathPipeline.pdf).

## ***Contents***
1. [Description](#description "Title")
2. [Contents](#contents "Title")
3. [Setting Up, Organization, and Scanning](scans/#3-setting-up-organization-and-scanning "Title")
   - [3.1. Description](scans/#31-description "Title")
   - [3.2. Definitions](scans/#32-definitions "Title")
     - [3.2.1. Identification Definitions](scans/#321-identification-definitions "Title")
     - [3.2.2. Path Definitions](scans/#322-path-definitions "Title")  
   - [3.3. Scanning, Verifying Complete, and Adding BatchIDs](scans/#33-scanning-verifying-complete-and-adding-batchids "Title")
     - [3.3.1. Specimen_Table](scans/#331-specimen_table "Title")
     - [3.3.2. SampleNames (Patient # or M Numbers)](scans/#332-samplenames-patient--or-m-numbers "Title")
     - [3.3.3. Control TMA Conventions](scans/#333-control-tma-conventions "Title")
     - [3.3.4. Whole Slide Scanning](scans/#334-whole-slide-scanning "Title")
       - [3.3.4.1. Preparing 20% Overlap](scans/#3341-preparing-20-overlap "Title")
       - [3.3.4.2 Scan the Control TMA](scans/#3342-scan-the-control-tma "Title")
       - [3.3.4.3 Scan the Whole Slide Specimens](scans/#3343-scan-the-whole-slide-specimens "Title")
       - [3.3.4.4 Creating a Whole Slide Scanning Plan](scans/#3344-creating-a-whole-slide-scanning-plan "Title")
       - [3.3.4.5 Important Scanning Notes](scans/#3345-important-scanning-notes "Title")
     - [3.3.5. BatchIDs](scans/#335-batchids "Title")
     - [3.3.6. Batch Tables](scans/#336-batch-tables "Title")
     - [3.3.7. MergeConfig Tables](scans/#337-mergeconfig-tables "Title") 
   - [3.4. AstroPathProcessing Directory and Initializing Projects](scans/#34-astropathprocessing-directory-and-initializing-projects "Title")
     - [3.4.1 AstroPath_Processing Directory](scans/#341-astropath_processing-directory "Title")
     - [3.4.2 Initializing Projects](scans/#342-initializing-projects "Title")
   - [3.5 Directory Organization](scans/#35-directory-organization "Title")
4. [HPF Processing]
   - [4.1 Description]
   - [4.2 Transfer]
   - [4.3 Flatfield]
   - [4.4 Segmentation \ Classification]
5. [Slide_Processing]
   - [5.1 Description]
   - [5.2 Align]
   - [5.3 Annotations]
   - [5.4 Cell Geometries]
6. [Sample_Processing]
   - [6.1 Description]
   - [6.2 Control TMAs]
   - [6.3 Calibration]
   - [6.4 Upload to Database]
