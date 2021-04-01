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
   - [3.1 Description](scans/#31-description "Title")
   - [3.2 AstroPathProcessing Directory and Initializing Projects](https://github.com/AstroPathJHU/AstroPathPipeline/blob/main/scans/#section-32-astropathprocessing-directory-and-initializing-projects "Title")
   - [3.3 Scanning, Verifying Complete, and Adding BatchIDs](https://github.com/AstroPathJHU/AstroPathPipeline/blob/main/scans/#section-33-scanning-verifying-complete-and-adding-batchids "Title")
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
