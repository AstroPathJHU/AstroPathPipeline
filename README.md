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
3. [Setting Up, Organization, and Scanning](https://github.com/AstroPathJHU/AstroPathPipeline/blob/main/scans/readme.md#section-3-setting-up-organization-and-scanning "Title")
   - [3.1 Description](https://github.com/AstroPathJHU/AstroPathPipeline/blob/main/scans/readme.md#section-31-description "Title")
   - [3.2 AstroPathProcessing Directory and Initializing Projects]
   - [3.3 Scanning, Verifying Complete, and Adding BatchIDs]
4. [HPF Processing](#section-4-hpf-processing "Title")
   - [4.1 Description](#section-41-description "Title")
   - [4.2 Transfer](#section-42-transfer "Title")
   - [4.3 Flatfield](#section-43-flatfield "Title")
   - [4.4 Segmentation \ Classification](#section-44-segmentation-classification "Title")
5. [Slide_Processing](#section-5-slide-processing "Title")
   - [5.1 Description](#section-51-description "Title")
   - [5.2 Align]
   - [5.3 Annotations]
   - [5.4 Cell Geometries]
6. [Sample_Processing](#section-6-sample-processing "Title")
   - [6.1 Description](#section-61-description "Title")
   - [6.2 Control TMAs]
   - [6.3 Calibration]
   - [6.4 Upload to Database]
