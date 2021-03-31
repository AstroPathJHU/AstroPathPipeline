# <div align="center"> AstropathPipeline </div>
#### <div align="center">***v0.05.0000***</div>
#### <div align="center">Created by: Benjamin Green, Jeffrey S. Roskes, Margaret Eminizer, Richard Wilton, Sneha Berry, Nicolas Giraldo-Castillo, Tricia Cottrell, Elizabeth Engle, Janis Taube, Alex Szalay </div>
#### <div align="center">The Johns Hopkins University Bloomberg~Kimmel Institute for Cancer Immunotherapy</div>
#### <div align="center">Correspondence to: bgreen42@jhu.edu</div>

## ***Section 1: Description***
The Astropath pipeline was designed to automate the processing of whole slide multiplex immunoflourecence histopathology image data, taken by Akoya’s Vectra imaging platform, from the microscope to database. The automated process begins after whole slide scans have been captured by the microscope and manually verified complete. Code is divided into three main stages; defined as ```hpf```, ```slide```, and ```sample``` level processing. In the ```hpf``` (or high powered field) processing stage, images are reorganized, corrected for camera\ imaging effects, and segmented\ phenotyped. Here images are mainly processed individually. In the next processing stage, aptly named ```slide``` the data is stiched together into a whole slide and the slides are annotated by a pathologist. Next, slides across a cohort are corrected for batch to batch and loaded into a database. Here the image, cell, and annotation data of each whole slide image is linked its clinical information thus providing a completed sample. Code for each stage is organized into its own folder, with each folder containing a particular set of modules. Each module is organized separately in subfolders and described in this documnetation. An overview of the current pipeline can be seen [here](https://github.com/AstropathJHU/AstroPathPipeline/blob/main/AstroPathPipeline.pdf).

## ***Section 2: Contents***
1. [Description](#section-1-description "Title")
2. [Contents](#section-2-contents "Title")
3. [Setting Up, Organization, and Scanning](#section-3-setting-up-organization-and-scanning "Title")
   - [3.1 Description](#section-31-description "Title")
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

## ***Section 3: Setting Up, Organization, and Scanning***
### *Section 3.1: Description*
The code relies on a main directory which contains key csv files in order to drive processing.
### *Section 3.2:  AstroPathProcessing Directory and Initializing Projects*
The Processing Specimens directory is located on the c drive of bki05 and is a shared network folder on bki05 (so it can be accessed using \\bki05\Processing_Specimens). In this directory There are a few important spreadsheets. 
1.	The “inForm_queue.csv” aids in monitoring which specimens are being processed through inForm and is used to provide information to the Virtual Machines on which specimen and antibody should be processed next. This spreadsheet it for record keeping purposes and should never be manually modified. 
2.	The “process_flatw_queue.csv” aids in monitoring which specimens need to be flat fielded. This spreadsheet records pertinent information for flat fielding of the slides. Further information in section 6. 
3.	The “Paths.csv” is the master document which tells each part of the code which directories should be processed and where that directories data should be processed to. To add a directory to the pipeline all columns must be filled out. An example of this spreadsheet is supplied below, with descriptions on each column following and tips on how to initialize the column when adding a new panel. Remember to only keep this spreadsheet open for as limited a time as possible since it is referenced to multiple times throughout the pipeline codes. Changing the order of the columns may cause errors in the transfer demon.

-	Source_Path: This is the source directory of the data and usually indicates a subdirectory from where the microscope will back up to such as: “\\tme1\VectraPolaris\Vectra Polaris 1 Scanning\Clinical_Specimen_7” 
-	Main_Path: This is the main path of the clinical specimen where all processing data will be stored. This folder is usually a shared network folder on one of the bki drives. This folder should be created and shared over the network with any specific users that need access to the data as well as the bki user group corresponding to that server. For bki04, use the Taubelab group; for bki03, use the users group. Be sure to create and share the drive before adding the directory to the spreadsheet. Tips on how to create and share the drives can be found in the ‘DriveSharing.docx’ Protocol located on the halo1 server. 
-	ColdStorage_Path: This is where compressed copies of the im3 image files and final Tables\ component data tiffs are stored after the pipeline has finished processing. This folder should be located in one of the cold storage folders on bki03 and should be created before adding the directory to the spreadsheet.
-	DeleteSource_Images: This indicates whether or not the original source data should be deleted from the source directory after is has been transfer to the main path and compressed to the cold storage path. Should be indicated as either ‘YES’ or ‘NO’.
-	Machine: This is the unique name of the machine where the scans were scanned. Check with Liz to verify the machine names before adding them to the spreadsheet.
-	Flatw_Path: This is the path for the flatw files trimmed to layer 1, necessary for the alignment codes. This path should preferably located on a different drive from the main path to improve pipeline performance. The path should be a shared network path created before adding the directory to the spreadsheet. 
-	MaSS_Path: This is the path which indicates where the MaSS and CreateImageQAQC installation directories are located. Should be the same for all directories.
-	DriveSpace_TB: This is the amount of available space on the drive. 
-	Process_Merge: Whether or not to run this data set through the pipeline indicated as either ‘yes’ or ‘no’, case insensitive.
-	Panel: Name of the panel type; from the spreadsheets located in the protocols folder 
-	Tissue: Name of the tissue; from the spreadsheets located in the protocols folder 
-	Method: Automated or Manual Staining was performed; from the spreadsheets located in the protocols folder 
-	Stain Configuration: The unique identifier for the Panel and stain Method; from the spreadsheets located in the protocols folder 
-	Cohort: A unique identifier for which cohort the staining belonged to

### *Section 3.3: Scanning, Verifying Complete, and Adding BatchIDs*
Directions for scanning can be found in the protocol folder. Key things to note that affect the pipeline include; naming convention for both the slides and the controls, all images are scanned with 20% overlap and are the same size, BatchID excel files are created for each batch in the clinical specimen ‘Main_Path’ before the BatchID files are added, and that most of the code runs off of the annotation.xml automatically created when drawing the annotations in Phenochart.  
Naming Convention: To avoid errors in the pipeline the slide names should only include alphanumeric characters and no special characters, ie (“ [, ], @, $, -, &”), the only exceptions being underscores and periods. Furthermore, for the code to correctly process the control data the slide name must contain the string ‘Control’, this can be located anywhere in the slide name. If this string is not included the code will attempt to process these slides similarly to the specimen files, starting with the flat fielding. The flat fielding protocol may fail due to the varying image sizes the control tma is usually scanned with. As a side note, flat fielding the control data is not required because the correction is established across batches and variation within the images of a batch is assumed to be the same across batches. 
Scanning with 20% overlap: In order to properly register the images and to measure image error, we use a 20% overlap scanning pattern. When drawing the ROIs it is important to heed attention to the image sizes and to remove images of blank regions at the edge of the tissue.
Blank regions are unnecessary to scan and often give inform trouble, because of this it is best to delete annotations that contain zero cells. In batch processing, inform scales each of the image layers to the image in order to create a segmentation map. If there is no tissue in the image or if the DAPI is too light, inform scales the image so that the noise is measure as single; we have seen this result in fields segmented to that there 10,000 cells or images of 500-1000 pixels in size. These errors can be misleading if they make there way into the database. We have since written in checks to attempt to filter these fields out of the database upload. 
If the images in the slides are different sizes we cannot apply illumination or image warping correction since both of these corrections are dependent on the size of the image and the microscope configuration they were imaged with. Akoya’s phenochart software is developed to decrease scanning time and is optimized for HPF analysis. This does not always correlate to the optimal way for whole slide analysis. One of the major feature’s that causes issues is when a ROI is very small the corresponding HPFs merge into a single very large image, this can be seen in the annotation file just after drawing the ROI. In order to be sure all images are of the same size, this image must be deleted and a larger ROI must be drawn.
BatchID excel files: These files are very important in determining which antibodies are tracked by the pipeline, their corresponding names, and how the antibodies are handled in the MaSS protocol. If a BatchID file is not created for a batch the slides from that batch will not be processed properly. A description of how to create this document for each batch and what each column means is located in the protocols folder. 
Annotation files: Akoya’s imaging platform uses these files to keep a record of when each image annotation was created, deleted, acquired, or failed. We use this record to track how many images should exist in a directory, what the images are named, and when they were scanned. Altering the im3 image names renders this document invalid and changes to this document prohibits opening the QPTiff image in phenochart. If changes must be made to the file names, those changes should be propagated to the annotation file (both it’s name and the image names in the file). A version without changes inside the document should be saved with the new name and the tag ‘-original’. 
To verify that the scans completed, open the QPtiff and check that most of the im3s scanned properly. If a large number of the HPFs failed, create a scan2 folder, copy all files except the MSI folder from scan1 to scan2, delete the annotation.xml documents, change the corresponding names to scan2, redraw the annotation, and rescan. If only a few HPFs failed, or none failed, the scan is complete and the BatchID.txt file should be created for that slide. The BatchID file relates all slide from a given staining batch together, important to apply proper correction factors, and lets the transfer demon know that these slides are ready to be transferred to the server for further processing. The file should only contain the batch number for the given slide and should be named BatchID.txt (case sensitive). The batch file should be placed inside the ‘Scan’ folder to be processed. Once placed, that slide is in the pipeline. 

## ***Section 4: HPF Processing***
### *Section 4.1: Description*

After the whole slide scans have been captured by the microscope and manually verified complete, the user adds the BatchID.txt file to initiate that slide into the pipeline and the HPF processing stage begins. The images are then transferred to the bki servers where they are reorganized and backed-up (```TransferDeamon```). The image data is corrected for both image warping and illumination (```Flatfield```). The user must then create segmentation and phenotype algorithms for each antibody stained on the panel (```Segmentation and Classification```). The names of these algorithms are used as input into a queue which tracks and initiates inform processing of all images in a slide via provided workers. Once all antibodies are processed for every image in a slide the separate antibody data must be merged and placed in a format that can be easily uploaded to a database. This merge is carried out by ```MaSS``` and quality control images are created by the ```CreateImageQAQC``` submodule. Once the images have been properly quality controlled, the data can be uploaded to a database or analyzed separately. Section 4 describes each of these steps in specific detail.

### *Section 4.2: Transfer*

### *Section 4.3: Flatfield*

### *Section 4.4: Segmentation \ Classification*

### *Section 4.5: InForm and QC*

## ***Section 5: Slide Processing***
### *Section 5.1: Description*

## ***Section 6: Sample Processing***
### *Section 6.1: Description*
