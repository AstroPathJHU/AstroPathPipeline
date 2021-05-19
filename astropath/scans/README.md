# 4. Scans
## 4.1. Description
The *AstroPath Pipeline* requires that a number of experimental protocols are understood and followed for the code to work. These protocols include methods for slide scanning, slide naming, reagent tracking, and directory organization. This section of the documentation describes these protocols and provides important definitions for terms used throughout the *AstroPath Pipeline* documentations. Additionally, there is a main directory, referred to as the ```<Mpath>```, which contains a set of csv files. These files contain pertinent information that drives processing such as directory locations, machine names, slide names, and project identifiers. The csv files and methods for adding projects to the pipeline are defined here [4.4](#44-astropath_processing-directory-and-initializing-projects "Title"). To reduce the length of this page, documentation was written into different pages and linked here by a table of contents.

## 4.2. Contents
 - [4.3. Definitions](docs/Definitions)
   - [4.3.1. Indentification Definitions](docs/Definitions.md/#431-identification-definitions)
   - [4.3.2. Path Definitions](docs/Definitions.md/#432-path-definitions)
 - [4.4. Scanning Instructions](docs/ScanningInstructions.md)
   - [4.4.1. Scanning Contents](docs/ScanningInstructions.md/#441-scanning-contents)
   - [4.4.2. Specimen_Table](docs/scanning/SpecimenTable.md)



#
## 4.6. Directory Organization 
The following folders should exist in the ```<Dpath>\<Dname>``` for processing. The code initializes these folders at the start of the pipeline.
1.	```upkeep_and_progress```
    - For any upkeep and progress tracking files
    - Location of the *AstropathAPIDdef_PP.csv* files, where ```PP``` indicates the numeric project id
2.	```flatfield```
    - Location of the flatfield parameter files
    - These files are named ```Flatfield_BatchID_BB.bin```, replacing the ```BB``` for the appropriate batch id.
3.	```logfiles```
    - Project level log files for the astropath pipeline 
4.	```Batch```
    - The batch and merge tables
    - These tables are described in further documentation located [here](#435-batchids "Title") repository
5.	```Clinical```
    - Location of the clinical table
    - These tables should be labeled as *Clinical_Table_Specimen_CSID_MMDDYYYY.csv*, where the ```CSID``` indicates the number on the ```<Dname>``` folder. 
    - We always use the clinical table with the most recent date in the data upload
6.	```Ctrl```
    - Location of control TMA data output
7.	```dbload```
    - Location of the files used for the database upload
8.	```tmp_inform_data```
    - Location of the inform data output and inform algorithms used. **Additional information on this folder is provided in the hpf processing documentation.**
9.	```reject```
    - Location of the rejected slides
