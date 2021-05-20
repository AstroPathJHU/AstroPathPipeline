# 4.5. AstroPath Processing Directory and Initializing Projects
The code is driven by the files located in a main processing folder, named the ```<Mpath>```. These files are described below followed by descriptions of the respectve columns. For columns without definitions provided, please check [4.3.](Definitions.md). After a description of the directory and files included, instructions for intializing projects into the pipeline are provided. The JHU directory is located at *\\bki04\astropath_processing*

## 4.5.1. AstroPath Processing Directory
- *AstropathCohortsProgress.csv*: 
  - This file contains information on the project's analysis status and important experimental variables. This table is manually updated. The file has the following columns:
  ```
  Project, Cohort, Dpath, Dname, Machine, Method, Panel, Tissue, StainConfig, Stain, Scan, Inform, Merged, QC, Annotations, ReadyForDB, DBLoad
  ```
  - ```Stain[string]```: The staining status of the panel. Should be indicated as blank (not started), *Started*, or *Done*.
  - ```Inform[string]```: The status of the inform algorithm for classification and segmentation. Should be indicated as blank (not started), *Started*, or *Done*.
  - ```Merged[string]```: The status of the inform processing and merge. Should be indicated as blank (not started), *Started*, or *Done*.
  - ```QC[string]```: The status of the quality control assessment of samples. Should be indicated as blank (not started), *Started*, or *Done*.
  - ```Annotations[string]```: Whether or not the slide annotations have been created for a panel. ***Annotation directions can be found in***. Should be indicated as blank (not started), *Started*, or *Done*.
  - ```ReadyForDB[string]```: Indicates whether final checks for the manual interaction steps with the data have been complete. ***A full checklist is still in progress.***. Should be indicated as blank (not started), *Started*, or *Done*.
  - ```DBLoad[string]```: The current status of the database load. Should be indicated as blank (not started), *Started*, or *Done*.
- *AstropathConfig.csv*
  - This file contains information on drive space and how different options should be handled in the code. This table is manually updated, except for the ```Space_TB``` columns which is updated by the ```mergeloop``` module. The file contains the following columns:
```
Project, Cohort, Dname, Delete, Space_TB, Process_Merge
```
 - ```Delete[string]```: An option of the ```transferdeamon``` module, input is either *yes* or *no* and is not case sensitive. Indicates whether or not to automatically delete data from the source directory after it is transferred to the ```<dpath>\<dname>```, compressed to the ```<Cpath>\<Dname>``` location, and verfied using *checksums*. ([Path name definitions](Definitions.md/#432-path-definitions))
 - ```Space_TB[float]```: updated by the ```mergeloop``` module to verify that there is enough space on the server for different processes, add a zero for intializing a project
 - ```Process_Merge[string]```: An option of the ```mergeloop``` module, input is either *yes* or *no* and is not case sensitive. Indicates whether or not to run the *mergeloop* module for that project. Once data is validated through QC ([documentation here]()) it is usually best to turn off the ```mergeloop``` processing for that ```Project```.
- *AstropathControldef.csv*
  - This file is updated by the code and records the control information. Users do not need to open, create, or modify this file. The columns are as follows:
  ```
  cohort, project, ctrlid, SlideID, Scan, BatchID
  ```
  - ```ctrlid[int]```: This is the control ID.
- *AstropathPaths.csv*
  - This file records pertinent path information for each ```Project```. The columns are as follows:
  ```
  Project, Dpath, Dname, Spath, Cpath, FWpath
  ```
- *AstropathSampledef.csv*
  - This file is created by the code to record the names of the specimens as they are injested into the database. This file is created and updated by the code. The columns are as follows:
  ```
  SampleID, SlideID, Project, Cohort, Scan, BatchID, isGood
  ```
  - ```SampleID[int]```: these ids are defined for the database only and are not the same as ```SampleNames```s defined in previous documentation.
  - ```isGood[int]```: whether (1) or not (0) the sample will be injested into the database
- *AstropathAPIDdef.csv*
   - This file relates the ```SampleName```s defined for the scanning process and the ```SlideID```s defined during the transfer process, used for the remainder of the processing in the ```hpfs``` workflow. This file is updated by the ```AstroIDGen``` module and should not be manually modified. A copy of this file for the ```Project``` level is kept at the *upkeep_and_progress* folder. The columns are as follows:
   ```
   SlideID, SampleName, Project, Cohort, BatchID
   ```
## 4.5.2. Initializing Projects
To initialize ```Project```s; first create the ```<Spath>```, ```<Dpath>\<Dname>```, and ```<Cpath>```. Next, add the ```Project``` to the *AstropathCohortsProgress.csv* with all the pertient information and the next available ```Project``` number, afterward update the *AstropathConfig* and *AstropathPaths* file with the new ```Project```. Additional details on the paths is below, the code should begin processing the slides as long as slides have been added to the *Specimen_Table.xlsx* and the scanning protocols have been adhered to as defined in [4.4.](ScanningInstructionsIntro.md).

The ```<Spath>``` folder is created on the scanning computer where the slides are scanned into. The folders are usually labeled *Clinical_Specimen_XX*, where the *XX* indicates a numeric value or unique lettering. Examples of these scanning folders incude *Clinical_Specimen_2* and *CLinical_Specimen_BMS_01*. In the JHU processing pipeline, this folder is backed up every night, using commercially available software, to a network server with significant storage capacity (~80TB). Once slide scans are completed they can be deleted from the local computer, the rest of the processing takes place on the network server location. In this way, the local computers never run in storage issues. *NOTE*: The fully qualified path for this scanning folder on the network server is designated as the ```<Spath>```. 

For the ```<Dpath>\<Dname>``` folder, usually ```<Dpath>``` is a server or drive name. The ```<Dname>``` is the *Clinical_Specimen_XX* folder name as described above. This folder is usually shared over the network. An example of one such combination is *\\bki04\Clinical_Specimen*. 

The ```<Cpath>``` is a separate network folder, located on a different server than the ```<Dpath>```. Slides are saved under a folder with the ```<Dname>``` indicator. 


