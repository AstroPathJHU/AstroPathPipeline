# 5.9. Mergeloop

## 5.9.1. Description
This module servers to track processing, interact with the inForm® processing queues and subsequently merge\ generate quality control images for each slide using the code MaSS (description found [here](MaSS#merge-a-single-sample-mass)). The module loops through each project in the *AstroPathCohortsProgresss.csv* file, tracking or updating all slides then continuing to the next project. Importantly, the code uses the *MergeConfig_NN.xlsx* files (defined in [4.4.8.](../../scans/docs/scanning/MergeConfigTables.md#448-mergeconfig-tables)) to intialize a project level inForm queue as well as antibody specific processing folders in the tmp_inform_data folder. The code will not process a cohort without these files as its main function is serve as an interaction between the user and the inform processing tasks. Additional details on what the code tracks, location of different files and processing steps can be found in this documentation. A typical user instruction is included [here](../docs/TypicalUserInstructions.md#532-typical-user-instructions) for people who will not be monitoring the code but will be using the pipeline to process slides.

## 5.9.2. Important Definitions

- ```<upkeep_and_progress>```: This folder contains all the necessary project related documents for the inForm® or practical user of the pipeline. 
   - add all additional files outside of those maintained by the code to this folder. Adding files or new folders to the main ```<Dname>``` directory may crash the processing or cause unexpected results
- *inform_queue.csv*: This file is the project level inform queue. 
   - The code transfers new tasks to the main inform queue and completed task infomation back to this queue.
   - adding tasks to this file is described in further documentation in [5.10.5.](../inform_processing/docs/AddingSlidestotheinFormQueue.md)
- *Main_inform_queue.csv*: The main queue is held in the ```<Mpath>``` location. Additional copies of this queue are created in the ```<upkeep_and_progress>``` folder of each project for reference
   - opening and modifying the main queue in the ```<Mpath>``` directory can cause the file to become corrupt resulting in loss of archived processes.
   - it is recommended that only the project level queues are modified.
   - it is also recommended that backup copies of the main queue are manually maintained
- ```<tmp_inform_data>```: this is subfolder under the ```<Dpath>\<Dname>```, 
  - contains processing subfolders for each antibody. The tasks run in the ```inform_processing``` module are run an distributed into this folder then dessiminated to subfolders under the ```<SlideID>```. 
  - The ```<Project_Development>``` folder is also in the ```<tmp_inform_data>``` folder
- ```<Project_Development>```: this is a sub folder of ```<tmp_inform_data>```. Algorithms to be processed by the inForm® processing code should be placed here. The code will not find algorithms placed elsewhere or in subfolders.
- *samples_summary.xlsx*: this spreadsheet records and maintains useful information for processing of each slide
- inForm® data - sample level file formatting:
```
+--	DIR\ inform_data <br>
| +-- Component_Tiffs<br>
| +-- Phenotyped:	add a folder for each  Antibody (ABx) in the panel<br>
| | +-- ABX1 (e.g.	CD8)<br>
| | +-- ABX2 (e.g.CD163)<br>
| | +-- ABX3 (e.g.FoxP3)<br>
| | +-- ABX4 (e.g.	Tumor)<br>
| | +-- ABX5 (e.g.PD1)<br>
| | +-- ABX6 (e.g.PDL1)<br>
```

*Note*: Additional directory information can be found [here](../../scans/docs/DirectoryOrganization.md#46-directory-organization)

## 5.9.3. Instructions
The code should be launched through MATLAB. To start download the repository to a working location. Next, open a new session of matlab and add the ```AstroPathPipline``` to the matlab path. Then use the following to launch:   
``` 
mergeloop(<Mpath>)
```
- ```<Mpath>[string]```: the full path to the directory containing the *AstropathCohortsProgress.csv* file
   - description of this file can be found [here](../../scans/docs/AstroPathProcessingDirectoryandInitializingProjects.md#451-astropath-processing-directory)

*Note:* For the code to process successfully be sure to create the merge configuration files (see [4.4.8.](../../scans/docs/scanning/MergeConfigTables.md#448-mergeconfig-tables)).

## 5.9.4. Workflow
This code starts by extracting the cohorts from the *AstropathCohortProgress.xlsx* spreadsheet. Each cohort is then looped over with the following steps:
- We check the space available on the ```<Dpath>\<Dname>``` processing drive and write this into the *AstropathConfig.csv* file (description found [here](../../scans/docs/AstroPathProcessingDirectoryandInitializingProjects.md#451-astropath-processing-directory))
- Check the ```<Process_Merge>``` variable in the *AstropathConfig.csv* file, if *Yes* continue processing otherwise skip to the next cohort
- Using the *MergeConfig_NN.xlsx* files, extract the antibodies stained on the slides
  - check or create folders for processing of different inform antibody\ sample tasks under the ```<tmp_inform_data>``` folder
  - Be sure that the folders are intialized with at least one extra folder to ensure smooth processing with the *inform_processing* module
- Initialize the samples spreadsheet & variables
- Next we loop through each sample extracting specified information
  - Information extracted includes: ```<Scan>, <BatchID>, <ScanDate>, <Im3files>, <expected Im3 files>, <flatw_im3s>``` etc. 
  - Integral in this step is when we process the merge files, here we check the ```<tmp_inform_data>``` antibody folders (defined in previous steps) for folders with completed *inForm* processing tasks. 
    - These tasks are denoted finished when inForm adds the *Batch.txt* which occurs only after slides are finished. 
    - If the folders contain both this *Batch.txt* file and inForm output for a specified slide (denoted by the file names) the files are transferred to the specimen folder under : ```<base>\<SlideID>\<inform_data>```
    - Files are organized according to the format in the [Important Definitions](#592-important-definitions) section
    - Once files *for each antibody* are processed the code merges all output to a ```<base>\<SlideID>\<inform_data>\<Phenotyped>\<Results>\<Tables>``` folder additional details on how to export files for different antibodies and their settings can be found in the *inform_processing* module
  - Once files are merged the QA\QC files are generated here  ```<base>\<SlideID>\<inform_data>\<Phenotyped>\<Results>\<QAQC>```
    - additional details on these files can be found in the *MaSS* documentation [here](MaSS#merge-a-single-sample-mass)
- After all specimens are looped through the code builds the *samples_summary.xlsx* spreadsheet and exports it
- Finially the code checks the *inForm_queue.csv* at the project level, compares it to the *inForm_queue.csv* in the ```<Mpath>``` directory. New tasks are added to the *main* *inForm_queue.csv* and completed tasks are added to the project level *inForm_queue.csv*. A copy of the *main* *inForm_queue.csv* is added to the project level and labeled *Main_inForm_queue.csv* for referencing.
