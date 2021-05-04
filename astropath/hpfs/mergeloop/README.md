# 5.9. Mergeloop

## 5.9.1. Description
This module servers to track processing, interact with the inform processing queues and subsequently merge\ generate quality control images for each slide. The module loops through each project in the *AstroPathCohortsProgresss.csv* file, tracking or updating all slides then contuing to the next project. Importantly, the code uses the *MergeConfig_NN.xlsx* files (***defined here***) to intialize a project level inform queue as well as antibody specific processing folders in the tmp_inform_data folder (***defined in organization here***). The code will not process a cohort without these files as its main function is serve as an interaction between the user and the inform processing tasks. Additional details on what the code tracks, location of different files and processing steps can be found in this documentation. A typical user instruction is included *here* for people who will not be monitoring the code but will be using the pipeline to process slides.

## 5.9.2. Important Definitions

- ```<upkeep_and_progress>```: This folder contains all the necessary project related documents for the inform or practical user of the pipeline. 
   - add all additional files outside of those maintained by the code to this folder. Adding files or new folders to the main ```<Dname>``` directory may crash the processing or cause unexpected results
- *inform_queue.csv*: This file is the project level inform queue. 
   - The code transfers new tasks to the main inform queue and completed task infomation back to this queue.
   - adding tasks to this file is described in further documentation 
- *Main_inform_queue.csv*: The main queue is held in the ```<Mpath>``` location. Additional copies of this queue are created in the ```<upkeep_and_progress>``` folder of each project for reference
   - opening and modifying the main queue in the ```<Mpath>``` directory can cause the file to become corrupt resulting in loss of archived processes.
   - it is recommended that only the project level queues are modified.
   - it is also recommended that backup copies of the main queue are manually maintained
- ```<tmp_inform_data>```: this is subfolder under the ```<Dpath>\<Dname>```, it contains processing subfolders for each antibody in addition to the ```<Project_Development>``` folder 
- ```<Project_Development>```: this is a sub folder of ```<tmp_inform_data>```. Algorithms to be processed by the inForm processing code should be placed here. The code will not find algorithms placed elsewhere or in subfolders.

*Note*: Additional directory information can be found *here*

## 5.9.3. Instructions
The code should be launched through matlab. To start download the repository to a working location. Next, open a new session of matlab and add the ```AstroPathPipline``` to the matlab path. Then use the following to launch:   
``` mergeloop(<Mpath>) ```
- ```<Mpath>[string]```: the full path to the directory containing the ***AstropathCohortsProgress.csv*** file
   - description of this file can be found [here](../../scans#441-astropath_processing-directory "Title")

*Note:* For the code to process successfully be sure to create the merge configuration files *link*.

