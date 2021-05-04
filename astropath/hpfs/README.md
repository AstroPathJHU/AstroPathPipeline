# 5. HPF Processing (HPFs)
## 5.1. Description
This section of the documentation and code picks up after the slides have been added to the *Specimen_Table_N.xlsx*, a description of that file is located [here](../scans/#431-specimen_table "Title"). All code assumes slides and file structure format in [4] have been adhered to, additional definitions can also found there. 

In this section the ```<Dpath>\<Dname>``` processing folders are intialized, a slide list is created, and slides are assigned new ```SlideID```s [5.3]. Next images are renamed, transferred and backed up (**[5.4]**). Images are then corrected for imaging warping and flatfield effects (**[5.5]**). After this, cells are segmented and classified in each image using the multipass phenotype method in inForm. Each of these steps are run by custom code that process on perpetual loops with limited user interaction. An important step of this process is the quality control images generated on the cell classification output, described **[here]**. After, images from all slides in a cohort pass the cell classification quality controll, cell segmenation maps are rebuilt to remove cells not defined by the final merged cell segmentation[**described here**]. Finally, slide macro regional annotations are applied to the images using the HALO software and distributed to the slides using a batch script[**]. 

A final checklist of things to verify and complete before loading into the database is added **here**. In this section there is also a ```cleanup``` module which checks for any missing files as well as additional upkeep instructions.

## 5.2 Instructions
### 5.2.1. Typical User Instructions
Usually the code is launched and monitored by one person. Most of the user interaction for those not running the code will be with the ```<upkeep_and_progress>``` subfolder in the ```<Dpath>\<Dname>``` directory, the ```<QA_QC>``` subfolders in the specimen directories, and the ```<Project_Development>``` folder (additional documentation on directory organization here). Instructions for running the code itself is below in *1*. Typical user interaction with the data occurs after the image corrections have been applied to the images and is where we pick up in this section. 

1. Once slides have image corrections applied, the user should create inform phenotype projects in the multipass format. 
2. Algorithms should have export settings applied according to the documentation then be saved into the ```<Project_Development>``` subfolder. 
   - If algorithms are placed anywhere else, including subfolders under the ```<Project_Development>``` folder, they will not be found for processing during the next steps (```<inform_processing>``` module).
3. Check that the *MergeConfig_NN.xlsx* files have been created for the slide's batch and added to the appropriate location. 
4. Add slides to the *inform_queue.csv* according to the protocol. 
   - When the code has verified that the slides are ready for inForm processing, it will preallocate initial rows for the slide. One row will be preallocated for each slide-antibody pair, with the antibodies defined in the *MergeConfig_NN.xlsx* file. 
     - If the slide-antibody pair does not pre-allocate or too many slide-antibody pairs pre-allocate, either check that the image corrections for the slide has completed or check the formatting of the *MergeConfig_NN.xlsx*.
5. Wait for the inform machines to process the queue and the code to pull data to the ```<SlideID>\<inform_data>``` folder (this processing is outlined here).
   - Processing information is updated in the *samples_summary.xlsx* spreadsheet, including the algorithms, date of processing, and number of files for each antibody
   - Once all antibodies for a slide have been completed the code will merge the data and create the ```<QA_QC>``` folder of a ```<SlideID>``` directory. [details on the merge found here].
   - For the quality control images to be generated the following must be present:
     - Data for each image indicated in the **export documentation** of each antibody must be present in the ```<SlideID>\<inform_data>\<Phenotyped>``` . 
	 - The component data must also be present in the ```<SlideID>\<inform_data>\<Component_Tiffs>``` directory.
6. Once QC images have been created, qc the slides according to the protocol.
7. If an antibody for a slide fails qc, re-work the phenotype algorithm and resubmit algorithms according to the protocol. 
   - Continue this process until all slide-antibody pairs pass quality control in a cohort.   

Simulanteously with cell classification a pathologist should use HALO to annotate regions of interest in the QPTiff images then export them according to the *documentation outlied.* 
 
Once processing reaches this point the user should direct processing to the person maintaining the code. The modules ```<segmaps>```, ```<transferanno>```, and ```<cleanup>``` should each be launched. The ```<cleanup>``` module should be launched last. Afterward steps in the *clean up documentation* should be taken to extract missing files, export the control tma component tiffs and convert the batch\ merge tables to the corresponding acceptable csv format. 
 
After the cleanup protocols the ```<Project>``` is ready to proceed to the the ```Samples``` processing stage.

### 5.2.2. Launching Code Instructions
For the hpfs workflow all code is launched either by a set of batch files or by matlab commands. All of the launching batch files are located in the launch folder of ```astropath\hpfs```. To get started, download the repository to a working destination. The following modules should be launched through the command line and require simple path input: ```AstroIDGen```, ```TransferDeamon```, ```transferanno```, and ```inform_queue```. The ```inform_worker``` can be launched with the shortcut file after it has been successfully installed according to the module **documentation**. The rest of the modules can be launched in matlab by adding the ```AstroPathPipeline``` repository to the matlab path and entering the module name. Most of the matlab modules require minimal input such as the location of the ```<Mpath>``` or a working folder\ drive, all of which is desribed in the respective module documentation. 

Each module, outside the ```transferanno``` and ```segmaps``` modules, are designed to indefinitely loop through all ```Project```s defined in the ```<Mpath>```. Files are only updated or processed in the event of new\updated files or queue input. A formal queued workflow is still being developed. The current workflow design requires the use of workers on multiple machines for best computation potential which increases the complications in designing a formal workflow. Modules can be turned off at convience by closing the corresponding command window. However, it should be noted that if there is nothing for the module to process computational impact will be minimal. Usually, modules process all available tasks then hibernate for a set peroid of time before checking for new tasks. The workflow steps are laid out below, this should not be used in lieu of the formal documentation in each section, linked to under contents.

Also note that the modules are expected to be running simultaneously such that the user runs each module from a separate command line or matlab instance. The modules designated as *workers* can be launched multiple times to reduce processing time though processing locations must be predefined usually hard coded in the respective *queue* module. 

## 5.3 Workflow Overview
As noted above, modules outside of ```segmaps``` and ```transferanno``` can be launched and allowed to run continuously. These steps outline how a slide might process through the workflow. 

1. Update the AstroPath files in the <mpath>
2. Update the slides in the *Specimen_Table_N.xlsx*
3. Scan the slides according to the documentation in [4.](../scans/#scans "Title")
4. Launch the ```AstroIDGen``` module to intialize the slides into the pipeline
5. Launch the ```TransferDeamon``` module to start transferring completed slides from the <Spath> location to the ```<Dpath>\<Dname>```
6. Launch the ```meanimages``` module to create mean images of each slide as it finishes transferring; these will be used to build the batch flatfields.
7. Launch the ```flatw_queue``` module to create the flatfields and assign new slides to the flatw_queue. Then the module distributes flatw jobs to the assigned workers.
8. Launch the ```flatw_worker``` module on the assigned worker machine to process the flatfielding and image warping corrections on a particular slide's hpf image set
9. Create the BatchID and MergeConfig files for the project according to documentation in scans
10. Launch the ```mergeloop``` module to initialize necessary antibody processing folders, create the local inform_queue for a project, recieve inform results, merge the data and create qa qc images to evaluate the inform classification algorithms.
11. Create phenotype algorithms in inForm according to the protocol established **here**
12. Launch the ```inform_queue``` module to send jobs from the main inform queue to the queues on the inform worker machines.
13. Launch the ```inform_worker``` module to process algorithms in inform
14. wait for QC images to process by the ```mergeloop``` module
    - evaluate the qc by the protocols established **here**
    -repeat 11-13 as needed
15. Launch ```segmaps``` module after qc has been completed for the cell classification of slides in a project to build the final segmenation maps.
16. Launch ```transferanno``` module after the slides have been successfully annotated in HALO and annotations have been exported to a desired location.
17. complete the final checklist located **here**

- ```AstroIDGen``` should be launched on the <Spath> location.  
- The following modules can be launched with **hpfs_main** at the same time: ```TransferDeamon```, ```meanimages```, ```flatw_queue```, ```inform_queue```, ```mergeloop```. 
- The following modules must be launched on their respective "worker" locations: ```flatw_worker``` and ```inform_worker```. 
- Launching segmaps runs over the entire set of projects but is not a continous loop and must be restarted to reprocess
- transferanno is a simple script that distributes and renames the halo annotations from a single folder to corresponding slide folders in a single project folder

## 5.4. Contents
- [5.1. Description](#51-description "Title")
- [5.2. Instructions](#52-instructions "Title")
- [5.3. Workflow Overview](#53-workflow-overview "Title")
- [5.4. Contents](#54-contents "Title")
- [5.5. AstroIDGen](AstroidGen#55-astroid-generation "Title")
  - [5.5.1. Description](AstroidGen#551-description "Title")
  - [5.5.2. Important Definitions](AstroidGen#552-important-definitions "Title")
  - [5.5.3. Workflow](AstroidGen#553-workflow "Title")
- [5.6. Transfer Daemon](TransferDaemon#56-transfer-daemon "Title")
  - [5.6.1. Description](TransferDaemon#561-description "Title")
  - [5.6.2. Important Definitions](TransferDaemon#562-important-definitions "Title")
  - [5.6.3. Instructions](TransferDaemon#563-instructions "Title")
  - [5.6.4. Workflow](TransferDaemon#564-workflow "Title")
    - [5.6.4.1. Initial Transfer](TransferDaemon#5641-initial-transfer "Title")
    - [5.6.4.2. MD5 Check](TransferDaemon#5642-md5-check "Title")
    - [5.6.4.3. Compression Into Backup](TransferDaemon#5643-compression-into-backup "Title")
    - [5.6.4.4. Source File Handling](TransferDaemon#5644-source-file-handling "Title")
  - [5.6.5. Notes](TransferDaemon#565-notes "Title") 
- [5.7. Meanimages](meanimages#57-meanimages "Title")
  - [5.7.1. Description](meanimages#571-description "Title")
  - [5.7.2. Important Definitions](meanimages#572-important-definitions "Title")
  - [5.7.3. Instructions](meanimages#573-instructions "Title")
  - [5.7.4. Workflow](meanimages#574-workflow "Title")
    - [5.7.4.1. Checking for Tasks](meanimages#5741-checking-for-tasks "Title")
	- [5.7.4.2. Shred Im3s](meanimages#5742-shred-im3s "Title")
	- [5.7.4.3. raw2mean](meanimages#5743-raw2mean "Title")
- [5.8. Flatfield](Flatfield#58-flatfield "Title")
  - [5.8.1. Description](Flatfield#581-description "Title")
  - [5.8.2. Important Definitions](Flatfield#582-important-definitions "Title")
    - [5.8.2.1. Flatw Expected Directory Structure](Flatfield#5821-flatw-expected-directory-structure "Title")
	- [5.8.2.2. Output Formatting](Flatfield#5822-output-formatting "Title")
  - [5.8.3. Instructions](Flatfield#583-instructions "Title")
    - [5.8.3.1. flatw_queue](Flatfield#5831-flatw_queue "Title")
	- [5.8.3.2. flatw_worker](Flatfield#5832-flatw_worker "Title")
	- [5.8.3.3. Im3tools](Flatfield#5833-im3tools "Title")
  - [5.8.4. Overview Workflow of Im3Tools](Flatfield#584-overview-workflow-of-im3tools "Title")
- [5.9. Mergeloop](mergeloop#59-mergeloop "Title")
- [5.10. Inform_processing](inform_processing#510-inform_processing "Title")
- [5.11. Segmaps](segmaps#511-segmaps "Title")
- [5.12. Transferanno](transferanno#512-transferanno "Title")