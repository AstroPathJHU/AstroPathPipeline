# 5. HPFs
## 5.1. Description
This section of the documentation and code picks up after the slides have been added to the *Specimen_Table_N.xlsx*, a description of that file is located [here](../scans/#431-specimen_table "Title"). All code assumes slides and file structure format in [4] have been adhered to, additional definitions can also found there. 

In this section the ```<Dpath>\<Dname>``` processing folders are intialized, a slide list is created, and slides are assigned new ```SlideID```s [5.3]. Next images are renamed, transferred and backed up (**[5.4]**) then corrected for imaging warping and flatfield effects (**[5.5]**). After this, cells are segmented and classified in each image using the multipass phenotype method in inForm. Each of these steps are run by custom code that process on perpetual loops with limited user interaction. An important step of this process is the quality control images generated on the cell classification output, described **[here]**. After, the cell classification is quality controlled, cell segmenation maps are rebuilt to remove cells not defined by the final merged cell segmentation[**described here**]. Finally, slide macro regional annotations are applied to the images using the HALO software and distributed to the slides using a batch script[**]. 

A final checklist of things to verify for loading into the database is added **here**. Additional upkeep instructions are also added in this section.

## 5.2 Instructions
For the hpfs workflow all code is launched by a set of batch files or matlab commands. All of the launching batch files are located in the launch folder of ```astropath\hpfs```. To get started, download the repository to a working destination. The following modules should be launched through the command line as they require simple path input: ```AstroIDGen```, ```TransferDeamon```, ```transferanno```, and ```inform_queue```. The ```inform_worker``` can be launched with the shortcut file after it has been successfully installed according to the module **documentation**. The rest of the modules can be launched in matlab by adding the ```AstroPathPipeline``` repository to the matlab path and entering the module name. Most of the matlab modules require minimal input such as the location of the ```<Mpath>``` or a working folder\ drive, all of which is desribed in the respective module documentation. 

Each module, outside the ```transferanno``` and ```segmaps``` modules, are designed to indefinitely loop through all ```Project```s defined in the ```<Mpath>```. Files are only updated or processed in the event of new\updated files or queue input. A formal queued workflow is still being developed. The current workflow design requires the use of workers on multiple machines for best computation potential which increases the complications in designing a formal workflow. Modules can be turned off at convience by closing the corresponding command window. However, it should be noted that if there is nothing for the module to process computational impact will be minimal. Usually, modules process all available tasks then hibernate for a set peroid of time before checking for new tasks. The workflow steps are laid out below, this should not be used in lieu of the formal documentation in each section, linked to under contents.

Also note that the modules are expected to be running simultaneously such that the user runs each module from a separate command line or matlab instance. The modules designated as *workers* can be launched multiple times to reduce processing time though processing locations must be predefined usually hard coded in the respective *queue* module. 

## 5.3 Workflow Overview
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
- [5.5. AstroIDGen](AstroidGen\#55-astroid-generation "Title")
  - [5.5.1. Description](AstroidGen\#551-description "Title")
  - [5.5.2. Important Definitions](AstroidGen\#552-important-definitions "Title")
  - [5.5.3. Workflow](AstroidGen\#553-workflow "Title")
- [5.6. Transfer Daemon](TransferDaemon\#56-transfer-daemon "Title")
  - [5.6.1. Description](TransferDaemon\#561-description "Title")
  - [5.6.2. Important Definitions](TransferDaemon\#562-important-definitions "Title")
  - [5.6.3. Instructions](TransferDaemon\#563-instructions "Title")
  - [5.6.4. Workflow](TransferDaemon\#564-workflow "Title")
    - [5.6.4.1. Initial Transfer](TransferDaemon\#5641-initial-transfer "Title")
    - [5.6.4.2. MD5 Check](TransferDaemon\#5642-md5-check "Title")
    - [5.6.4.3. Compression Into Backup](TransferDaemon\#5643-compression-into-backup "Title")
    - [5.6.4.4. Source File Handling](TransferDaemon\#5644-source-file-handling "Title")
  - [5.6.5. Notes](TransferDaemon\#565-notes "Title") 
- [5.7. meanimages](meanimages\#57-meanimages "Title")
  - [5.7.1. Description](meanimages\#571-description "Title")
  - [5.7.2. Important Definitions](meanimages\#572-important-definitions "Title")
  - [5.7.3. Instructions](meanimages\#573-instructions "Title")
  - [5.7.4. Workflow](meanimages\#574-workflow "Title")
    - [5.7.4.1. Checking for Tasks](meanimeages\#5741-checking-for-tasks "Title")
	- [5.7.4.2. Shred Im3s](meanimages\#5742-shred-im3s "Title")
	- [5.7.4.3. raw2mean](meanimages\#5743-raw2mean "Title")
- [5.8. flatfield](Flatfield\#58-flatfield "Title")
  - [5.8.1. Description](Flatfield\#581-description "Title")
  - [5.8.2. Important Definitions](Flatfield\#582-important-definitions "Title")
    - [5.8.2.1. Flatw Expected Directory Structure](Flatfield\#5821-flatw-expected-directory-structure "Title")
	- [5.8.2.2. Output Formatting](Flatfield\#5822-output-formatting "Title")
  - [5.8.3. Instructions](Flatfield\#583-instructions "Title")
    - [5.8.3.1. flatw_queue](Flatfield\#5831-flatw_queue "Title")
	- [5.8.3.2. flatw_worker](Flatfield\#5832-flatw_worker "Title")
	- [5.8.3.3. Im3tools](Flatfield\#5833-im3tools "Title")
  - [5.8.4. Overview Workflow of Im3Tools](Flatfield\#584-overview-workflow-of-im3tools "Title")
- [5.9. mergeloop](mergeloop\#59-mergeloop "Title")
- [5.10. inform_processing](inform_processing\#510-inform_processing "Title")
- [5.11. segmaps](segmaps\#511-segmaps "Title")
- [5.12. transferanno](transferanno\#512-transferanno "Title")