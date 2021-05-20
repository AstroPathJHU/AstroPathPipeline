# 5.3 Instructions
## 5.3.1. Typical User Instructions
Usually the code is launched and monitored by one person. Most of the user interaction for those not running the code will be with the ```<upkeep_and_progress>``` subfolder in the ```<Dpath>\<Dname>``` directory, the ```<QA_QC>``` subfolders in the specimen directories, and the ```<Project_Development>``` folder [4.5.4.](). Instructions for running the code itself is below in *1*. Typical user interaction with the data occurs after the image corrections have been applied to the images and is where we pick up in this section. 

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

## 5.3.2. Launching Code Instructions
For the hpfs workflow all code is launched either by a set of batch files or by matlab commands. All of the launching batch files are located in the launch folder of ```astropath\hpfs```. To get started, download the repository to a working destination. The following modules should be launched through the command line and require simple path input: ```AstroIDGen```, ```TransferDeamon```, ```transferanno```, and ```inform_queue```. The ```inform_worker``` can be launched with the shortcut file after it has been successfully installed according to the module **documentation**. The rest of the modules can be launched in matlab by adding the ```AstroPathPipeline``` repository to the matlab path and entering the module name. Most of the matlab modules require minimal input such as the location of the ```<Mpath>``` or a working folder\ drive, all of which is desribed in the respective module documentation. 

Each module, outside the ```transferanno``` and ```segmaps``` modules, are designed to indefinitely loop through all ```Project```s defined in the ```<Mpath>```. Files are only updated or processed in the event of new\updated files or queue input. A formal queued workflow is still being developed. The current workflow design requires the use of workers on multiple machines for best computation potential which increases the complications in designing a formal workflow. Modules can be turned off at convience by closing the corresponding command window. However, it should be noted that if there is nothing for the module to process computational impact will be minimal. Usually, modules process all available tasks then hibernate for a set peroid of time before checking for new tasks. The workflow steps are laid out below, this should not be used in lieu of the formal documentation in each section, linked to under contents.

Also note that the modules are expected to be running simultaneously such that the user runs each module from a separate command line or matlab instance. The modules designated as *workers* can be launched multiple times to reduce processing time though processing locations must be predefined usually hard coded in the respective *queue* module. 
