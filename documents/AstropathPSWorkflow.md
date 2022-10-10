# 8.1. AstroPath Powershell Workflow

## 8.1.1. PS Workflow
1.	On startup
	1.	Imports all AstroPath tables and stores in respective ‘data’ objects
		1.	full_project_data, slide_data, worker_data, etc. 
	1.	Gets the module names from the dependency data
	1.	Creates project_data which is a hashtable 
		1.	Where project_data.(module) references a list of active projects for the module. 
	1.	Reads existing or creates empty module – project logs
	1.	Defines the module queues by reading them in or creating them, storing them in a moduleobj.(module) hashtable
	1.	Defines rerun and update status states
1.	Build sample db
	1.	Defines or updates module status for each sample in the set
	1.	If the sample is in the project_data.(module) project list it is queued for processing
1.	Defines the module file and log file watchers
	1.	If either was updated from a previous run then the slides with update rows will be rechecked and corresponding module tables and queues are updated
1.	Launch the tasks from the queue
1.	Wait for a task to finish or an event to trigger. Take the corresponding action of updating the memory variables for file changes or launch a new task 

## 8.1.2. AstroPath Task Distribution
1.	Build the sample db
	1.	builds the module task queues with the tasks to launch
1.	Define the worker list
	1.	Gets the current workers table
	1.	Searches for orphaned jobs as indicated by the workertasklog being present (is deleted on normal completion). 
		1.	Attempt to delete the file and check for errors in the provided task
		1.	If file exists and cannot be deleted the psexec task is still running. 
			1.	Create a job to watch the file for changes and delete it when changes occur. 
			1.	Job will then return and the worker can be handled as IDLE
1.	Distribute tasks 
	1.	For each module
		1.	Check for IDLE workers that are set to ON. 
		1.	Check for tasks in the module queue
		1.	Launch tasks until either (i) or (ii) are empty 
1.	Wait for tasks to finish 
	1.	Checks for new events or tasks that finish
	1.	For events
		1.	Launches events in the apeventhandler in sampledb
	1.	For tasks
		1.	Check the psexec log for changes
		1.	Check the worker log for changes
		
Defines three tasks files: 
 - Workertasklog: [jobname]’-taskfile-job.log’
    - The log file from psexec 
 - Workertaskfile: [jobname]’-taskfile.ps1’
    - The file with the actual powershell task to run
 - Workerlogfile: [jobname]’.log’
    - The log file for the launched task from psexec

## 8.1.3. AstroPath File Change Monitors
- Update dependency table
    - Changes could result in:
       - New module
       - Change in dependency
    - Actions:
       - For new modules, throw an error:
          - Must restart
       - For change in dependency:
          - Update dependencies
          - Update all samples
          - Update all module queues<br>
 - Update cohorts file, paths file, config file 
    - Changes could result in:
       - New project
       - Change in project location 
       - Update to version for a module
       - Update to module status for a project
    - Actions:
       - Update cohorts_data for new projects
          - Only includes projects in config, cohorts, and paths file
          - Checks that the dpath \ dname exists
       - Update $this.allprojects, $this.project_data (module projects turned on)
       - For project with changed row rerun sample db and module db. 
       - For new project
          - Add the file watchers 
          - Slides will be added on running them
- Update to slide file
    - Changes could result in:
       - updated slide data
       - new slides
    - Actions:
       - Update the slide_data in the workflow
 - Update to ffmodels, corrmodels, micomp
    - Changes could result in:
       - Change in dependencies for slides
    - Actions:
       - Get the new slide rows and update slide statuses
 - Update to worker_file:
    - Changes could result in:
       - Updated workers or worker status
    - Actions 
       - should update the worker status in the code base
       - should try to relaunch tasks
 - Update to main inform queue
    - Changes could result in:
    - Actions 
 - Update to local inform queue
    - Changes could result in:
    - Actions
 - Update to log files
    - Changes could result in:
    - Actions
 - Update to mergeconfig files
    - Changes could result in:
    - Actions
 - Update to image qa files
    - Changes could result in:
    - Actions

## 8.1.4. Adding Modules
Adding modules to the powershell astropath workflow is easy. There are three steps:

1.	Create the module class
	1.	Write the module as a powershell class and save the module in astropath workflow repo. Add the class to the ‘astropath.psm1’ manifest file.
	1.	To allow for compatibility and testing:
		1.	The class name should be the name of the module
		1.	The main method to run the module should be named ‘run[modulename]’
		1.	Ex. For the segmaps module
			1.	there is a segmaps class in the segmaps.ps1 file.
			1.	The main running methods is named runsegmaps.
1.	Add dependencies to the samplereqs.ps1 file
	1.	Add a method named test[modulename]files() to check for the output files for this module. 
		1.	Ex. For segmaps module testsegmapsfiles(). 
	1.	The method should return true if the output exists and false otherwise
1.	Add the module to the AstroPathDependencies.csv file with the module output that should exist before the module runs

|Module    |Dependency     |
|----------|---------------|
|shredxml  |transfer       |
|meanimage |shredxml       |
|mersmerseg|imagecorrection|
|vminform  |imagecorrection|
|merge     |vminform       |
|segmaps   |imageqa        |

Optional:
1.	Add version dependent checks for the module
	1.	Some modules need to do or check for different things depending on the version that runs. If this is the case add the special case to the ‘versiondependentchecks’ method at the bottom of the dependencies.ps1. 
1.	Add the module to the AstroPathConfig.ps1 file
	1.	This file allows you to turn the module on or off for different projects. 
	1.	This file also allows you to run a specified version for a project. The code typically checks for the 3-part version number rather than the full version number.
	1.	Both columns [modulename] and [modulename]version must be added to this file when adding the module

## 8.1.5. AstroPath Requirements
1.	Check for required software in the code and install them or throw an error if necessary. Some code does part of this already in shared tools
1.	Check if miniconda is installed on the system
	1.	If not prompt the user to download and install it
	1.	Save the location of the install
	1.	Launch the anaconda powershell window 
		1.	Execute module
3.	Can start a conda environment by
	1.	Entering: “conda init powershell” to a conda powershell window and restarting powershell
	1.	Need to check the installation of packages

## 8.1.6. AstroPath Dependencies
 - Scan
    - Actions:
       - Scan Overview (QPTIFF)
       - Scan im3 files
       - Add slides to APIDdef file and get ScanID to APID translation
    - Output:
       - Check for the *dpath\slideid* if it does exist the deps passed
       - If the *spath\slideid* exists
          - Xml file contains ‘Acquired’ items
          - im3 files in the directory match the number of ‘Acquired’ fields
             - all im3 files meet the minimum requirement sizes
 - ScanValidation
    - Dependencies:
       - If the *spath\slideid* exists
          - Xml file contains ‘Acquired’ items
          - im3 files in the directory match the number of ‘Acquired’ fields
             - all im3 files meet the minimum requirement sizes
       - If *spath\slideid* does not exist
             - Check for the dpath\slideid
    - Actions:
       - Run a small utility to check the accurate positioning and focus of im3 files
    - Output:
       - Batchid.txt file with BatchID in it
 - Transfer
    - Dependencies:
       - Scan module
    - Actions:
       - Transfer a slide from spath to dpath
       - Compress slide from spath to cpath
       - Remove spath
    - Output: 
       - Loglines
       - A new dpath slide directory with:
          - Checksums.txt file
          - Annotation.xml
          - Qptiff
          - BatchID.txt
          - Im3s exist and all im3s meet the minimum size requirements
 - Shredxml
    - Dependencies:
       - transfer
    - Actions:
       - Extracts exposure time xml information from all im3 files in an im3 directory to the [xml] directory
       - Extracts the Full.xml, Parameters.xml
    - Output: 
       - Xml directory with:
          - Full.xml
          - Parameters.xml
          - .xml for each im3
 - Meanimage
    - Dependencies:
       - shredxml
    - Actions:
       - Gets the mean image and mask stack of an im3 directory
       - Extracts the tissue masks for each im3 file
       - All output in slideid/im3/meanimage directory
    - Output:
       -  A new meanimage folder containing:
          -  <Slide_ID>-sum_images_squared.bin
          -  <Slide_ID>-std_err_of_mean_image.bin
          -  <Slide_ID>-mean_image.bin
          -  Optional:
             - <Slide_ID>-mask_stack.bin
             - Image_masking directory with [imageid]_tissue_mask.bin and _full_mask.bin for available images
             - <Slide_ID>-background_thresholds.csv 
 - batchmicomp
    - Dependencies:
       - meanimage
    - Actions:
       - Finds the set of samples whose mean images should be used to determine a single flatfield correction model
    - Output: 
       - A meanimagecomparison folder containing:
          - meanimagecomparison_table.csv
          - meanimage_comparison_average_over_all_layers.png
 - batchflatfield
    - Dependencies:
       - batchmicomp
    - Actions:
       - Build flatfields and add to the *[mpath]\flatfield* directory
       - Update AstroPathFlatfieldModules.csv with slides used to build a flatfield
       - Update AstroPathCorrectionModels.csv for slides to run with a specified flatfield  
    - Output:
       - SlideID in the AstroPathCorrectionModels.csv
       - Flatfield file exists in the flatfield directory
       - Labeled flatfield_[\*].bin ([\*]: from ‘FlatfieldVersion’ column in models file)
 - warpoctets
    - Dependencies:
       - batchflatfield
    - Actions:
       - Measures the warping affecting raw data files in a cohort
       - Only finds the octets for each sample and writes them out
    - Output: 
       - Check if the slide log base for warpoctets exists, if so:
          - check if the <Slide_ID>-mask_stack.bin file exists in the meanimage folder
       - Creates a <Batch_ID>-all_overlap_octets.csv
 - batchwarpkeys
    - Dependencies:
       - warpoctets
    - Actions:
       - Creates *[basepath]\warping\Batch_XX\octets* folder
       - Adds batch slides <Slide_ID>-all_overlap_octets.csv, initial, principle, final _pattern_octets_selected.csv files
       - Create image_keys_needed.txt
    - Output: 
       - A batch warp keys folder containing:
          - final_pattern_octets_selected.csv
          - initial_pattern_octets_selected.csv
          - image_keys_needed.txt
          - principal_point_octets_selected.csv
 - batchwarpfits
    - Dependencies:
       - batchwarpkeys
    - Actions:
       - Does initial, principle, and final pattern fits for images
       - Builds the final warp file ‘weighted_average_warp.csv’
       - Copies final warp file to *[basepath]\warping*  and renames to ‘warping_BatchID_XX.csv’
    - Output: 
       - warping_BatchID_XX.csv
 - imagecorrection
    - Dependencies:
       - batchwarpfits
    - Actions:
       - Creates a directory of flat field and warping corrected .im3 images files for each slide
    - Output: 
       - Tests that there are the correct number of .fw, .fw01, and .flatwim3 for given batch
 - Vminform
    - Dependencies:
       - imagecorrection
    - Actions:
       - Runs the provided algorithm through inForm
       - Add cell seg data, cell seg summary data for all images in the slide
       - Adds binary seg and component seg data for slides to appropriate antibody algorithms
    - Output: 
       - Cell seg data for each image in the slide
 - merge
    - Dependencies:
       - vminform
    - Actions:
       - Merge’s the inForm cell seg data
       - Creates the cleaned_phenotype_tables.csv
       - Creates the image qa image views
    - Output: 
       - A cleaned phenotype table for all im3 images
       - Image QA views 
 - imageqa
    - Dependencies:
       - merge
    - Actions:
       - Look over the image QA views for the slide
       - Edit the algorithm as needed and rerun through inForm via adding new row to inForm queue
       - Update Project level ImageQA.csv for each AB
    - Output: 
       - ‘X’ in each antibody column of the ImageQA.csv for a slide
 - segmaps
    - Dependencies:
       - imageqa
    - Actions:
       - Create the segmentation maps for each image from the inForm binary seg maps, component tiffs, and final cleaned_phenotype_table.csv files
    - Output: 
       - [imageid]_component_data_w_seg.tif for each image of the slide 
 - dbload
    - Dependencies:
       - segmaps
    - Actions:
       - Database load
    - Output: 
       - Publications
