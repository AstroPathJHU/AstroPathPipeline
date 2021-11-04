# 5.8.7. Proccessing inForm Tasks 
## 5.8.7.1. Description
Afer algorithms are created slides are processed through inForm®. This process is onerous for a few reasons. First, for whole slide scans of 100-3000 images, inForm Cell Analysis® could take on the order of 6-7 hours. This on its own is a significant amount of time. However, since we use separate algorithms for each antibody, processing a 7-color panel on one large specimen could take over 24 hours. Second, we change the path locations and path names of the slides as part of the ```flatfield``` processing. Because of these changes, typical methods for batch processing images at the slide or directory levels no longer function as expected in inForm Cell Analysis®. Instead to process all images from a slide, we must navigate to the ```flatw_path``` directories and manually add all images. In order to automate this processing, JHU installed a set of inForm licenses onto virtual machines located on a server. Code was then written in the programming language *AutoIt* to simulate mouse clicks and run algorithms in batch mode for slides in the *AstroPath Pipeline* directory structure. Afterward a queued system was developed to process algorithms at scale using this utility, documentation on this queued based system is found here.

The code is designed to run continously. The code starts by opening the *inForm_queue.csv* spreadsheet and disseminates tasks to worker virtual machines. Note that the code requires that the image data is reformatted into the *AstroPath* directory structure and that the algorithms are located in a set location adjacent to the images. Both are described below in [Important Definitions](#5872-important-definitions) and in more detail [here](../../../scans/docs/DirectoryOrganization.md#46-directory-organization "Title"). Project export settings should be set up before saving the file according to the protocol defined [here](SavingProjectsfortheinFormJHUProcessingFarm.md#584-saving-projects-for-the-inform-jhu-processing-farm). Because of a bug noticed in inForm v2.4.8, only projects should be added to the queue. Since the code simulates mouse clicks and changes in the active window, it is best not to use the computer when running the worker module. 

*NOTE*: For use in the AstroPath pipeline or with the MaSS utility, export settings should be appied according to the documentation located [here](SavingProjectsfortheinFormJHUProcessingFarm.md#584-saving-projects-for-the-inform-jhu-processing-farm). Once batch processing is finished for an antibody, data files are transferred to the proper MaSS-slide organization defined in the MaSS documentation [here](../../mergeloop/MaSS#merge-a-single-sample-mass) or in more depth with relation to the rest of the pipeline [here](../../../scans/docs/DirectoryOrganization.md#46-directory-organization). 

## 5.8.7.2. Important Definitions
- The code requires that an active version of inForm is installed on the computer and that the version installed matches the version number coded in at the bottom of **BatchProcessing.ps1**.
- The main *vminform-queue.csv* (the local inform queue is [described here](AddingSlidestotheinFormQueue.md#5853-instructions))
  - This file should be located in a ```<mpath>```\across_project_queues directory. 
  - This file consists of a five columns ```Path,Specimen,Antibody,Algorithm,Processing Location,Start```
    - ```Path```: The path up to but not including the specimen folder or ```\\<Dname>\<Dpath>``` ([described here](../../../scans/docs/Definitions.md/#432-path-definitions))
      - E.g. *\\\\bki04\Clinical_Specimen_2* 
    - ```Specimen```: The name of the specimen to be analyzed
      - The image data to be processed should be located in an *```<Path>```\\```<Specimen>```\\im3\\flatw* folder 
      - E.g. *M1_1*
    - ```Antibody```: The antibody name that will be processed.
      - All data will be exported into a *```<Path>```\\tmp_inform_data\\```<Antibody>```* subfolder
      - E.g. *CD8*
    - ```Algorithm```: The name for the project to do the processing **include the file extension**
      - The project should be location in a *```<Path>```\\tmp_inform_data\\Project_Development* folder for the code to be able to find it
      - Only the *Project_Development* folder will be search, the search is not recursive. Subfolders will not be searched for the algorithm
      - E.g. *CD8.ifp*
    - ```Processing Location```: Where the slide is being processed
    - ```Start```: The time that the processing started, this is updated by the code and should be left blank
  - The column headers should always be the first line in the csv file.
    - To manually update, add the following variables ```<Path>,<Specimen>,<CD8>``` to the next line in the csv file
    - E.g. \\\\bki04\Clinical_Specimen_2,M1_1,CD8,CD8.ifp,
- Directory Structure: The code requires that the files are set up in the following format additional information on formatting can be found [here](../../../scans/docs/DirectoryOrganization.md#46-directory-organization "Title"): <br>
```
  +--<Path>
  | +-- <Specimen>
  | | +-- *im3\flatw\*.im3*
  | +-- tmp_inform_data 
  | | +-- Project_Development 
  | | | +-- <Algorithm>(e.g. CD8.ifp) 
  | | +-- <Antibody> *(initialize at least 1 folder here)*
  | | | +-- <computer_name>
  | | | | +-- *Results*
``` 
## 5.8.7.3. Instructions
### 5.8.7.3.1. Setting up the Virtual Machines for inForm®
We have set-up a server with multiple virtual machines each running its own instance of InForm Cell Analysis. Typically the virtual machines are not used as local workstations but as processing nodes for batch algorithms. Linking multiple computers on the same network with static licenses can be set-up to simulate a similar processing capacity, though this would be an even more complex system (particularly since using remote desktop with InForm will corrupt the license). The *AstroPath* group has been given explicit permission to set up this system through a special agreement with Akoya Biosciences. 

The *AstroPath* group used Hyper-V with Windows to set-up the virtual machines. We use Hyper-V because of its support in PowerShell where much of the code for processing was piloted and is maintained. An introduction to Hyper-V can be found [here](https://docs.microsoft.com/en-us/virtualization/hyper-v-on-windows/about/). Once virtual machines are created, a Windows OS should be installed and network capabilities should be added. InForm can then be installed and activated as on a normal computer. Be sure that you only log into the virtual machine using the Hyper-V Manager or through a screen mirroring server like TightVNC or RealVNC (using the Hyper-V Manager method will yield the best results). **Never use a windows remote desktop session, this will corrupt the InForm license.** Logging in using these screen mirroring techniques allows us to use the virtual machines as normal computers and still only allows one InForm session to be active at a time. 

Once virtual machines are set up install and launch the processing code according to the documentation found [5.8.7.3.3.](#58732-running-the-vminform-module).

### 5.8.7.3.2. Running the ```vminform``` Module
This module can be launched acrross all projects using the DispatchTasks-vminform.bat file located in the [launch folder](../../../launch). 

You can also launch a specific project using the following:
```
Import-Module '*.\astropath'; DispatchTasks  -mpath:<mpath> -module:'vminform' -project:<project>
```
- replace '\*' with the location up to and including the *AstroPathPipeline* repository
- ```<mpath>```: the main path for all the astropath processing .csv configuration files; the current location of this path is *\\bki04\astropath_processing*
- ```<Project>```: Project Number

The workflow will ask for your credentials in a windows credential window, these are stored as an encrpyted network key used to send tasks to virtual machines accordingly. Tasks are launched on the worker locations defined in the *AstroPathHPFsWLoc.csv* file for each slide that has not yet been completed. Trigger events for this module are when new rows are added to the main *vminform-queue.csv*, [described above](#5872-important-definitions), with the algorithm column filled in but the task has not been sent to a processing location defined by the 'Processing Location' column. This file should be autogenerated by the ```mergeloop``` module and slides that finish flatfield will be intialized in this table for the first time with the first 4 columns filled in for each antibody defined in the *MergeConfig_Batch.xlsx* spreadsheet.
