# 5.10.7. Proccessing inForm Tasks 
## 5.10.7.1. Description
Afer algorithms are created slides are processed through inForm®. This process is onerous for a few reasons. First, for whole slide scans of 100-3000 images, inForm Cell Analysis® could take on the order of 6-7 hours. This on its own is a significant amount of time. However, since we use separate algorithms for each antibody, processing a 7-color panel on one large specimen could take over 24 hours. Second, we change the path locations and path names of the slides as part of the ```flatfield``` processing. Because of these changes, typical methods for batch processing images at the slide or directory levels no longer function as expected in inForm Cell Analysis®. Instead to process all images from a slide, we must navigate to the ```flatw_path``` directories and manually add all images. In order to automate this processing, JHU installed a set of inForm licenses onto virtual machines located on a server. Code was then written in the programming language *AutoIt* to simulate mouse clicks and run algorithms in batch mode for slides in the *AstroPath Pipeline* directory structure. Afterward a queued system was developed to process algorithms at scale using this utility, documentation on this queued based system is found here.

The code is designed to run continously. There are two module scripts. The first is a ```inform_queue``` script which opens an *inForm_queue.csv* spreadsheet and disseminates tasks to spreadsheets kept on worker virtual machines. The second script (```inform_worker```) runs on each of the worker locations, checking this worker spreadsheet for new batch tasks and launching those new batch algorithm tasks through inForm. 

Note that the worker module that runs inForm requires that the image data is reformatted into the *AstroPath* directory structure and that the algorithms are located in a set location adjacent to the images. Both locations are described below in [*Important Definitions*](#51072-important-definitions) and in more detail [here](../../../scans/docs/DirectoryOrganization.md#46-directory-organization "Title"). Algorithm or project export settings should be set up before saving the file according to the protocol defined [here](SavingProjectsfortheinFormJHUProcessingFarm.md#5104-saving-projects-for-the-inform-jhu-processing-farm). Since the code simulates mouse clicks and changes in the active window, it is best not to use the computer when running the worker module.

*NOTE*: For use in the AstroPath pipeline or with the MaSS utility, export settings should be appied according to the documentation located [here](SavingProjectsfortheinFormJHUProcessingFarm.md#5104-saving-projects-for-the-inform-jhu-processing-farm). Once batch processing is finished for an antibody, data files can be transferred to the proper MaSS-slide organization defined in the MaSS documentation [here](../../mergeloop/MaSS#merge-a-single-sample-mass) or in more depth with relation to the rest of the pipeline [here](../../../scans/docs/DirectoryOrganization.md#46-directory-organization). Data can either be transferred manually or with the ```mergeloop``` module located [here](../../mergeloop#59-mergeloop). 

## 5.10.7.2. Important Definitions
- The code requires that an active version of inForm is installed on the computer and that the version installed matches the version number coded in at the bottom of **BatchProcessing.ps1**.
- The main *inForm_queue.csv* ([described here](AddingSlidestotheinFormQueue.md#51053-instructions))
- The virtual machine queue file: *VM_inForm_queue.csv*
  - This file should be located in the ```*\hpfs\inform_processing\BatchProcssing``` folder. 
  - This file consists of a five columns ```Path,Specimen,Antibody,Algorithm,Start,Finish```
    - ```Path```: The path up to but not including the specimen folder or ```\\<Dname>\<Dpath>``` ([desribed here](../../../scans/docs/Definitions.md/#432-path-definitions))
      - E.g. *\\bki04\Clinical_Specimen_2* 
    - ```Specimen```: The name of the specimen to be analyzed
      - The image data to be processed should be located in an ```<Path>\<Specimen>\im3\flatw``` folder 
      - E.g. *M1_1*
    - ```Antibody```: The antibody name that will be processed.
      - All data will be exported into a ```<Path>\tmp_inform_data\<Antibody>``` subfolder
      - E.g. *CD8*
    - ```Algorithm```: The name for the algorithm to do the processing **include the file extension**
      - The algorithm should be location in a ```<Path>\tmp_inform_data\Project_Development``` folder for the code to be able to find it
      - Only the *Project_Development* folder will be search, the search is not recursive. Subfolders will not be searched for the algorithm
      - E.g. *CD8.ifp*
    - ```Start```: The time that the processing started, this is updated by the code and should be left blank
    - ```Finish```: The time that the processing finished, this is updated by the code and should be left blank
  - The column headers should always be the first line in the csv file.
  - The file can be updated either manually or by the ```inform_queue``` code. 
    - To manually update, add the following variables ```<Path>,<Specimen>,<CD8>``` to the next line in the csv file
    - E.g. \\bki04\Clinical_Specimen_2,M1_1,CD8,CD8.ifp,
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
## 5.10.7.3. Instructions
## 5.10.7.3.1. Create VMs for the Processing
We have set up a server with multiple virtual machines each running it's own instance of InForm Cell Analysis. This set up is quite complex and Akoya Biosciences does not recommend installing the software on virtual machines. The *AstroPath* group has been given explicit permission to set up a processing system like this through a work agreement with Akoya Biosciences. The system set up only allows server administrators access to the virtual machines and virtual machines are not typically used as local workstations. Linking multiple computers on the same network with static licenses can be set-up to simulate a similar processing capacity, though this would be an even more complex system (particularly since using remote desktop with InForm will corrupt the license). Consult with Akoya Biosciences and your local IT professionals before attempting to set up a similar system.

The *AstroPath* group uses Hyper-V with Windows to set up virtual machines. We use Hyper-V becuase of it's support in PowerShell where much of the code for processing was piloted and is maintained. An introduction to Hyper-V can be found [here](https://docs.microsoft.com/en-us/virtualization/hyper-v-on-windows/about/). Once virtual machines are created, a windows OS should be installed and network capabilities should be added. InForm can then be installed and activated as on a normal computer. Be sure that you only log into the virtual machine using the Hyper-V Manager or through a screen mirroring server like TightVNC or RealVNC (using the Hyper-V Manager method will yield the best results). **Never use a windows remote desktop session, this will corrupt the InForm license.** Logging in using these screen mirroring techniques allows us to use the virtual machines as normal computers and still only allows one InForm session to be active at a time. 

Once virtual machines are set up install and launch the processing code according to the documentation found in [5.10.7.3.3.](#570733-running-the-inform-worker-module).

## 5.10.7.3.2. Running the ```inform queue``` Module
1. Download the repository to a processing computer or location where inForm is installed
2. Edit the username and password in line 78. 
   - JHU has created a local user account for each virtual machine. The workers run under that local account for processing. This script is designed to use the specified username and password dedicated to each virtual machine for processing on that machine.
3. Navigate to *\*astropath\\hpfs\\launch* and double click on the ```inform_queue-Shortcut``` to launch processing

## 5.10.7.3.3. Running the ```inform worker``` Module
1. Download the repository to a processing computer or location where inForm is installed *under the "Program Files* folder*
2. Edit the username and password in ```RunFullBatch.au3```
   - Download and install [AutoIt](https://www.autoitscript.com/site/)
   - open the ```RunFullBatch.au3``` found in the *BatchProcessing* folder
   - Edit line 124
   - Go to 'tools' at the top of the page and compile (make sure the .exe is saved under the *BatchProcessing* folder)
3. Make sure the the username used in [5.10.7.3.2](#570732-running-the-inform_queue-module) has full access to the *AstroPathPipeline* folder
4. Copy the 'inform_worker-Shortcut' from *\*astropath\\hpfs\\launch* to the desktop of the virtual machine
5. Double click on the shortcut to launch processing
6. Add a jobs to the queue as described [above](#51072-important-definitions)

*NOTE*: Because some of the workflow uses automatic scripts windows defender flags these files as Trojans. Usually we just turn off windows defender since our virtual machines cannot be accessed outside of the JHU network. 

## 5.10.7.4. Workflow
The first step in processing is to test for the input version of inForm that will be used to run the software. The inForm version can be found at the bottom of **BatchProcessing.ps1**. Next the code attempts to injest the VM_inForm_queue.txt, described above in [Important Definitions](#51072-important-definitions). This file should be located in the *\*\hpfs\\inform_processing\\BatchProcssing* folder. The code searches along the path location where the script was launched for the queue file. 

After all jobs are marked complete, the code waits and rechecks the queue after a set period of time for new jobs. Jobs can be added to the queue either manually or by the code. The queue file should be located in 'BatchProcessing' folder. The queue string is simple and consists of a five part comma separated list. To add a new job to the queue The code requires that the image data is reformatted into the AstroPath directory structure and that the algorithms are located in a set location adjacent to the images. Algorithm export settings should be set up before saving.
