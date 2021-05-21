# 5.10.7. Proccessing inForm Tasks 
## 5.10.7.1. Description
This module was designed to launch batch processing for a set of images in a directory through Akoya's InForm Cell Analysis software automatically. The code uses a queued based system for processing and is designed to run continously. There are two module scripts. The first is a ```queue``` script which opens an *inForm_queue.csv* spreadsheet and disseminates tasks to spreadsheets kept on worker computers. The second script runs on each of the worker locations, checking this worker spreadsheet for new batch tasks and launching those new batch algorithm tasks through inForm. 

The worker module that runs inForm requires that the image data is reformatted into the AstroPath directory structure and that the algorithms are located in a set location adjacent to the images. Both locations are described below in *Important Definitions* and in more detail [here](../../../scans#45-directory-organization "Title"). Algorithm or project export settings should be set up before saving the file according to the protocol defined [here](). Since the code simulates mouse clicks and changes in the active window, it is best not to use the computer when running the worker module.

*NOTE*: For use in the AstroPath pipeline or with the MaSS utility, export settings should be appied according to the documentation located **[here]**. Once batch processing is finished for an antibody, data files can be transferred to the proper MaSS-slide organization defined in the MaSS documentation **[here]** or in more depth with relation to the rest of the pipeline **[here]**. Data can either be transferred manually or with the progress track code located **[here]**. 

## 5.10.7.2. Important Definitions
- The code requires that an active version of inForm is installed on the computer and that the version installed matches the version number coded in at the bottom of **BatchProcessing.ps1**.
- The queue file: *VM_inForm_queue.csv*
  - This file should be located in the ```*\hpfs\RunInForm\BatchProcssing``` folder. 
  - This file consists of a five columns ```Path,Specimen,Antibody,Algorithm,Start,Finish```
    - ```Path```: The path up to but not including the specimen folder or ```\\<Dname>\<Dpath>``` (**desribed here**)
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
  - The file can be updated either manually or by the code **here**. 
    - To manually update, add the following variables ```<Path>,<Specimen>,<CD8>``` to the next line in the csv file
    - E.g. \\bki04\Clinical_Specimen_2,M1_1,CD8,CD8.ifp,
- Directory Structure: The code requires that the files are set up in the following format additional information on formatting can be found [here](../../../scans#45-directory-organization "Title"): <br>
  +--```<Path>```<br>
  | +-- ```<Specimen>``` <br>
  | | +-- *im3\flatw\*.im3*<br>
  | +-- tmp_inform_data <br>
  | | +-- Project_Development <br>
  | | | +-- ```<Algorithm>```(e.g. CD8.ifp) <br>
  | | +-- ```<Antibody>``` *(initialize at least 1 folder here)* <br>
  | | | +-- ```<computer_name>``` <br>
  | | | | +-- *Results* <br>
  
## 5.10.7.3. Instructions
1. Download the repository to a processing computer or location where inForm is installed
2. Copy the 'BatchProcessing-Shortcut' to the desktop 
3. Double click on the shortcut to launch processing
4. Add a jobs to the queue as described [above](#important-definitions "Title")

## 5.10.7.4. Workflow
The first step in processing is to test for the input version of inForm that will be used to run the software. The inForm version can be found at the bottom of **BatchProcessing.ps1**. Next the code attempts to injest the VM_inForm_queue.txt, described above in *important definitions*. This file should be located in the ```*\hpfs\RunInForm\BatchProcssing``` folder. The file consists of 
The code searches along the path location where the script was launched for the queue file. After the file is launched

After all jobs are marked complete, the code waits and rechecks the queue after a set period of time for new jobs. Jobs can be added to the queue either manually or by the code **here**. The queue file should be located in 'BatchProcessing' folder. The queue string is simple and consists of a five part comma separated list. To add a new job to the queue The code requires that the image data is reformatted into the AstroPath directory structure and that the algorithms are located in a set location adjacent to the images. Algorithm export settings should be set up before saving.
