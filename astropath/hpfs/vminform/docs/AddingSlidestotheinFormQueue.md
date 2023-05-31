# 5.8.5. Adding Slides to the inForm Queue
## 5.8.5.1. Description
The inForm Queue was designed to send processing tasks to the virtual machines for batch processing images from slides set up in the *AstroPath Pipeline* directory structure. The code processes the images from the ```flatw_path``` created by the ```image correction``` module ([5.7.](../../imagecorrection/README.md#57-image-correction "Title")). The user adds projects to the queue located in the project's *\*\Clinical_Specimen_XX\upkeep_and_progress* folder. Once projects are added, the ```mergeloop``` module automatically transfers the projects to the main queue located at the *```<mpath>```\across_project_queues\vminform-queue.csv*, where the ```vminform``` code automatically dessiminates tasks to the servers. Typically, users will only interface with the *inForm_queue.csv* in the project's *\*\Clinical_Specimen_XX\upkeep_and_progress* folder. The most recent projects used are recorded in the *samples_summary.xlsx* spreadsheet (also in the *upkeep_and_progress* folder).

## 5.8.5.2. Important Notes
- Once the pipeline properly finishes the ```imagecorrection``` module for a slide and correctly registers the *MergeConfig_NN.xlsx* file for that slide it will intialize the *inForm_queue.csv* with a line for each slide - antibody pair. If a slide - antibody pair is not intialized it means either the image corrections are not finished or the *MergeConfig_NN.xlsx* file was not set up\ not set up correctly. Directions on that file are located [here](../../../scans/docs/scanning/MergeConfigTables.md#448-mergeconfig-tables). 
- Only manually add rows when it is time to rerun a project. The code search for algorithms in the *\Clinical_Specimen_XX\tmp_inform_data\Project_Development* folder. For saftey reasons, the code will not search subfolder and the input is case senstive. Additional details on saving algorithms can be found [here](SavingProjectsfortheinFormJHUProcessingFarm.md#584-saving-projects-for-the-inform-jhu-processing-farm).
- For antibodies with multiple segmentaions, there will be a second row labeled ```<Antibody>_2```. Usually this is done when a marker can express on different sizes of cells (like PDL1), be sure to keep track of which algorithm was used for the large cells and which for the small cells.

## 5.8.5.3. Instructions
1. Navigate to the *\*\Clinical_Specimen_XX\upkeep_and_progress* folder.
2. Open the *InForm_Queue.csv* file and fill out the columns
   - this file has the following columns: ```Path, Specimen, Antibody, Algorithm, Processing```
    - ```Path```: The path up to but not including the specimen folder or ```\\<Dname>\<Dpath>``` ([described here](../../../scans/docs/Definitions.md/#432-path-definitions))
      - E.g. *\\\\bki04\Clinical_Specimen_2* 
    - ```Specimen```: The name of the specimen to be analyzed
      - The image data to be processed should be located in an *```<Path>```\\```<Specimen>```\\im3\\flatw* folder 
      - E.g. *M1_1*
    - ```Antibody```: The antibody name that will be processed.
      - All data will be exported into a *```<Path>```\\tmp_inform_data\\```<Antibody>```* subfolder
      - E.g. *CD8*
    - ```Algorithm```: The name for the **project** to do the processing **include the file extension**
      - The **project** should be location in a *```<Path>```\\tmp_inform_data\\Project_Development* folder for the code to be able to find it
      - Only the *Project_Development* folder will be search, the search is not recursive. Subfolders will not be searched for the algorithm
      - E.g. *CD8.ifp*
    - ```Processing Location```: Where the slide is being processed
    - ```Start```: The time that the processing started, this is updated by the code and should be left blank
3. Save and close the file

*NOTE*: you only need to fill out the ```Algorithm``` column for the first time a slide - antibody pair is run. To rerun a slide - antibody pair add it to a new row with the new **project** (be sure to fill out the ```Path``` column as well).

