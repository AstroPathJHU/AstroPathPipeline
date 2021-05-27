# 5.10.5. Adding Slides to the inForm Queue
## 5.10.5.1. Description
The inForm Queue was designed to send processing tasks to the virtual machines for batch processing images from slides set up in the *AstroPath Pipeline* directory structure. The code processes the images from the ```flatw_path``` created by the ```flatfield``` (or image correction) module ([5.8.](../../flatw#58-flatw "Title")). The user adds algorithms to the queue located in the project's *\*\Clinical_Specimen_XX\upkeep_and_progress* folder. Once algorithms are added, the ```mergeloop``` module automatically transfers the algorithms to the main queue located at the *```<mpath>```\across_project_queues\inForm_queue.csv*, where the ```inform_queue``` code automatically dessiminates tasks to the servers. Typically, users will only interface with the *inForm_queue.csv* in the project's *\*\Clinical_Specimen_XX\upkeep_and_progress* folder. The most recent algorithms used are recorded in the *samples_summary.xlsx* spreadsheet (also in the *upkeep_and_progress* folder).

## 5.10.5.2. Important Notes
- Once the pipeline properly finishes the ```flatfield``` module for a slide and correctly registers the *MergeConfig_NN.xlsx* file for that slide it will intialize the *inForm_queue.csv* with a line for each slide - antibody pair. If a slide - antibody pair is not intialized it means either the image corrections are not finished or the *MergeConfig_NN.xlsx* file was not set up\ not set up correctly. Directions on that file are located [here](../../../scans/docs/scanning/MergeConfigTables.md#448-mergeconfig-tables). 
- Only manually add rows when it is time to rerun an algorithm. The code search for algorithms in the *\Clinical_Specimen_XX\tmp_inform_data\Project_Development* folder. For saftey reasons, the code will not search subfolder and the input is case senstive. Additional details on saving algorithms can be found [here](SavingProjectsfortheinFormJHUProcessingFarm.md#5104-saving-projects-for-the-inform-jhu-processing-farm).
- For antibodies with multiple segmentaions, there will be a second row labeled ```<Antibody>_2```. Usually this is done when a marker can express on different sizes of cells (like PDL1), be sure to keep track of which algorithm was used for the large cells and which for the small cells.

## 5.10.5.3. Instructions
1. Navigate to the *\*\Clinical_Specimen_XX\upkeep_and_progress* folder.
2. Open the *InForm_Queue.csv* file and fill out the columns
   - this file has the following columns: ```Path, Specimen, Antibody, Algorithm, Processing```
   - ```Path```: the full path up to and including the *Clinical_Specimen_XX* folder or the ```<Dname>\<Dpath>``` folder
     - E.g. *\\bki04\Clinical_Specimen*
   - ```Specimen```: The slide name
     - E.g. *M1_1*
   - ```Antibody```: The antibody to run
     - E.g. *CD8*
   - ```Algorithm```: The complete name of the algorithm or project to use for processing (**include the file extension**)
     - E.g. *Tumor.ifp*
     - Be sure that this algorithm is in the *\Clinical_Specimen_XX\tmp_inform_data\Project_Development* folder, the code will not search subfolders
   - ```Processing```: whether or not the slide has been sent for prcoessing
     - **leave this column blank the code will fill it in**
3. Save and close the file

*NOTE*: you only need to fill out the ```Algorithm``` column for the first time a slide - antibody pair is run. To rerun a slide - antibody pair add it to a new row with the new algorithm (be sure to fill out the ```Path``` column as well).

