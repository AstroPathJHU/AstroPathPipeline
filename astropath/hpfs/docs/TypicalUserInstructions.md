# 5.3.2. Typical User Instructions
Typical user interaction with the data occurs after the image corrections have been applied to the images and is where we pick up in this section. 

1. Once slides have image corrections applied, the user should create inform phenotype projects in the multipass format. (see [5.8.3.](../vminform/docs/inFormMultipassPhenotype.md#583-inform-multipass-phenotype))
2. Algorithms should have export settings applied according to the documentation in [5.8.4.](../vminform/docs/SavingProjectsfortheinFormJHUProcessingFarm.md)) then be saved into the ```<Project_Development>``` subfolder. 
   - If algorithms are placed anywhere else, including subfolders under the ```<Project_Development>``` folder, they will not be found for processing during the next steps (```<inform_processing>``` module).
3. Check that the *MergeConfig_NN.xlsx* files have been created for the slide's batch and added to the appropriate location, according to the documentation [here](../../scans/docs/scanning/MergeConfigTables.md#448-mergeconfig-tables). 
4. Add slides to the *inform_queue.csv* according to the protocol in [5.8.5.](../vminform/docs/AddingSlidestotheinFormQueue.md#585-adding-slides-to-the-inform-queue)
   - When the code has verified that the slides are ready for inForm processing, it will preallocate initial rows for the slide. One row will be preallocated for each slide-antibody pair, with the antibodies defined in the *MergeConfig_NN.xlsx* file. 
     - If the slide-antibody pair does not pre-allocate or too many slide-antibody pairs pre-allocate, either check that the image corrections for the slide has completed or check the formatting of the *MergeConfig_NN.xlsx*.
5. Wait for the inform machines to process the queue and the code to pull data to the ```<SlideID>\<inform_data>``` folder (this processing is outlined here).
   - Processing information is updated in the *samples_summary.xlsx* spreadsheet, including the algorithms, date of processing, and number of files for each antibody
   - Once all antibodies for a slide have been completed the code will merge the data and create the ```<QA_QC>``` folder of a ```<SlideID>``` directory. [details on the merge found here](../mergeloop/MaSS#merge-a-single-sample-mass).
   - For the quality control images to be generated the following must be present:
     - Data for each image indicated in the export documentation ([5.8.4.](../vminform/docs/SavingProjectsfortheinFormJHUProcessingFarm.md#584-saving-projects-for-the-inform-jhu-processing-farm)) of each antibody must be present in the ```<SlideID>\<inform_data>\<Phenotyped>``` . 
	 - The component data must also be present in the ```<SlideID>\<inform_data>\<Component_Tiffs>``` directory.
6. Once QC images have been created, qc the slides according to the protocol in [5.8.6.](../vminform/docs/EvaluatinginFormPhenotypeQCOutputfortheAstroPathPipeline.md#86-evaluating-inform-phenotype-qc-output-for-the-astropath-pipeline)
7. If an antibody for a slide fails qc, re-work the phenotype algorithm and resubmit algorithms according to the protocol. 
   - Continue this process until all slide-antibody pairs pass quality control in a cohort.   

Simulanteously with cell classification a pathologist should use HALO to annotate regions of interest in the QPTiff images then export them according to the documentation outlined in [5.11.](../transferanno#511-transfer-annotations).
 
Once processing reaches this point the user should direct processing to the person maintaining the code. The modules ```<segmaps>```, ```<transferanno>```, and ```<cleanup>``` should each be launched. The ```<cleanup>``` module should be launched last. Afterward steps in the [clean up documentation](../validatedata#512-validate-data) should be taken to extract missing files, export the control tma component tiffs and convert the batch\ merge tables to the corresponding acceptable csv format. 
 
After the cleanup protocols the ```<Project>``` is ready to proceed to the the ```Samples``` processing stage.
