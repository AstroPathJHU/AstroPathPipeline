# 5.4. Workflow Overview
As noted above, modules outside of ```segmaps``` and ```transferanno``` can be launched and allowed to run continuously. These steps outline how a slide might process through the workflow. 

1. Update the AstroPath files in the ```<mpath>```
2. Update the slides in the *Specimen_Table_N.xlsx*
3. Scan the slides according to the documentation in [4.4.](../../scans/docs/ScanningInstructionsIntro.md#44-scanning-instructions-intro)
4. Launch the ```AstroIDGen``` module to intialize the slides into the pipeline
5. Launch the ```TransferDeamon``` module to start transferring completed slides from the ```<Spath>``` location to the ```<Dpath>\<Dname>```
6. Launch the ```meanimages``` module to create mean images of each slide as it finishes transferring; these will be used to build the batch flatfields. This is usually launched after all slides from a batch have been transferred as it only loops through the ```Project```s once then stops. Creating the *meanimage* for all slides in a batch is also the trigger for the next step, but the code does not yet have the ability to know beforehand how many slides are in each batch which may cause a false start of the next step.
7. Launch the ```flatw_queue``` module to create the flatfields and assign new slides to the flatw_queue. Then the module distributes flatw jobs to the assigned workers.
8. Launch the ```flatw_worker``` module on the assigned worker machine to process the flatfielding and image warping corrections on a particular slide's hpf image set
9. Create the BatchID and MergeConfig files for the project according to documentation in scans
10. Launch the ```mergeloop``` module to initialize necessary antibody processing folders, create the local inform_queue for a project, recieve inform results, merge the data and create qa qc images to evaluate the inform classification algorithms.
11. Create phenotype algorithms in inForm according to the protocol established in [5.8.3.](../vminform/docs/inFormMultipassPhenotype.md#583-inform-multipass-phenotype)
12. Launch the ```inform_queue``` module to send jobs from the main inform queue to the queues on the inform worker machines.
13. Launch the ```inform_worker``` module to process algorithms in inForm
14. wait for QC images to process by the ```mergeloop``` module
    - evaluate the qc by the protocols established in [5.8.6.](../vminform/docs/EvaluatinginFormPhenotypeQCOutputfortheAstroPathPipeline.md#586-evaluating-inform-phenotype-qc-output-for-the-astropath-pipeline)
    -repeat 11-13 as needed
15. Launch ```segmaps``` module after qc has been completed for the cell classification of slides in a project to build the final segmenation maps.
16. Launch ```transferanno``` module after the slides have been successfully annotated in HALO and annotations have been exported to a desired location.
17. Launch ```cleanup``` and the associated script ```convert_batch``` from the ```cleanup``` module
18. manually unmix the control tma component tiffs and place them in the *inform_data\Component_Tiffs* folder of each respective tma

- ```AstroIDGen``` should be launched on the ```<Spath>``` location.  
- The following modules can be launched at the same time: ```TransferDeamon```, ```meanimages```, ```flatw_queue```, ```inform_queue```, ```mergeloop```. 
- The following modules must be launched on their respective "worker" locations: ```flatw_worker``` and ```inform_worker```. 
- Launching ```segmaps``` runs over the entire set of projects but is not a continous loop and must be restarted to reprocess
- ```transferanno``` is a simple script that distributes and renames the halo annotations from a single folder to corresponding slide folders in a single project folder
