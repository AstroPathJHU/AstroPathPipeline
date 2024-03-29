# 4.7. AstroID Generation ***v.0.00.0001***

## 4.7.1. Description
This code is part of the astropath processing pipeline; it intializes a ```SlideID``` for each incoming specimen. These ```SlideID```s are used to standardize slide naming and replace the ```SampleName```s (names applied to the slides during the scanning process) on all subfiles for a specimen and in the scanning plan ('annotations.xml') files. The code prepares an *AstropathAPIDdef.csv* for each project to record the pairings of ```SlideID```s and ```SampleName```s. This file contains the following 5 columns ([described below](#472-important-definitions "Title")): 
```
SlideID, SampleName, ProjectID, CohortID, BatchID
```
Additionally, the code intializes the necessary [processing folders](#474-workflow "Title") for each project directory and saves a project level *AstropathAPIDdef_PP.csv* (```PP``` indicates the ```ProjectID```) in the ```upkeep_and_progress``` folder, containing only the specimens related to that project.
Next, this code updates the Batch files in the ```<Dpath>\<Dname>\Batch``` location by copying any files in ```<Spath>\<Dname>\Batch``` that are missing.
The format for the ```SlideID``` is ```APpppXXXX``` where the ```ppp``` indicates the numeric ```ProjectID``` and the ```XXXX``` is a slide number which is unique within a project.```SlideID```s are generated by comparing the *AstropathAPIDdef_PP.csv* to the cohort specific *SpecimenTable.xlsx* and assigning each new specimen with a new value in sequential ordering (AP0010001, AP0010002, AP0020001 …). 

***Please Note:*** Since this code needs direct access to the *Specimen_Table.xlsx* which contains PPI, the naming generator code should only be run on the ```<Spath>``` server (HIPPA side server) to avoid opening the *Specimen_Table.xlsx* over the network. In addition, ```SlideID```s are assigned in the order they are added to their respective *Specimen_Table.xlsx* files.

## 4.7.2. Important Definitions
1. The  ```ProjectID``` and ```CohortID``` are defined in the *AstropathCohortsProgress.csv* file. 
   - A description of this file is contained [here](../docs/AstroPathProcessingDirectoryandInitializingProjects.md#451-astropath_processing-directory)
2. The ```BatchID``` is generated for each staining batch 
   - a longer description of this value can be found in the description [here](../docs/scanning/BatchIDs.md#446-batchids "Title")
3. New ```SampleName```s are detected from the *SpecimenTable.xlsx* files contained in each cohort folder
   - The ```SampleName```s are the names defined during the scanning process
   - A description of the *SpecimenTable.xlsx* file is contained [here](../docs/scanning/SpecimenTable.md#442-specimen_table)
4. ```SlideID```s are the names for the specimens in the astropath processing pipeline and replace the ```SampleName```s on all corresponding files and inside the scanning plan, annotations.xml, files generated during the scanning process
   - using these names allows us to avoid outside-the-organization changes to naming conventions
   - The IDs have the format; ```APpppXXXX```
     - ```ppp``` indicates the numeric ```ProjectID```
     - ```XXXX``` is a slide number which is unique within a project
   - The IDs are generated by comparing the *AstropathAPIDdef.csv* to the cohort specific *SpecimenTable.xlsx*
     - we assign each new specimen with a new value in sequential ordering (AP0010001, AP0010002, AP0020001 …) 
5. We use path specifiers to shorten descriptions, further description of these paths can be found in the additional documentation [here](../docs/Definitions.md#432-path-definitions) repository:
   - ```<Mpath>```: the main path for all the astropath processing *.csv* configuration files; the current location of this path is ```\\bki04\astropath_processing```
   - ```<Dname>```: the data name or the name of the clinical specimen folder
   - ```<Dpath>```: the data or destination path
      - this is the path to the project's data on the bki servers
   - ```<Spath>```: the source path to the project's data
   
   *NOTE:* the ```<path>``` variables do not contain the ```<Dname>```

## 4.7.3. Instructions
For python download the repository and install the astroidgen. Then launch using:

```ASTgen.py <Mpath>```

- ```<Mpath>```: should contain the ***AstropathCohortsProgress.csv***, ***AstropathPaths.csv***, and the ***AstropathCohorts.csv*** files
  - description of these files can be found [here](../docs/AstroPathProcessingDirectoryandInitializingProjects.md#451-astropath_processing-directory)

## 4.7.4. Workflow
We begin by opening the *AstropathCohortsProgress.csv* file from the ```<Mpath>```. We process each available cohort from this file sequentially. We first check that data folder for the cohort on the bki servers (```<Dpath>\<Dname>```) and the source folder for the cohort (```<Spath>\<Dname>```) exist. Then, we either intialize or ensure that the following folders are intialized in the ```<Dpath>\<Dname>``` for processing. 
1.	```upkeep_and_progress```
    - For any upkeep and progress tracking files
    - Location of the *AstropathAPIDdef_PP.csv* files, where ```PP``` indicates the numeric project id
2.	```flatfield```
    - Location of the flatfield parameter files
    - These files are named ```Flatfield_BatchID_BB.bin```, replacing the ```BB``` for the appropriate batch id.
3.	```logfiles```
    - Project level log files for the astropath pipeline 
4.	```Batch```
    - The batch and merge tables
    - These tables are described in further documentation located [here](../docs/scanning/BatchIDs.md#446-batchids "Title")
5.	```Clinical```
    - Location of the clinical table
    - These tables should be labeled as *Clinical_Table_Specimen_CSID_MMDDYYYY.csv*, where the ```CSID``` indicates the number on the ```<Dname>``` folder. 
    - We always use the clinical table with the most recent date in the data upload
6.	```Ctrl```
    - Location of control TMA data output
7.	```dbload```
    - Location of the files used for the database upload
8.	```tmp_inform_data```
    - Location of the inform data output and inform algorithms used 
9.	```reject```
    - Location of the rejected slides

Information on these folders is also located in documentation located [here](../docs/DirectoryOrganization.md#461-directory-subfolders) repository.

Next, we update the available Batch folder ```<Dpath>\<Dname>\Batch``` with any possible missing Batch files in ```<Spath>\<Dname>\Batch```.

Finally, we compile the *AstropathAPIDdef.csv* as follows:
1.	Check that *Specimen_Table.xlsx* exists. 
    - If it does ***not*** exist, move to the next directory. 
    - If it does, extract the ```Patient #``` and the ```BatchID``` columns for each sample
2.	Open the *AstropathAPIDdef.csv* if it exists
    - If the file does ***not*** exist start ```SlideIDs``` at APppp0001
    - If the file exists
      - Compare the ```SampleName```s in  *AstropathAPIDdef.csv* to the ‘Patient #’ in *Specimen_Table.xlsx* to determine new specimens
      - Acquire next available ```SlideID``` from the *AstropathAPIDdef.csv*
3.	Update the new ```SampleName``` rows to the *AstropathAPIDdef.csv* file
4.	Compare the local *AstropathAPIDdef_PP.csv* file with *AstropathAPIDdef.csv* 
    - If there are no new ```SampleName```s, do not update *AstropathAPIDdef_PP.csv*
    - If there are new ```SampleName```s, recreate *AstropathAPIDdef_PP.csv* with all project-relevant entries


Once all cohorts have been checked the code will wait for 30 minutes, reload the *AstropathCohortsProgress.csv* file, then recheck all directories in such a way that the code is running in a continous loop.

## Credits
#### <div align="center">Created by: Sigfredo Soto-Diaz & Benjamin Green</div>
#### <div align="center">Tumor Microenvironment Technology Development Center</div>
#### <div align="center">The Johns Hopkins University Bloomberg~Kimmel Institute for Cancer Immunotherapy</div>
