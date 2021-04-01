# AstroID_gen
#### <div align="center">***v.0.00.0001***</div>
#### <div align="center">Created by: Sigfredo Soto-Diaz & Benjamin Green</div>
#### <div align="center">Tumor Microenvironment Technology Development Center</div>
#### <div align="center">The Johns Hopkins University Bloomberg~Kimmel Institute for Cancer Immunotherapy</div>
#### <div align="center">Correspondence to: ssotodi@jh.edu</div>

## ***Section 1: Summary***
This code is part of the astropath processing pipeline; it intializes a ```SlideID``` for each incoming specimen. These ```SlideID```s are used to standardize slide naming and replace the ```SampleName```s (names applied to the slides during the scanning process) on all subfiles for a specimen and in the scanning plan ('annotations.xml') files. The code prepares an *AstropathAPIDdef.csv* for each project to record the pairings of ```SlideID```s and ```SampleName```s. This file contains the following 5 columns ([described here](#section-2-important-definitions "Title")): 
```
SlideID, SampleName, ProjectID, CohortID, BatchID
```
The code also intializes the necessary [processing folders](#section-3-workflow "Title") for each project directory and saves a project level *AstropathAPIDdef_PP.csv* (```PP``` indicates the ```ProjectID```) in the ```upkeep_and_progress``` folder, containing only the specimens related to that project. The format for the ```SlideID``` is ```APpppXXXX``` where the ```ppp``` indicates the numeric ```ProjectID``` and the ```XXXX``` is a slide number which is unique within a project.```SlideID```s are generated by comparing the *AstropathAPIDdef_PP.csv* to the cohort specific *SpecimenTable.xlsx* and assigning each new specimen with a new value in sequential ordering (AP0010001, AP0010002, AP0020001 …). 

***Please Note:*** Since this code needs direct access to the *SpecimenTable.xlsx* which contains PPI, as such the naming generator code should only be run on the ```<Spath>``` server (HIPPA side server) to avoid opening the *SpecimenTable.xlsx* over the network. In addition, ```SlideID```s are assigned in the order they are added to their respective *SpecimenTable.xlsx* files. ```SlideID```s may ***not*** be sequential or may be *missing* for a project directory either because of failed batches or due to the order in which they are added to the *SpecimenTable.xlsx* files.

## ***Section 2: Important Definitions***
1. The  ```ProjectID``` and ```CohortID``` are defined in the *AstropathCohortsProgress.csv* file. 
   - A description of this file is contained in the [AstropathJHU/AstropathPipeline](https://github.com/AstropathJHU/AstropathPipeline "Title") repository
2. The ```BatchID``` is generated for each staining batch 
   - a longer description of this value can be found in the description in the [AstropathJHU/AstropathPipeline](https://github.com/AstropathJHU/AstropathPipeline "Title") repository
3. New ```SampleName```s are detected from the *SpecimenTable.xlsx* files contained in each cohort folder
   - The ```SampleName```s are the names defined during the scanning process
   - A description of the *SpecimenTable.xlsx* file is contained in the [AstropathJHU/AstropathPipeline](https://github.com/AstropathJHU/AstropathPipeline "Title") repository
4. ```SlideID```s are the names for the specimens in the astropath processing pipeline and replace the ```SampleName```s on all corresponding files and inside the scanning plan, annotations.xml, files generated during the scanning process
   - using these names allows us to avoid outside the organization changes to naming conventions
   - The IDs have the format; ```APpppXXXX```
     - ```ppp``` indicates the numeric ```ProjectID```
     - ```XXXX``` is a slide number which is unique within a project
   - The IDs are generated by comparing the *AstropathAPIDdef_PP.csv* to the cohort specific *SpecimenTable.xlsx*
     - we assign each new specimen with a new value in sequential ordering (AP0010001, AP0010002, AP0020001 …) 
5. We use path specifiers to shorten descriptions, further description of these paths can be found in the additional documentation under the [AstropathJHU/AstropathPipeline](https://github.com/AstropathJHU/AstropathPipeline "Title") repository:
   - ```<Mpath>```: the main path for all the astropath processing *.csv* configuration files; the current location of this path is ```\\bki04\astropath_processing```
   - ```<Dname>```: the data name or the name of the clinical specimen folder
   - ```<Dpath>```: the data or destination path
      - this is the path to the project's data on the bki servers
   - ```<Spath>```: the source path to the project's data
   
   *NOTE:* the ```<path>``` variables do not contain the ```<Dname>```
   
## ***Section 3: Workflow***
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
    - These tables are described in further documentation located in the [AstropathJHU/AstropathPipeline](https://github.com/AstropathJHU/AstropathPipeline "Title") repository
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

Information on these folders is also located in documentation located in the [AstropathJHU/AstropathPipeline](https://github.com/AstropathJHU/AstropathPipeline "Title") repository.

Finally, we compile the project level *AstropathAPIDdef_PP.csv* as follows:
1.	Check that *SpecimenTable.xlsx* exists. 
    - If it does ***not*** exist, move to the next directory. 
    - If it does, extract the ```Patient #``` and the ```BatchID``` columns for each sample
2.	Open the specimen level  *AstropathAPIDdef_PP.csv* if it exists
    - If the file does ***not*** exist start ```SlideIDs``` at APppp0001
    - If the file exists
      - Compare the ```SampleName```s in  *AstropathAPIDdef_PP.csv* to the ‘Patient #’ in *SpecimenTable.xlsx* to determine new specimens
      - Acquire next available ```SlideID``` from the *AstropathAPIDdef_PP.csv*
3.	Update the new ```SampleName``` rows to the *AstropathAPIDdef_PP.csv* file in the ```<Dpath>\<Dname>\upkeep_and_progress``` folder 
4.	Update the *AstropathAPIDdef.csv* file in the ```<Mpath>```

Once all cohorts have been checked the code will wait for 30 minutes, reload the *AstropathCohortsProgress.csv* file, then recheck all directories in such a way that the code is running in a continous loop.
