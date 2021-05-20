# 4.6. Directory Organization
The following is the directory structure for the *AstroPath Pipeline* in the ```<Dpath>\<Dname>``` folder. The code initializes most of the folders if running the pipeline from the beginning and places data in the proper locations. 

*NOTE*: ```<>``` indicate variables, lines without brackets indicate full names

+-- ```<Dname>\<Dpath>``` <br>
|  +-- ```<SlideID>``` <br>
|  |  +-- im3 <br>
|  |  |  +-- ```<ScanNN>``` <br>
|  |  |  | +-- BatchID.txt <br>
|  |  |  | +-- CheckSums.txt <br>
|  |  |  | +-- ```<SlideID>```\_```<ScanNN>```.annotations.polygons.xml <br>
| | | | +-- ```<SlideID>```\_```<ScanNN>```\_annotations.xml <br>
| | | | +-- ```<SlideID>```\_```<ScanNN>```\_annotations.xml.lock <br>
| | | | +-- ```<SlideID>```\_```<ScanNN>```.qptiff <br>
| | | | +-- MSI <br>
| | | | | +-- \*.im3 files <br>
| | | +-- flatw <br>
| | | | +-- \*.im3 files <br>
| | | +-- xml <br>
| | | | +-- \*.xml files<br>
| | | +-- ```<SlideID>```-mean.csv <br>
| | | +-- ```<SlideID>```-mean.flt <br>


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
    - These tables are described in further documentation located [here](#435-batchids "Title") repository
5.	```Clinical```
    - Location of the clinical table
    - These tables should be labeled as *Clinical_Table_Specimen_CSID_MMDDYYYY.csv*, where the ```CSID``` indicates the number on the ```<Dname>``` folder. 
    - We always use the clinical table with the most recent date in the data upload
6.	```Ctrl```
    - Location of control TMA data output
7.	```dbload```
    - Location of the files used for the database upload
8.	```tmp_inform_data```
    - Location of the inform data output and inform algorithms used. **Additional information on this folder is provided in the hpf processing documentation.**
9.	```reject```
    - Location of the rejected slides
