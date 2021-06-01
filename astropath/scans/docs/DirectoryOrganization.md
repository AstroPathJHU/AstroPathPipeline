# 4.6. Directory Organization
## 4.6.1. Directory Subfolders
The following folders are located in the ```<Dpath>\<Dname>``` folder and are intialized by the code for the *AstroPath Pipeline*. The most interesting folders for typical users are the ```upkeep_and_progress``` and the ```tmp_inform_data``` folders.

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
    - These tables are described in further documentation located [here](scanning/BatchTables.md) and [here](scanning/MergeConfigTables.md) respectively
5.	```Clinical```
    - Location of the clinical table
    - These tables should be labeled as *Clinical_Table_Specimen_XX_MMDDYYYY.csv*, where the ```XX``` indicates the number on the ```<Dname>``` folder. 
    - We always use the clinical table with the most recent date in the data upload
6.	```Ctrl```
    - Location of control TMA data output
7.	```dbload```
    - Location of the files used for the database upload
8.	```tmp_inform_data```
    - Location of the inform data output and inform algorithms used. Additional information on this folder is provided [here](../../hpfs/mergeloop#592-important-definitions)
9.	```reject```
    - Location of the rejected slides

## 4.6.2. SlideID Subfolders
The following is the directory structure for each ```<SlideID>``` in the *AstroPath Pipeline* in the ```<Dpath>\<Dname>``` folders. The code initializes most of the folders if running the pipeline from the beginning and places data in the proper locations. 

*NOTE*: ```<>``` indicate variables, lines without brackets indicate full names
```
+-- <Dname>\<Dpath> <br>
|  +-- <SlideID> <br>
|  |  +-- im3 <br>
|  |  |  +-- <ScanNN> <br>
|  |  |  |  +-- BatchID.txt <br>
|  |  |  |  +-- CheckSums.txt <br>
|  |  |  |  +-- <SlideID>\_<ScanNN>.annotations.polygons.xml <br>
|  |  |  |  +-- <SlideID>\_<ScanNN>\_annotations.xml <br>
|  |  |  |  +-- <SlideID>\_<ScanNN>\_annotations.xml.lock <br>
|  |  |  |  +-- <SlideID>\_<ScanNN>.qptiff <br>
|  |  |  |  +-- MSI <br>
|  |  |  |  |  +-- <SlideID>_[XXXX,YYYY].im3 files <br>
|  |  |  +-- flatw <br>
|  |  |  |  +-- <SlideID>_[XXXX,YYYY].im3 files <br>
|  |  |  +-- xml <br>
|  |  |  |  +-- <SlideID>_[XXXX,YYYY].xml files<br>
|  |  |  |  +-- <SlideID>.Full.xml
|  |  |  |  +-- <SlideID>.Parameters.xml
|  |  |  +-- <SlideID>-mean.csv <br>
|  |  |  +-- <SlideID>-mean.flt <br>
|  |  +-- inform_data
|  |  |  +-- Component_Tiffs
|  |  |  |  +-- <SlideID>_[XXXX,YYYY]_component_data.tif files
|  |  |  |  +-- <SlideID>_[XXXX,YYYY]_component_data_w_seg.tif files
|  |  |  |  +-- Batch.log
|  |  |  |  +-- batch_procedure.ifp
|  |  |  +-- Phenotyped
|  |  |  |  +-- <ABX>
|  |  |  |  |  +-- <SlideID>_[XXXX,YYYY]_cell_seg_data.txt
|  |  |  |  |  +-- <SlideID>_[XXXX,YYYY]_cell_seg_summary_data.txt
|  |  |  |  |  +-- <SlideID>_[XXXX,YYYY]_binary_seg_maps.tif
|  |  |  |  |  +-- Batch.log
|  |  |  |  |  +-- batch_procedure.ifp
|  |  |  |  +-- Results
|  |  |  |  |  |  +-- Tables
|  |  |  |  |  |  |  +-- <SlideID>_[XXXX,YYYY]_cleaned_phenotype_table.csv
|  |  |  |  |  |  |  +-- MaSSlog.txt
|  |  |  |  |  |  +-- QA_QC *See QC documentation for files*
|  |  +-- logfiles
|  |  +-- dbload
|  |  +-- geom
|  |  |  +-- <SlideID>_[XXXX,YYYY]_cellGeomLoad.csv
```

\*[link to QC documentation](../../hpfs/mergeloop/MaSS#section-83-output)
