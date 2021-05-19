# 4.5. AstroPath_Processing Directory and Initializing Projects
The code is driven by the files located in a main processing folder, named the ```<Mpath>```. These files are described below followed by descriptions of the respectve columns. For columns without definitions provided, please check [4.2](#42-definitions "Title") above. After a description of the directory and files included, instructions for intializing projects into the pipeline are provided.

## 4.5.1. AstroPath_Processing Directory
- *AstropathCohortsProgress.csv*: This file contains information on the project's analysis status and important experimental variables. This table is manually updated. The file has the following columns:
  ```
  Project, Cohort, Dpath, Dname, Machine, Method, Panel, Tissue, StainConfig, Stain, Scan, Inform, Merged, QC, Annotations, ReadyForDB, DBLoad
  ```
  - ```Stain[string]```: The staining status of the panel. Should be indicated as blank (not started), *Started*, or *Done*.
  - ```Inform[string]```: The status of the inform algorithm for classification and segmentation. Should be indicated as blank (not started), *Started*, or *Done*.
  - ```Merged[string]```: The status of the inform processing and merge. Should be indicated as blank (not started), *Started*, or *Done*.
  - ```QC[string]```: The status of the quality control assessment of samples. Should be indicated as blank (not started), *Started*, or *Done*.
  - ```Annotations[string]```: Whether or not the slide annotations have been created for a panel. ***Annotation directions can be found in***. Should be indicated as blank (not started), *Started*, or *Done*.
  - ```ReadyForDB[string]```: Indicates whether final checks for the manual interaction steps with the data have been complete. ***A full checklist is still in progress.***. Should be indicated as blank (not started), *Started*, or *Done*.
  - ```DBLoad[string]```: The current status of the database load. Should be indicated as blank (not started), *Started*, or *Done*.
- *AstropathConfig.csv*
- *AstropathControldef.csv*
- *AstropathPaths.csv*
- *AstropathSampledef.csv*
- *AstropathAPIDdef.csv*

## 4.5.2. Initializing Projects


