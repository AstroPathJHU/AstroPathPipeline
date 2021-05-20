# 5.13. Clean Up
## 5.13.1. Description
Here we check that there is the proper number of orginial *.im3*, flatw *.im3s*, *.fw* files, *.fw01* files, *.xml* files, *component_data*, *component_data_w_seg*, and *cleaned_phenotyped_tables* for each ```<SlideID>```. We also check that the bytes in the files are within reasonable ranges. This code should be launched after all modules have been completed including evaluating all phenotyped quality control images and the *segmaps* module. This module can be run itermittently during processing to confirm that the *transferdeamon* or *flatw* modules successfully processed each ```<SlideID>```. The code produces a *sample_error_codes.csv* which is located in the ```<upkeep_and_progress>``` folder. The file contains the following columns:

```<slideid>,<im3>,<fw>,<flatw>,<xml>,<comps>,<tbl>,<total>```
- ```<slideid>```: this is the slide id for a slide
- ```<im3>```: takes a value of 0 for no errors or integer > 0 for files with flagged bytes or missing orignial *.im3*s 
- ```<fw>```: takes a value of 0 for no errors or integer > 0 for files with flagged bytes or missing *.fw*/ *.fw01* files 
- ```<flatw>```: takes a value of 0 for no errors or integer > 0 for files with flagged bytes or missing flatw *.im3*s 
- ```<xml>```: takes a value of 0 for no errors or integer > 0 for files with flagged bytes or missing *.xml* files
- ```<comps>```: takes a value of 0 for no errors or integer > 0 for files with flagged bytes or missing *component_data* or *component_data_w_seg* files
- ```<tbl>```: takes a value of 0 for no errors or integer > 0 for files with flagged bytes or missing *cleaned_phenotyped_tables*  files
- ```<total>```: a summation of all error values takes a value of 0 for no errors or integer > 0 for files with flagged bytes or missing files

## 5.13.2. Instructions
The code should be launched through matlab. To start download the repository to a working location. Next, open a new session of matlab and add the ```AstroPathPipline``` to the matlab path. To launch across a single cohort use:
``` cleanup_cohort(<base>, <FWpath>) ``` 
- ```<base>[string]```: The ```<Dpath>\<Dname>```folder for a project
  - E.g. *\\bki04\Clinical_Specimen* 
- ```<FWpath>[string]```: The flatw path for a project
  - E.g. *\\bki04\flatw*  
