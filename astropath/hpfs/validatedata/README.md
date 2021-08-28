# 5.13. Validate Data
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

## 5.13.3. Next Steps
Once the code finishes check the *sample_error_codes.csv* for errors in slides. If the error code exists for ```flatw``` or ```fw``` relaunch the entire slide through the flatfielding protocol. For errors in the component images or inform; first note whether or not the slide has errors earlier in its processing and handle those before attempting to move on. Next, either reprocess each algorithm from the slide through the inForm queue or just reprocess those images that failed through inForm (the batch logs are saved in the respective *```<SlideID>```\inform_data\antibody folders*). Put the respective antidody outputs into the respective folders. At this point if the ```mergeloop``` is running it should reprocess the merge tables and ```segmaps``` can be subsequently relaunched. (A given slide can also be re-merged outside the ```mergeloop``` using the standalone utility```MaSS```). Some inForm error codes cannot be solved. In that case, ```<comps>``` should not have an error code but ```<tbl>``` may. An example of this kind of error is when inForm declares "one or more cells where defined in the tissue seg but not in cell". There is nothing we can do to further process those images at this time. This code can and should be peridocially on slides to make sure everything is processing as expected.

After all error codes have been handled launch the ```convert_batch``` code to convert the *BatchID.xlsx* and *MergeConfig.xlsx* files to *csv* files with the correct formatting. Finally, the last step in processing (though this can be done anytime after the tma has transferred), unmix the control tmas through inform manually and place the *component_data.tif* images for each slide into a *inform_data/Component_Tiffs* folder for that control tma.
