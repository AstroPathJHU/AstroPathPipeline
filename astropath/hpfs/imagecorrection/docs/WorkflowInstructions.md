# 5.7.4. Workflow Instructions
This module can be launched acrross all projects using the DispatchTasks-imagecorrection.bat file located in the [launch folder](../../../launch). 

You can also launch a specific project using the following:
```
Import-Module '*.\astropath'; DispatchTasks  -mpath:<mpath> -module:'imagecorrection' -project:<project>
```
- replace '\*' with the location up to and including the *AstroPathPipeline* repository
- ```<mpath>```: the main path for all the astropath processing .csv configuration files; the current location of this path is *\\bki04\astropath_processing*
- ```<Project>```: Project Number

The workflow will ask for your credentials in a windows credential window, these are stored as an encrpyted network key used to spin workers up and down accordingly. Next, tasked are launched on the worker locations defined in the *AstroPathHPFsWLoc.csv* file for each slide that has not yet been completed. Trigger events for this module include, the flatfield_BatchID.bin file for this batch exists, that the flatfield files do not yet exist, that the files have not yet started, or if they have did not finish with an error. The code will attempt to rerun slides if they finish with an error. 
