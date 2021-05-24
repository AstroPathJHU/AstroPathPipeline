# 5.8.4. Workflow Instructions
This module consists of two submodules, the first ```flatw_queue``` builds the flatfield model, adds slides to the queue, and distributes jobs from the queue to pre-defined workers. The second ```flatw_worker```, launches and carries out the flatfield and image warping corrections on a set of images belonging to a slide. Each worker location should have a copy of the repository and a *Processing_Specimens\flatw_qo.txt* file in the directory, the ```flatw_queue``` module will skip directories without this file. This file provides a simple method for scaling the number of workers in use. 

## 5.8.4.1. flatw_queue
The code should be launched through MATLAB. To start download the repository to a working location. Next, open a new session of matlab and add the ```AstroPathPipline``` to the MATLAB path. Then use the following to launch:   
``` 
flatw_queue(<Mpath>)
``` 
- ```<Mpath>[string]```: the full path to the directory containing the *AstropathCohortsProgress.csv* file
   - description of this file can be found [here](../../../scans/docs/AstroPathProcessingDirectoryandInitializingProjects.md#451-astropath_processing-directory "Title")
  
## 5.8.4.2. flatw_worker   
The code should be launched through MATLAB. To start download the repository to a working location. Next, open a new session of matlab and add the ```AstroPathPipline``` to the MATLAB path. Create a *Processing_Specimens\flatw_qo.txt* file in the directory that the slides should be processed in. Because the flatw workflow reads and writes a number of files, this directory should ideally be on a SSD with a significant (~500GB) amount of storage available. The code copies all files here before processing so that processing is not hindered by network performance so it is also advised that the worker location have a good access to the orginial data files. Then use the following to launch:
``` 
flatw_worker(loc)
``` 
- loc: The drive location for the worker to use when writing temporary files. All files will be cleaned up after successful processing.  
