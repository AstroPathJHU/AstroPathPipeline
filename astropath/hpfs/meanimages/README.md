# 5.7. meanimages

## 5.7.1. Description
The module is used to build a mean image for each slide after it is transferred. The module searches through each slide under all projects defined in the *AstropathCohortsProgress.csv* file for the mean image in the designated location. If the mean image does not exist but the *.qptiff* has been generated the code begins building a mean image. The code uses the *.qptiff* file as an indicator for a successful transfer because this is the last file transferred by the ```TransferDeamon``` ([described here](../transferdaemon#56-transfer-daemon). The input to the code is the directory where the *AstropathCohortsProgress.csv* file exists and a folder location for the code to read and write necessary files (see [4.5.](../../scans/docs/AstroPathProcessingDirectoryandInitializingProjects.md#45-astropath-processing-directory-and-initializing-projects) for details on that file).

## 5.7.2. Important Definitions
   - ```<Mpath>```: the main path for all the astropath processing *.csv* configuration files; the current location of this path is ```\\bki04\astropath_processing``` for the JHU pipeline. Further description of all files is in [4.5.](../../scans/docs/AstroPathProcessingDirectoryandInitializingProjects.md#45-astropath-processing-directory-and-initializing-projects)
   - ```<Dname>```: the data name or the name of the clinical specimen folder
   - ```<Dpath>```: the data or destination path
   - ```<drive>```: the location where the code will read and write temporary files to during processing
   - ```<meanimage-output-flt>```: the output flt file should be named *```<SlideID>```-mean.flt*. This file contains the *total* image for all im3s in a slide in as a single column vector. The order is layer, width, height. 
   - ```<meanimage-output-csv>```: the output csv file should be named *```<SlideID>```-mean.csv*. This file contains four numbers: the number of images used, number of image layers, image width, and image height.
   
   *NOTE*: the ```<path>``` variables do not contain the ```<Dname>```. 
   *NOTE*: Both the *.flt* and the *.csv* output files should be located in the ```<Dpath>\<Dname>\<SlideID>\<im3>``` folder.
   

## 5.7.3. Instructions
The code should be launched through matlab. To start download the repository to a working location. Next, open a new session of matlab and add the ```AstroPathPipline``` to the matlab path. Then use the following to launch:
   ``` meanimages(<Mpath>, <drive>) ``` 
   - ```<Mpath>[string]```: the full path to the directory containing the *AstropathCohortsProgress.csv* file
      - description of this file can be found [here](../../scans/docs/AstroPathProcessingDirectoryandInitializingProjects.md#45-astropath-processing-directory-and-initializing-projects)
   - ```<drive>[string]```: the full path to a drive or folder where the code can read and write necessary temporary files to
   
## 5.7.4. Workflow
### 5.7.4.1. Checking for Tasks
The code starts by checking each slide in the first project for the ```<meanimage-output-csv>``` file located in the ```<Dpath>\<Dname>\<SlideID>\<im3>``` folder. A full description of the folder structure can be found [here](../../scans/docs/DirectoryOrganization.md#46-directory-organization). If there is a ```<meanimage-output-csv>``` or no *.qtiff* the code moves on. The code uses this file because this is the last file transferred by the ```TransferDeamon``` ([described here](../transferdaemon#56-transfer-daemon)). 

### 5.7.4.2. Shred Im3s
If a slide does not have a mean image the code immediately begins processing that slide. The first step in this processing is to extract the image data from each im3 into a new *```<drive>```\Processing_Specimens\raw* directory. The code uses the ```ConvertIm3Path.ps1``` utility to extract this data into *Data.dat* files. The documentation for this code can be found [here](../flatw/docs/AdditionalTools.md#5853-convertim3path--convertim3cohort).

### 5.7.4.3. raw2mean
After the im3s are extracted by the ```ConvertIm3Path.ps1``` utility, the code checks the image sizing from the *.xml* files the utility creates. After this each image is read in and added together. Finally, the ```<meanimage-output-flt>``` and ```<meanimage-output-csv>``` files are produced. 

Next, the code cleans up all files, transfers the output files back to the ``<Dpath>\<Dname>\<SlideID>\<im3>``` folder, and continues to the next slide.
