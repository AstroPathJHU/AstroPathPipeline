# 5.8. Flatfield
## 5.8.1. Description
This workflow serves to create a directory of flat field and warping corrected *.im3* images files for each slide. In addition, this workflow saves the full metadata for the first *.im3* (*.full.im3*) for a slide, the single column bitmap of each corrected *.im3* (*.fw* files) as well as relevant image metadata (*.SpectralBasisInfo.xml*) of each im3 image. We assume that the directory format of the input image files is in the ```AstroPathPipeline``` processing file stucture described **here** and below in [5.8.2.1.](#5821-flatw-expected-directory-structure "Title"). 

## 5.8.2. Important Definitions
### 5.8.2.1. Flatw Expected Directory Structure
Since this section of the pipeline can be used with standalone funcationality, we define the directory structure here. A more detailed directory structure for the whole AstroPath Pipeline can be found *here*: 
```
<base>\<astroID>\<im3_path>\<filename>
```
Example:  “\\bki04\Clinical_Specimen\ AST123456\ im3\Scan1\MSI\M41_1_[34888,4694].im3”
- ```<base>```: “\\bki04\Clinical_Specimen”
- ```<SlideID>```: “AST123456
- ```<im3_path>```: “im3\ScanX\MSI or im3\ScanXX\MSI”
- ```<filename>```: “AST123456_[34888,4694]”
- ```<extension>```:  “.im3”
*NOTE*: The Scan number directory would be the highest scan number in the <SlideID> folder, not always ‘1’ and may be multiple digits
- *BatchID.txt*: This file should contain the batch id that the slides were stained with. The code looks for the batch id as a specifier on the flatfield bin file.
- *flatfield_BatchID_NN.bin*: This is the bitmap data of the average image for a batch. 
  - The code will use the specifier *NN* from the *BatchID.txt* to find the appropriate flatfield file to use. 
  - This file should be kept in a *```<base>```\flatfield* directory (A directory adjacent to the ```<SlideID>``` directory)

## 5.8.2.2. Output Formatting
- For the output images we replace the ```<im3_path>``` with the ```<flatw_im3_path>```: “im3\flatw”, but nothing from the file name changes.
- For the additional output data of each image (the single column bitmap and additional image metadata) we change the base path to an adjacent directory we call the “FWpath” directory. The data for each given <SlideID> will be contained in the respective <FWpath>\<SlideID> directory.
- The full metadata for the first im3 image will be located at the “im3” level of the source path.

## 5.8.3. Instructions
This workflow consists of two modules, the first ```flatw_queue``` builds the flatfield model, adds slides to the queue, and distributes jobs from the queue to pre-defined workers. The second ```flatw_worker```, launches and carries out the flatfield and image warping corrections on a set of images belonging to a slide. The predefined workers should be added to the *AstropathHPFsWlocs.csv* file located in the ```<Mpath>``` folder, documentation on that file is located **here**. Each worker location should have a copy of the repository and a *Processing_Specimens\flatw_qo.txt* file in the directory, the ```flatw_queue``` module will skip directories without this file. This file provides a simple method for scaling the number of workers in use. 

## 5.8.3.1. flatw_queue
The code should be launched through matlab. To start download the repository to a working location. Next, open a new session of matlab and add the ```AstroPathPipline``` to the matlab path. Then use the following to launch:   
``` flatw_queue(<Mpath>) ``` 
- ```<Mpath>[string]```: the full path to the directory containing the ***AstropathCohortsProgress.csv*** file
   - description of this file can be found [here](../../scans#441-astropath_processing-directory "Title")
  
## 5.8.3.2. flatw_worker   
The code should be launched through matlab. To start download the repository to a working location. Next, open a new session of matlab and add the ```AstroPathPipline``` to the matlab path. Make sure to add the worker location to the *AstropathHPFsWlocs.csv* file located in the ```<Mpath>``` folder, documentation on that file is located **here**. Create a *Processing_Specimens\flatw_qo.txt* file in the directory that the slides should be processed in. Because the flatw workflow reads and writes a number of files, this directory should ideally be on a SSD with a significant (~500GB) amount of storage available. The code copies all files here before processing so that processing is not hindered by network performance so it is also advised that the worker location have a good access to the orginial data files. Then use the following to launch:
``` flatw_worker(loc) ``` 
- loc: The drive location for the worker to use when writing temporary files. All files will be cleaned up after successful processing.  

## 5.8.3.3. Im3tools
This is the standalone processing tool for the flatfield and image warping applications on a directory of im3 slides. These commands can be used outside of the ```AstroPathPipeline``` as long as the slides are still in the aforementioned directory structure. The Im3Tools are located in the *astropath\hpfs\Flatfield\flatw\Im3Tools* folder. A single slide can be launched as follows from a command prompt:
```
*\astropath\hpfs\Flatfield\flatw\Im3Tools\doOneSample <base> <FWpath> <SlideID>
```

## 5.8.4. Overview Workflow of Im3Tools
The following is the overview workflow for the flatfield processing itself. This does not include the wrappers that go along with the code for the ```AstroPathPipeline```
Input Parameters: ```<base>```, ```<FWpath>```, ```<SlideID>```
- Extract parameters from the Im3s
  - Input
    - ```<base>```, ```<FW_path>```, ```<SlideID>```
  - Output
    - a full XML extraction of the first *.im3* file (*.full.im3*)
      - *```<SlideID>```.full.xml*
      - Into ```<FW_path>\<SlideID>``` 
    - Extract the ‘shape’ and ‘magnification’
      - *```<SlideID>```.Parameters.xml*
    - Loop through all the *.im3* images and extract SpectralBasisInfo (*.SpectralBasisInfo.xml*)
      - Into ```<FW_path>\<SlideID>```
      - <filename>.SpectralBasisInfo.Exposure.Protocol.DarkCurrentSettings.xml for each *.im3*
    - Loop through all the *.im3* images and extract the binary data (*.DATA.dat*)
      - into a ```<FW_path>\<SlideID>``` 
      - *```<filename>```.Data.dat* for each *.im3* image
- Apply the flat field and warping (*.fw*)
  - Input
    - ```<FW_path>```
    - ```<SlideID>```
  - Output: 
    - in the ```<FW_path>\<SlideID>```
    - *```<filename>```.fw* for each *.im3* image
      - the bitmap image in single column format 
- Re-insert binary data into the *.im3* files
  - Input
    - ```<base>```
    - ```<FW_path>```
    - ```<SlideID>```
  - Output: 
    - into ```<base>\<SlideID>\<flatw_im3_path>```
    - corrected *```<filename>```.im3* for each *.im3* image
- Delete *.DATA.dat* files but keep *.fw* and *.SpectralBasisInfo.xml* and send them to a ```<base>\<SlideID>\<im3>\xml``` directory
