# 5.8. Flatfield
## 5.8.1. Description
This workflow serves to create a directory of flat field and warping corrected *.im3* images files for each slide. In addition, this workflow saves the full metadata for the first *.im3* (*.full.im3*) for a slide, the single column bitmap of each corrected *.im3* (*.fw* files), some shape parameters for the first *.im3* (*.Parameters.xml*), as well as relevant image metadata (*.SpectralBasisInfo.Exposure.Protocol.DarkCurrentSettings.xml*) of each im3 image. We assume that the directory format of the input image files is in the ```AstroPathPipeline``` processing file stucture described **here** and below in [5.8.2.1.](#5821-flatw-expected-directory-structure "Title"). 

*NOTE*: The warping corrections are only applied in this version of the code to images with the same shape parameters (defined in the *Parameters.xml* extracted from the image metadata) as the JHU Vectra 3.0 machine. A warping model could be estimated and added to the *mkwarp* function in the *runflatw.m* file under *AstroPathPipelinePrivate\astropath\hpfs\Flatfield\flatw\mtools* for a different microscope. That functionality is not included in this processing. 

## 5.8.2. Important Definitions
### 5.8.2.1. Flatw Expected Directory Structure
Since this section of the pipeline can be used with standalone funcationality to apply corrections, we define the directory structure here. A more detailed directory structure for the whole AstroPath Pipeline can be found *here*: 
```
<base>\<SlideID>\<im3_path>\<filename>
```
Example:  “\\bki04\Clinical_Specimen\ AST123456\ im3\Scan1\MSI\M41_1_[34888,4694].im3”
- ```<base>```: “\\bki04\Clinical_Specimen”
- ```<SlideID>```: “AST123456
- ```<im3_path>```: “im3\ScanX\MSI or im3\ScanXX\MSI”
- ```<filename>```: “AST123456_[34888,4694]”
- ```<extension>```:  “.im3” <br>

*NOTE*: 
- The Scan number directory would be the highest scan number in the <SlideID> folder, not always ‘1’ and may be multiple digits
- *BatchID.txt*: This file should contain the batch id that the slides were stained with. The code looks for the batch id as a specifier on the flatfield bin file.
- *flatfield_BatchID_NN.bin*: This is the bitmap data of the average image for a batch. 
  - The code will use the specifier *NN* from the *BatchID.txt* to find the appropriate flatfield file to use. 
  - This file should be kept in a *```<base>```\flatfield* directory (A directory adjacent to the ```<SlideID>``` directory)

## 5.8.2.2. Output Formatting
- For the output images we replace the ```<im3_path>``` with the ```<flatw_im3_path>``` (“im3\flatw”),in the above directory structure but nothing from the file name changes.
  - ```<base>\<SlideID>\<flatw_im3_path>\<filename>```
- For the *.fw* and *.fw01* output data of each image (a single column bitmap for the whole image and just the first image plane) we change the base path to an adjacent directory we call the “FWpath” directory. The data for each given ```<SlideID>``` will be contained in the respective ```<FWpath>\<SlideID>``` directory.
  - ```<FWpath>\<SlideID>\<fw_filename>```
- The full metadata for the first *.im3* (*.full.im3*) for a slide, some shape parameters for the first *.im3* (*.Parameters.xml*), as well as relevant image metadata (*.SpectralBasisInfo.Exposure.Protocol.DarkCurrentSettings.xml*) of each im3 image are saved in a ```<xml_path>``` ("im3\xml"; replaces the ```<im3_path>``` in the base path)
  - ```<base>\<SlideID>\<xml_path>\<filename>```

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
This is the standalone processing tool for the flatfield and image warping applications on a directory of im3 slides. These commands can be used outside of the ```AstroPathPipeline``` as long as the slides are still in the aforementioned [directory structure](#5821 flatw-expected-directory-structure "Title"). All output described [here](#5822-output-formatting "TItle") result from this tool the rest of the code included in *flatfield* simply allows bulk processing of slides.

The Im3Tools are located in the [*astropath\hpfs\Flatfield\flatw\Im3Tools* folder](flatw). A single slide can be launched as follows from a command prompt:
```
*\astropath\hpfs\Flatfield\flatw\Im3Tools\doOneSample <base> <FWpath> <SlideID>
```

## 5.8.3.4. ConvertIm3
The ConvertIM3 application reads and writes AKOYA IM3 ("image cube") files.

ConvertIM3 uses a simple set of C# classes to represent the contents of an IM3 file. Unfortunately, the IM3 format is undocumented. There are no software tools available that manage IM3 files directly.  It is more practical to transform IM3 into a well-known format that can be handled by generally-available software.  Since the IM3 format is essentially a hierarchy of self-described groups of data, a straightforward way to work with IM3 is to convert it to and from XML.

ConvertIM3 can carry out the following operations:
- convert IM3 to XML
- convert XML to IM3
- extract one IM3 data field (that is, one XML element) from IM3-formatted data
- replace the contents of one IM3 data field (that is, one XML element) in IM3-formatted data

Command-line syntax:

 ```
 ConvertIM3 <input_file_specification> IM3|XML|DAT
            [/o <output_directory_path>]
            [/t <max_XML_data_length>] 
            [/x <XPath_expression>]
            [/i <injected_data_file_specification>]
``` 

Extended directions can be found [here](flatw/Im3Tools/ConvertIM3Usage.txt).

## 5.8.3.4. ConvertIm3Path & ConvertIm3Cohort
ConvertIm3Path & ConvertIm3Cohort are soft wrappers written in powershell for the executable to run *ConvertIm3* on the ```<SlideID>``` level for specific output desired by the *AstroPathPipeline*, with optional inputs and exports. Both codes are dependent on the images being in the aforementioned [directory structure](#5821 flatw-expected-directory-structure "Title"). ConvertIm3Cohort.ps1 is actually just a wrapper for ConvertIm3Path.ps1 which runs through all specimens in a directory, the cohort location is hard coded at present and should be modified in the code. 
 
Usage: 
- To "shred" a directory of im3s use:
  - ```ConvertIm3Path -<base> -<FWpath> -<SlideID> -s [-a -d -xml]```
  - Optional arguements:
	  - ```-d```: only extract the binary bitmap for each image into the output directory
	  - ```-xml```: extract the xml information only for each image, xml information includes:
		  - one <sample>.Parameters.xml: sample location, shape, and scale
		  - one <sample>.Full.xml: the full xml of an im3 without the bitmap
		  - a .SpectralBasisInfo.Exposure.xml for each image containing the exposure times of the image
- To "inject" a directory of .dat binary blobs for each image back into the directory of im3s use:
  - ```ConvertIm3Path -<base> -<FWpath> -<SlideID> -i```
  - Exports the new '.im3s' into the ```<flatw_im3_path>``` directory

## 5.8.4. Overview Workflow of Im3Tools
The following is the overview workflow for the flatfield processing itself (Im3Tools). This does not include the wrappers that go along with the code for the ```AstroPathPipeline```
Input Parameters: ```<base>```, ```<FWpath>```, ```<SlideID>```
- Make sure that the *_M2.im3* files are properly resolved. Remove the original, and rename the *_M2* 
  - Input
    - ```<base>```
    -  ```<SlideID>```
- Extract parameters from the Im3s
  - Input
    - ```<base>```
    - ```<FW_path>```
    - ```<SlideID>```
  - Output
    - a full XML extraction of the first *.im3* file (*.full.im3*)
      - *```<SlideID>```.full.xml*
      - Into ```<FW_path>\<SlideID>``` 
    - Extract the ‘shape’ and ‘magnification’
      - *```<SlideID>```.Parameters.xml*
    - Loop through all the *.im3* images and extract SpectralBasisInfo.Exposure.Protocol.DarkCurrentSettings.xml (*.SpectralBasisInfo.Exposure.Protocol.DarkCurrentSettings.xml*)
      - Into ```<FW_path>\<SlideID>```
      - *```<filename>```.SpectralBasisInfo.Exposure.Protocol.DarkCurrentSettings.xml* for each *.im3*
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
- Extract the DAPI layer from *.fw* files. We extract the first layer with an *.fw01* extension
  - Input
    - ```<FW_path>```
    - ```<SlideID>```
  - Output
    - in the ```<FW_path>\<SlideID>```
    - *```<filename>```.fw01* for each *.im3* image
      -  the bitmap image in single column format for the first layer (DAPI)
- Clean up the path
  - Input 
    - ```<FW_path>```
    - ```<SlideID>``` 
  - Output
    - Delete *.DATA.dat* files but keep *.fw*, *.fw01*, and *.SpectralBasisInfo.Exposure.Protocol.DarkCurrentSettings.xml* and send them to a ```<base>\<SlideID>\<xml_path>``` directory
