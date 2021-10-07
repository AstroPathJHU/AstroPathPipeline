# 5.7.6.Overview Workflow
The following is the overview workflow for the flatfield processing itself. This does not include the wrappers that go along with the code for the *AstroPath
Pipeline*. 

Input Parameters: ```<base>```, ```<FWpath>```, ```<SlideID>```

- fixM2: Make sure that the *_M2.im3* files are properly resolved. Remove the original, and rename the *_M2* 
  - Input
    - ```<base>```
    -  ```<SlideID>```
- ShredDat: Extract parameters from the Im3s
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
- ApplyCorr: Apply the flat field and warping (*.fw*)
  - Input
    - ```<FW_path>```
    - ```<SlideID>```
  - Output: 
    - in the ```<FW_path>\<SlideID>```
    - *```<filename>```.fw* for each *.im3* image
      - the bitmap image in single column format 
- InjectDat: Re-insert binary data into the *.im3* files
  - Input
    - ```<base>```
    - ```<FW_path>```
    - ```<SlideID>```
  - Output: 
    - into ```<base>\<SlideID>\<flatw_im3_path>```
    - corrected *```<filename>```.im3* for each *.im3* image
- ExtractLayer: Extract the DAPI layer from *.fw* files. We extract the first layer with an *.fw01* extension
  - Input
    - ```<FW_path>```
    - ```<SlideID>```
  - Output
    - in the ```<FW_path>\<SlideID>```
    - *```<filename>```.fw01* for each *.im3* image
      -  the bitmap image in single column format for the first layer (DAPI)
- cleanup: Clean up the path
  - Input 
    - ```<FW_path>```
    - ```<SlideID>``` 
  - Output
    - Delete *.DATA.dat* files but keep *.fw*, *.fw01*, and *.SpectralBasisInfo.Exposure.Protocol.DarkCurrentSettings.xml* and send them to a ```<base>\<SlideID>\<xml_path>``` directory
