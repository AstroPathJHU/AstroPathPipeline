# 5.8.3. Important Definitions
## 5.8.3.1. Flatw Expected Directory Structure
Since this section of the pipeline can be used with standalone funcationality to apply corrections, we define the directory structure here. A more detailed directory structure for the whole AstroPath Pipeline can be found [here](../../../scans/docs/DirectoryOrganization.md#46-directory-organization): 
```
<base>\<SlideID>\<im3_path>\<filename>
```
Example:  *\\\\bki04\\Clinical_Specimen\\AST123456\\im3\\Scan1\\MSI\\M41\_1\_\[34888,4694\].im3*
- ```<base>```: *\\\\bki04\\Clinical_Specimen* (also refered to as the ```<Dpath>\<Dname>```
- ```<SlideID>```: *AST123456*
- ```<im3_path>```: *im3\\ScanX\\MSI* or *im3\\ScanXX\\MSI*
- ```<filename>```: *AST123456\_\[34888,4694\]*
- ```<extension>```:  *.im3* <br>

*NOTE*: 
- The Scan number directory would be the highest scan number in the ```<SlideID>``` folder, not always ‘1’ and may be multiple digits
- *BatchID.txt*: This file should contain the batch id that the slides were stained with. The code looks for the batch id as a specifier on the flatfield bin file, additional documentation found [here](../../../scans/docs/scanning/BatchIDs.md#446-batchids)
- *flatfield_BatchID_NN.bin*: This is the bitmap data of the average image for a batch. 
  - The code will use the specifier *NN* from the *BatchID.txt* to find the appropriate flatfield file to use. 
  - This file should be kept in a *```<base>```\\flatfield* directory (A directory adjacent to the ```<SlideID>``` directory) for proper coding processing

## 5.8.3.2. Output Formatting
- For the output images we replace the ```<im3_path>``` with the ```<flatw_im3_path>``` (*im3\\flatw*), in the above directory structure but nothing from the file name changes.
  - ```<base>\<SlideID>\<flatw_im3_path>\<filename>```
- For the *.fw* and *.fw01* output data of each image (a single column bitmap for the whole image and just the first image plane) we change the base path to an adjacent directory we call the “FWpath” directory. The data for each given ```<SlideID>``` will be contained in the respective ```<FWpath>\<SlideID>``` directory.
  - ```<FWpath>\<SlideID>\<fw_filename>```
- The full metadata for the first *.im3* (*.full.im3*) for a slide, some shape parameters for the first *.im3* (*.Parameters.xml*), as well as relevant image metadata (*.SpectralBasisInfo.Exposure.Protocol.DarkCurrentSettings.xml*) of each im3 image are saved in a ```<xml_path>``` (*im3\xml*; replaces the ```<im3_path>``` in the base path)
  - ```<base>\<SlideID>\<xml_path>\<filename>```
