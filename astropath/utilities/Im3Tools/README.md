## Im3tools
This is the standalone processing tool for the flatfield and image warping applications on a directory of im3 slides. These commands can be used outside of the *AstroPath Pipeline* as long as the slides are still in the aforementioned [directory structure](../../hpfs/imagecorrection/docs/ImportantDefinitions.md#5731-image-correction-expected-directory-structure "Title"). All output described [here](../../hpfs/imagecorrection/docs/ImportantDefinitions.md#5732-output-formatting "Title") result from this tool the rest of the code included in *flatfield* simply allows bulk processing of slides.

A single slide can be launched as follows from a command prompt:

```
*AstroPathPipline\astropath\hpfs\flatw\flatw_matlab\Im3Tools\doOneSample <base> <FWpath> <SlideID>
```
- ```<base>```: *\\\\bki04\\Clinical_Specimen* (also refered to as the ```<Dpath>\<Dname>```
- ```<FWpath>```: This is the full path for the single column flat field and warping image (.fw) as well as the exposure time data for each image (.SpectralBasisInfo.xml). 
  - This path should preferably located on a different drive from the main path to improve pipeline performance. 
  - E.g. “bki03\flatw_7”
- ```<SlideID>```: *AST123456*

## 5.8.5.2. ConvertIm3
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

Extended directions can be found [here](./ConvertIM3Usage.txt).

## 5.8.5.3. ConvertIm3Path & ConvertIm3Cohort
ConvertIm3Path & ConvertIm3Cohort are soft wrappers written in powershell for the executable to run *ConvertIm3* on the ```<SlideID>``` level for specific output desired by the *AstroPathPipeline*, with optional inputs and exports. Both codes are dependent on the images being in the aforementioned [directory structure]../../hpfs/imagecorrection/docs/ImportantDefinitions.md#5731-image-correction-expected-directory-structure "Title"). ConvertIm3Cohort.ps1 is actually just a wrapper for ConvertIm3Path.ps1 which runs through all specimens in a directory, the cohort location is hard coded at present and should be modified in the code. 
 
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
