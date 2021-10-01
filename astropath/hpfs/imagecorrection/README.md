
# 5.7. Image Correction
## 5.7.1. Description
This workflow serves to create a directory of flat field and warping corrected *.im3* images files for each slide. In addition, this workflow saves the full metadata for the first *.im3* (*.full.im3*) for a slide, the single column bitmap of each corrected *.im3* (*.fw* files), some shape parameters for the first *.im3* (*.Parameters.xml*), as well as relevant image metadata (*.SpectralBasisInfo.Exposure.Protocol.DarkCurrentSettings.xml*) of each im3 image. We assume that the directory format of the input image files is in the ```AstroPathPipeline``` processing file stucture described [here](../../scans/docs/DirectoryOrganization.md#46-directory-organization) and in [5.7.3.1.](docs/ImportantDefinitions.md#5631-flatw-expected-directory-structure). 

There are multiple parts of this module provided which can be run collectively as a workflow through powershell ([5.7.4.](docs/WorkflowInstructions.md#574-workflow-instructions)) or as separate tools ([5.7.5.](docs/AdditionalTools.md#575-additional-tools)). The code is maintained and updated to be used through the workflow and uses outside of this workflow will not be supported by the *AstroPath* group at this time. The workflow loops through all ```Project```s in the [*AstropathCohortsProgress.csv*](../../scans/docs/AstroPathProcessingDirectoryandInitializingProjects.md#451-astropath_processing-directory) and processes slides, if a *flatfield_BatchID_NN.bin* file has been created for that batch (see [5.7.3.1.](docs/ImportantDefinitions.md#5731-flatw-expected-directory-structure)).

*NOTE*: After the intial AstroPath publication, significant changes to the image corrections took place. To keep the code backward compatible both the older version of the code and newer version to apply the image correction are housed here. A desciption of both version is located in [5.7.5](docs/AdditionalTools.md#575-additional-tools). The older version of the code is run via the pipeline, in matlab, by specifying a version number of *0.0.1* in the [*AstroPathConfig.csv*](../../scans/docs/AstroPathProcessingDirectoryandInitializingProjects.md#451-astropath_processing-directory). The older version of the code has significant drawbacks and should only be used for backwards compatibility. 

## 5.7.2. Contents
- [5.7.3. Important Definitions](docs/ImportantDefinitions.md#573-important-definitions)
  - [5.7.3.1. Image Correction Expected Directory Structure](docs/ImportantDefinitions.md#5731-image-correction-expected-directory-structure)
  - [5.7.3.2. Output Formatting](docs/ImportantDefinitions.md#5732-output-formatting)
- [5.7.4. Workflow Instructions](docs/WorkflowInstructions.md#574-workflow-instructions)
- [5.7.5. Additional Tools](docs/AdditionalTools.md#575-additional-tools)
  - [5.7.5.1. Processing One Sample](docs/AdditionalTools.md#5751-processing-one-sample)
  - [5.7.5.2. Directions for Applying the Image Corrections](docs/AdditionalTools.md#5752-directions-for-applying-the-image-corrections)
  - [5.7.5.3. Directions for Running Version *0.0.1*](docs/AdditionalTools.md#5753-directions-for-applying-the-image-corrections-version-001)
- [5.7.6. Overview Workflow of Image Correction Module](docs/OverviewWorkflowofImageCorrectionModule.md#576overview-workflow-of-image-correction-module)

