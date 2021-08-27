
# 5.8. Image Correction
## 5.8.1. Description
This workflow serves to create a directory of flat field and warping corrected *.im3* images files for each slide. In addition, this workflow saves the full metadata for the first *.im3* (*.full.im3*) for a slide, the single column bitmap of each corrected *.im3* (*.fw* files), some shape parameters for the first *.im3* (*.Parameters.xml*), as well as relevant image metadata (*.SpectralBasisInfo.Exposure.Protocol.DarkCurrentSettings.xml*) of each im3 image. We assume that the directory format of the input image files is in the ```AstroPathPipeline``` processing file stucture described [here](../../scans/docs/DirectoryOrganization.md#46-directory-organization) and in [5.8.3.1.](docs/ImportantDefinitions.md#5831-flatw-expected-directory-structure). 

There are multiple parts of this module provided which can be run collectively as a workflow through powershell ([5.8.4.](docs/WorkflowInstructions.md#584-workflow-instructions)) or as separate tools ([5.8.5.](docs/AdditionalTools.md#585-additional-tools)). The code is maintained and updated to be used through the workflow and uses outside of this workflow will not be supported by the *AstroPath* group at this time. The workflow loops through all ```Project```s in the [*AstropathCohortsProgress.csv*](../../scans/docs/AstroPathProcessingDirectoryandInitializingProjects.md#451-astropath_processing-directory) and processes slides, if a *flatfield_BatchID_NN.bin* file has been created for that batch (see [5.8.3.1.](docs/ImportantDefinitions.md#5831-flatw-expected-directory-structure)).

*NOTE*: After the intial AstroPath publication, significant changes to the image corrections took place. To keep the code backward compatible both the older version of the code and newer version to apply the image correction are housed here. The older version of the code is run in matlab by specifying a version number of *0.0.1* in the [*AstroPathConfig.csv*](../../scans/docs/AstroPathProcessingDirectoryandInitializingProjects.md#451-astropath_processing-directory). The older version of the code has significant drawbacks and should only be used for backwards compatibility. 

## 5.8.2. Contents
- [5.8.3. Important Definitions](docs/ImportantDefinitions.md#583-important-definitions)
  - [5.8.3.1. Image Correction Expected Directory Structure](docs/ImportantDefinitions.md#5831-image-correction-expected-directory-structure)
  - [5.8.3.2. Output Formatting](docs/ImportantDefinitions.md#5832-output-formatting)
- [5.8.4. Workflow Instructions](docs/WorkflowInstructions.md#584-workflow-instructions)
- [5.8.5. Additional Tools]((docs/AdditionalTools.md#585-additional-tools)
  - [5.8.5.1. Directions for Applying the Image Corrections]()
  - [5.8.5.2. Directions for Running Version *0.0.1*]()
- [5.8.6. Overview Workflow of Image Correction Module](docs/OverviewWorkflowofIm3Tools.md#586-overview-workflow-of-im3tools)

