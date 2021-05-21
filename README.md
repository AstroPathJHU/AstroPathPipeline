# <div align="center"> AstroPath Pipeline </div>
#### <div align="center"> The *AstroPath Pipeline* was developed to process whole slide multiplex immunofluorescence data from microscope to database </div>
#### <div align="center">Correspondence to: bgreen42@jhu.edu</div>

## 1. Description
The *AstroPath Pipeline* was designed to automate the processing of whole slide multiplex immunoflourecence histopathology image data, taken by Akoya Biosciences’ Vectra imaging platform, from the microscope to database. The automated process begins after whole slide scans have been captured by the microscope and manually verified complete. Code is divided into three main stages; defined as ```hpf```, ```slide```, and ```sample``` level processing. In the ```hpf``` (or high powered field) processing stage, images are reorganized, corrected for camera\ imaging effects, and segmented\ phenotyped. Here images are mainly processed individually. In the next processing stage, aptly named ```slide``` the data is stiched together into a whole slide and the slides are annotated by a pathologist. Finally, slides across a cohort are corrected for batch to batch variation and loaded into a database. Here the image, cell, and annotation data of each whole slide image is linked its clinical information thus providing a completed ```sample```. Code for each stage is organized into its own folder under ```astropath```, with each folder containing a particular set of modules. Each module is organized separately in subfolders and described with linked documenation. An overview of the current pipeline can be seen [here](https://github.com/AstropathJHU/AstroPathPipeline/blob/main/AstroPathPipeline.pdf).

## 2. Getting Started
## 2.1. Prerequisites
- [Windows 10](https://www.microsoft.com/en-us/software-download/windows10)
- [Python 3.6/3.7](https://www.python.org/)
- [MATLAB 2020a](https://www.mathworks.com/products/matlab.html)

## 2.2. Instructions
### 2.2.1. Python Instructions
To install the code, first check out the repository, enter its directory, and run
```bash
pip install .
```
If you want to continue developing the code after installing, run instead
```bash
pip install --editable .
```

Once the code is installed using either of those lines, you can run
```python
import astropath
```
from any directory.

### 2.2.2. MATLAB Instructions
Check out\ download the github repository. In MATLAB add the entire *AstroPath Pipeline* to the MATLAB path. The *AstroPath Pipeline* commands should then be available in MATLAB. 

*NOTE*: For specific Python, MATLAB, cmd, or PowerShell commands of a particular module check the module or workflow instructions.

## 3. Contents
- [1. Description](#1-description "Title")
- [2. Instructions](#2-instructions "Title")
- [3. Contents](#3-contents "Title")
- [4. Scanning Slides (scans)](astropath/scans#4-scans "Title")
   - [4.1. Description](astropath/scans#41-description "Title")
   - [4.2. Contents](astropath/scans#42-contents "Title")
   - [4.3. Definitions](astropath/scans/docs/Definitions.md/#43-definitions)
     - [4.3.1. Indentification Definitions](astropath/scans/docs/Definitions.md/#431-identification-definitions)
     - [4.3.2. Path Definitions](astropath/scans/docs/Definitions.md/#432-path-definitions)
   - [4.4. Scanning Instructions](astropath/scans/docs/ScanningInstructionsIntro.md)
     - [4.4.1. Contents](astropath/scans/docs/ScanningInstructionsIntro.md/#441-contents) 
     - [4.4.2. Specimen Table](astropath/scans/docs/scanning/SpecimenTable.md)
     - [4.4.3. SampleNames (Patient # or M Numbers)](astropath/scans/docs/scanning/SampleNames.md)
     - [4.4.4. Control TMA Conventions](astropath/scans/docs/scanning/ControlTMAConventions.md)
     - [4.4.5. Whole Slide Scanning Introduction](astropath/scans/docs/scanning/WholeSlideScanning.md)
       - [4.4.5.1. Contents](astropath/scans/docs/scanning/WholeSlideScanning.md/#4451-contents)
       - [4.4.5.2. Preparing 20% Overlap](astropath/scans/docs/scanning/Preparing20Overlap.md)
       - [4.4.5.3. Scan the *Control TMA*](astropath/scans/docs/scanning/ScantheControlTMA.md)
       - [4.4.5.4. Scan the Whole Slide Specimens](astropath/scans/docs/scanning/ScantheWholeSlideSpecimens.md)
       - [4.4.5.5. Creating a Whole Slide Scanning Plan](astropath/scans/docs/scanning/CreatingaWholeSlideScanningPlan.md)
       - [4.4.5.6. Important Scanning Notes](astropath/scans/docs/scanning/ImportantScanningNotes.md)
     - [4.4.6. BatchIDs](astropath/scans/docs/scanning/BatchIDs.md)
     - [4.4.7. Batch Tables](astropath/scans/docs/scanning/BatchTables.md)
     - [4.4.8. Merge Configuration Tables](astropath/scans/docs/scanning/MergeConfigTables.md)
  - [4.5. AstroPath Processing Directory and Initalizing Projects](astropath/scans/docs/AstroPathProcessingDirectoryandInitializingProjects.md)
    - [4.5.1. AstroPath Processing Directory](astropath/scans/docs/AstroPathProcessingDirectoryandInitializingProjects.md/#451-astropath-processing-directory)
    - [4.5.2. Initializing Projects](astropath/scans/docs/AstroPathProcessingDirectoryandInitializingProjects.md/#452-initializing-projects)
  - [4.6. Directory Organization](astropath/scans/docs/DirectoryOrganization.md)
    - [4.6.1. Directory Subfolders](astropath/scans/docs/DirectoryOrganization.md#461-directory-subfolders)
    - [4.6.2. SlideID Subfolders](astropath/scans/docs/DirectoryOrganization.md#462-slideid-subfolders)
- [5. HPF Processing (hpfs)](astropath/hpfs#5-hpf-processing-hpfs "Title")
  - [5.1. Description](astropath/hpfs#51-description "Title")
  - [5.2. Contents](astropath/hpfs#52-contents "Title")
  - [5.3. Instructions](astropath/hpfs/docs/Instructions.md)
    - [5.3.1. Contents](astropath/hpfs/docs/Instructions.md#531-contents)
    - [5.3.2. Typical User Instructions](astropath/hpfs/docs/TypicalUserInstructions.md#532-typical-user-instructions)
    - [5.3.3 Launching Code Instructions](astropath/hpfs/docs/LaunchingCodeInstructions.md#533-launching-code-instructions)
  - [5.4. Workflow Overview](astropath/hpfs/docs/WorkflowOverview)
  - [5.5. AstroIDGen](astropath/hpfs/AstroidGen#55-astroid-generation "Title")
    - [5.5.1. Description](astropath/hpfs/AstroidGen#551-description "Title")
    - [5.5.2. Important Definitions](astropath/hpfs/AstroidGen#552-important-definitions "Title")
    - [5.5.3. Workflow](astropath/hpfs/AstroidGen#553-workflow "Title")
  - [5.6. Transfer Daemon](astropath/hpfs/TransferDaemon#56-transfer-daemon "Title")
    - [5.6.1. Description](astropath/hpfs/TransferDaemon#561-description "Title")
    - [5.6.2. Important Definitions](astropath/hpfs/TransferDaemon#562-important-definitions "Title")
    - [5.6.3. Code Input](astropath/hpfs/TransferDaemon#563-code-input "Title")
    - [5.6.4. Workflow](astropath/hpfs/TransferDaemon#564-workflow "Title")
      - [5.6.4.1. Initial Transfer](astropath/hpfs/TransferDaemon#5641-initial-transfer "Title")
      - [5.6.4.2. MD5 Check](astropath/hpfs/TransferDaemon#5642-md5-check "Title")
      - [5.6.4.3. Compression Into Backup](astropath/hpfs/TransferDaemon#5643-compression-into-backup "Title")
      - [5.6.4.4. Source File Handling](astropath/hpfs/TransferDaemon#5644-source-file-handling "Title")
    - [5.6.5. Notes](astropath/hpfs/TransferDaemon#565-notes "Title") 
  - [5.7. Meanimages](astropath/hpfs/meanimages#57-meanimages "Title")
    - [5.7.1. Description](astropath/hpfs/meanimages#571-description "Title")
    - [5.7.2. Important Definitions](astropath/hpfs/meanimages#572-important-definitions "Title")
    - [5.7.3. Instructions](astropath/hpfs/meanimages#573-instructions "Title")
    - [5.7.4. Workflow](astropath/hpfs/meanimages#574-workflow "Title")
      - [5.7.4.1. Checking for Tasks](astropath/hpfs/meanimages#5741-checking-for-tasks "Title")
      - [5.7.4.2. Shred Im3s](astropath/hpfs/meanimages#5742-shred-im3s "Title")
      - [5.7.4.3. raw2mean](astropath/hpfs/meanimages#5743-raw2mean "Title")
  - [5.8. Flatw](astropath/hpfs/flatw#58-flatw "Title")
    - [5.8.1. Description](astropath/hpfs/flatw#581-description "Title")
    - [5.8.2. Contents](astropath/hpfs/flatw#582-contents "Title")
    - [5.8.3. Important Definitions](astropath/hpfs/flatw/docs/ImportantDefinitions.md#583-important-definitions)
      - [5.8.3.1. Flatw Expected Directory Structure](astropath/hpfs/flatw/docs/ImportantDefinitions.md#5831-flatw-expected-directory-structure)
      - [5.8.3.2. Output Formatting](astropath/hpfs/flatw/docs/ImportantDefinitions.md#5832-output-formatting)
    - [5.8.4. Workflow Instructions](astropath/hpfs/flatw/docs/WorkflowInstructions.md#584-workflow-instructions)
      - [5.8.4.1. flatw queue](astropath/hpfs/flatw/docs/WorkflowInstructions.md#5841-flatw_queue)
      - [5.8.4.2. flatw worker](astropath/hpfs/flatw/docs/WorkflowInstructions.md#5842-flatw_worker)
    - [5.8.5. Additional Tools](astropath/hpfs/flatw/docs/AdditionalTools.md#585-additional-tools)
      - [5.8.5.1. Im3Tools](astropath/hpfs/flatw/docs/AdditionalTools.md#5851-im3tools)
      - [5.8.5.2. ConvertIm3](astropath/hpfs/flatw/docs/AdditionalTools.md#5852-convertim3)
      - [5.8.5.3. ConvertIm3Path & ConvertIm3Cohort](astropath/hpfs/flatw/docs/AdditionalTools.md#5853-convertim3path--convertim3cohort)
    - [5.8.6. Overview Workflow of Im3Tools](astropath/hpfs/flatw/docs/OverviewWorkflowofIm3Tools.md#586-overview-workflow-of-im3tools)
  - [5.9. Mergeloop](astropath/hpfs/mergeloop#59-mergeloop "Title")
    - [5.9.1. Description](astropath/hpfs/mergeloop#591-description)
    - [5.9.2. Important Definitions](astropath/hpfs/mergeloop#592-important-definitions)
    - [5.9.3. Instructions](astropath/hpfs/mergeloop#593-instructions)
    - [5.9.4. Workflow](astropath/hpfs/mergeloop#594-workflow)
    - [5.9.5. Merge a Single Sample (MaSS)](astropath/hpfs/mergeloop/MaSS#merge-a-single-sample-mass)
    - [5.9.6. Create Image QA QC utility](astropath/hpfs/mergeloop/MaSS#section-8create-image-qa-qc-utility)
  - [5.10. Inform_processing](astropath/hpfs/inform_processing#510-inform_processing "Title")
    - [5.10.1. Description](astropath/hpfs/inform_processing/README.md#5101-description)
    - [5.10.2. Contents](astropath/hpfs/inform_processing/README.md#5102-contents)
    - [5.10.3. inForm® Multipass Phenotyping](astropath/hpfs/inform_processing/docs/inFormMultipassPhenotype.md#5103-inform-multipass-phenotype "Title")
      - [5.10.3.1. Description](astropath/hpfs/inform_processing/docs/inFormMultipassPhenotype.md#51031-description "Title")
      - [5.10.3.2. Instructions](astropath/hpfs/inform_processing/docs/inFormMultipassPhenotype.md#51032-instructions "Title")
        - [5.10.3.2.1. Getting Started](astropath/hpfs/inform_processing/docs/inFormMultipassPhenotype.md#510321-getting-started "Title")
        - [5.10.3.2.2. Core Icons to Remember](astropath/hpfs/inform_processing/docs/inFormMultipassPhenotype.md#510322-core-icons-to-remember "Title")
        - [5.10.3.2.3. Segment Tissue](astropath/hpfs/inform_processing/docs/inFormMultipassPhenotype.md#510323-segment-tissue "Title")
        - [5.10.3.2.4. Adaptive Cell Segmentation](astropath/hpfs/inform_processing/docs/inFormMultipassPhenotype.md#510324-adaptive-cell-segmentation "Title")
        - [5.10.3.2.5. Phenotyping](astropath/hpfs/inform_processing/docs/inFormMultipassPhenotype.md#510325-phenotyping "Title")
        - [5.10.3.2.6. Export](astropath/hpfs/inform_processing/docs/inFormMultipassPhenotype.md#510326-export "Title")
    - [5.10.4. Saving Project for the inForm® JHU Processing Farm](astropath/hpfs/inform_processing/docs/SavingProjectsfortheinFormJHUProcessingFarm.md#5104-saving-projects-for-the-inform-jhu-processing-farm "Title")
      - [5.10.4.1. Description](astropath/hpfs/inform_processing/docs/SavingProjectsfortheinFormJHUProcessingFarm.md#51041-description "Title")
      - [5.10.4.2. Instructions](astropath/hpfs/inform_processing/docs/SavingProjectsfortheinFormJHUProcessingFarm.md#51042-instructions "Title")
    - [5.10.5. Adding Slides to the inForm Queue](astropath/hpfs/astropath/hpfs/inform_processing/docs/AddingSlidestotheinFormQueue.md#5105-adding-slides-to-the-inform-queue)
    - [5.10.6. Evaluating inForm® Phenotype QC Output for the *AstroPath Pipeline*](astropath/hpfs/inform_processing/docs/EvaluatinginFormPhenotypeQCOutputfortheAstroPathPipeline.md#5106-evaluating-inform-phenotype-qc-output-for-the-astropath-pipeline)
    - [5.10.7. Processing inForm® Tasks](astropath/hpfs/inform_processing/docs/ProcessinginFormTasks.md#5107-proccessing-inform-tasks)
  - [5.11. Segmaps](astropath/hpfs/segmaps#511-segmaps "Title")
    - [5.11.1. Description](segmaps#5111-description)
    - [5.11.2. Instructions](segmaps#5112-instructions)
    - [5.11.3. Segmentation Map Structure Definition](segmaps#5113-segmenation-map-structure-definition)
  - [5.12. Transferanno](astropath/hpfs/transferanno#512-transferanno "Title")
    - [5.12.1. Description](astropath/hpfs/transferanno#5121-description)
    - [5.12.2. Create HALO Annoations for the *AstroPath Pipeline*](astropath/hpfs/transferanno/README.md#5122-creating-halo-annotations-for-the-astropath-pipeline)
      - [5.12.2.1. Prepare the HALO Project](astropath/hpfs/transferanno/README.md#51221-prepare-the-halo-project)
      - [5.12.2.2. Create and Edit Annotation Layers](astropath/hpfs/transferanno/README.md#51222-create-and-edit-annotation-layers)
      - [5.12.2.3. Annotation Layer Data Dictionary](astropath/hpfs/transferanno/README.md#51223-annotation-layer-data-dictionary)
    - [5.12.3. Exporting Annotations](astropath/hpfs/transferanno/README.md#5123-exporting-annotations)
    - [5.12.4. Transfer Annotations to the BKI Server](astropath/hpfs/transferanno/README.md#5124-transfer-annotations-to-the-bki-server)
      - [5.12.4.1. Description](astropath/hpfs/transferanno/README.md#51241-description)
      - [5.12.4.2. Instructions](astropath/hpfs/transferanno/README.md#51242-instructions)
 - [6. Slide Processing](astropath/slides/#6-slide-processing)
   - [6.1 Description](astropath/slides/#61-description)
   - [6.2 Prepdb](astropath/slides/prepdb/#62-prepdb)
   - [6.3 Align](astropath/slides/align/#63-align)
   - [6.4 Annotations]
   - [6.5 Cell Geometries]
 - [7. Sample Processing]
   - [7.1 Description]
   - [7.2 Control TMAs]
   - [7.3 Calibration]
   - [7.4 Upload to Database]

## Credits
#### <div align="center">Created by: Benjamin Green<sup>1</sup>, Jeffrey S. Roskes<sup>4</sup>, Margaret Eminizer<sup>4</sup>, Richard Wilton<sup>4</sup>, Sigfredo Soto-Diaz<sup>2</sup>, Sneha Berry<sup>2</sup>, Elizabeth Engle<sup>2</sup>, Nicolas Giraldo<sup>3</sup>, Peter Nguyen<sup>2</sup>, Tricia Cottrell<sup>3</sup>, Janis Taube<sup>1,2,3</sup>, and Alex Szalay<sup>4</sup></div>

 <div align="center">Departments of <sup>1</sup>Dermatology, <sup>2</sup>Oncology, <sup>3</sup>Pathology at Johns Hopkins University SOM, the Mark Center for Advanced Genomics and Imaging, the Sidney Kimmel Comprehensive Cancer Center, and the Bloomberg~Kimmel Institute for Cancer Immunotherapy at Johns Hopkins, Baltimore, MD, USA</div>
 <div align="center"> Departments of <sup>4</sup>Astronomy and Physics at Johns Hopkins University and IDIES, Baltimore, MD, USA</div> 
 <br>
 
Individual Contributions: **Benjamin Green**: Conceptualization, Methodology, Software, Writing – Original Draft, Visualization **Jeffrey S. Roskes**: Conceptualization, Methodology, Software, Writing – Original Draft **Margaret Eminizer**: Conceptualization, Methodology, Software, Writing – Original Draft, Visualization **Richard Wilton**: Methodology, Software **Sigfredo Soto-Diaz**: Methodology, Software, Writing – Original Draft **Sneha Berry**: Conceptualization, Validation, Visualization **Liz Engle**: Conceptualization, Resources, Validation **Nicolas Giraldo-Castillo**: Conceptualization **Peter Nguyen**: Conceptualization, Methodology **Tricia Cottrell**: Conceptualization, Validation, Writing – Review & Editing **Janis Taube**: Conceptualization, Resources, Supervision **Alex Szalay**: Conceptualization, Methodology, Validation, Software, Supervision
