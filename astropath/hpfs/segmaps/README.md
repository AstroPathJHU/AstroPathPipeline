# 5.9. Seg Maps
## 5.9.1. Description
As part of the multipass segmentation and classification system, described in the *MaSS* documentation and elsewhere in the *AstroPathPipeline* documentation, multiple segmentation algorithms can be used on a single panel for cells of different sizes. The cell tables are merged as part of the *MaSS* protocol here we use the information from the *cleaned_phenotype_tables* to select the resultant cell geometries from the *binary_seg_maps*. The geometeries are then saved in separate layers with the *component_data* into a *component_data_w_seg* file in the *Component_Tiffs* folder of the *AstroPathPipeline* directory structure. This module should be run after the classification and segmentation of slides have been verified complete according to the protocols laid out in the QA QC step of the *inform_processing* module. The module is set to run once across all cohorts after being launched, it checks the following conidtions before running on a set of slides:
- That the *cleaned_phenotype_table* files exist
- That the *component_data_w_seg* files exist 
- If they do exist, that they were created after the last *cleaned_phenotype_table* file was created

## 5.9.2. Instructions
The code should be launched through matlab. To start download the repository to a working location. Next, open a new session of matlab and add the ```AstroPathPipline``` to the matlab path. Then use the following to launch:   
``` segmaps(<Mpath>) ```
- ```<Mpath>[string]```: the full path to the directory containing the ***AstropathCohortsProgress.csv*** file
   - description of this file can be found [here](../../scans/docs/AstroPathProcessingDirectoryandInitializingProjects.md#451-astropath_processing-directory "Title")

## 5.9.3. Segmenation Map Structure Definition
The segmentation maps, *component_data_w_seg*, start with the component tiff layers from the *component_data* files, the number of layers correspond to the layers added for unmixing in inForm. For a standard 7-color slide unmixed by inForm this would be:
- DAPI
- Opal 520
- Opal 540
- Opal 570
- Opal 620
- Opal 650
- Opal 690
- Autofluorescence

The next layers correspond to the segmentation label images from inForm. The layers start with the tissue segmentation, followed by the nuclear segmentation for each segmentation type and the membrane segmentation for each segmentation type. The *segmentation types* are defined in the ```<SegmentationStatus>``` column of the *MergeConfig_NN.xlsx* files. The layers in the segmentation maps are added in the same numeric order of the ```<SegmentationStatus>``` column, such that the segmentation map layers for a panel with two segmentation types and a 7-color panel would be as follows:

- 9: Tissue Segmentation
- 10: Nuclear: ```<SegmentationStatus>``` 1
- 11: Nuclear: ```<SegmentationStatus>``` 2
- 12: Membrane: ```<SegmentationStatus>``` 1
- 13: Membrane: ```<SegmentationStatus>``` 2

The values in the segmentation layers are set up as a label matrix with the values corresponding to the ```<CellID>``` column in the *cleaned_phenotyped_tables.csv* files. For example, in the nuclear layer all values with 1 correspond to ```<CellID>```: 1; all values with 2 correspond to ```<CellID>```: 2; and so on.
