# 5.11. Seg Maps
## 5.11.1. Description
As part of the multipass segmentation and classification system, described in the *MaSS* documentation and elsewhere in the *AstroPathPipeline* documentation, multiple segmentation algorithms can be used on a single panel for cells of different sizes. The cell tables are merged as part of the *MaSS* protocol here we use the information from the *cleaned_phenotype_tables* to select the resultant cell geometries from the *binary_seg_maps*. The geometeries are then saved in separate layers with the *component_data* into a *component_data_w_seg* file in the *Component_Tiffs* folder of the *AstroPathPipeline* directory structure. This module should be run after the classification and segmentation of slides have been verified complete according to the protocols laid out in the QA QC step of the *inform_processing* module. 

## 5.11.2. Instructions

## 5.11.3. Segmenation Map Structure Definition
