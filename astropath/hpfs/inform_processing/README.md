# 5.10. inForm Processing
## 5.10.1. Description
After the images are corrected by the ```flatfield``` module and the ```flatw_path``` a are created([5.8.](\..\flatfield#58-flatfield "Title"), the inForm Cell Analysis® (Akoya Biosciences®) classification algorithms can be developed. The *AstroPathPipeline* uses the so-called 'multipass' method for cell classification. This is a novel method of using inForm Cell Analysis®, developed to improve sensitivity and specificty of classication algorithms (described in detail here [5.10.3.](docs/inFormCellAnalysisMultipassPhenotype.md#5103-inform-cell-analysis-multipass-phenotype "Title")). 

Once algorithms are created, each algorithm should be processed through inForm Cell Analysis®. This process is onerous for a few reasons. First, for whole slide scans of 100-3000 images, inForm Cell Analysis® could take on the order of 6-7 hours. This on its own is a significant amount of time. However, since we use separate algorithms for each antibody, processing a 7-color panel on one large specimen could take over 24 hours. Second, we change the path locations and path names of the slides as part of the ```flatfield``` processing. Because of these changes, typical methods for batch processing images at the slide or directory levels no longer function as expected in inForm Cell Analysis®. Instead to process al images from a slide, we must navigate to the ```flatw_path``` directories and manually add all images. In order to automate this processing, JHU installed a set of inForm licenses onto virtual machines located on a server. Code was then written in the programming language *AutoIt* to simulate mouse clicks and run algorithms in batch mode for slides in the *AstroPath Pipeline* directory structure. Afterward a queued system was developed to process algorithms at scale using this utility. 

Each of these steps (creating algorithms, setting up VMs, adding algorithms to the queue, and running the code) is detailed in this section. Because there are so many steps and in order to limit the length of this page, documentation was written into different pages and linked here by a table of contents.

## 5.10.2. Contents
- [5.10.3. inForm Cell Analysis® Multipass Phenotyping](docs/inFormCellAnalysisMultipassPhenotype.md#5103-inform-cell-analysis-multipass-phenotype "Title")
  - [5.10.3.1. Description](docs/inFormCellAnalysisMultipassPhenotype.md#51031-description "Title")
  - [5.10.3.2. Instructions](docs/inFormCellAnalysisMultipassPhenotype.md#51032-instructions "Title")
    - [5.10.3.2.1. Getting Started](docs/inFormCellAnalysisMultipassPhenotype.md#510321-getting-started "Title")
    - [5.10.3.2.2. Core Icons to Remember](docs/inFormCellAnalysisMultipassPhenotype.md#510322-core-icons-to-remember "Title")
    - [5.10.3.2.3. Segment Tissue](docs/inFormCellAnalysisMultipassPhenotype.md#510323-segment-tissue "Title")
    - [5.10.3.2.4. Adaptive Cell Segmentation](docs/inFormCellAnalysisMultipassPhenotype.md#510324-adaptive-cell-segmentation "Title")
    - [5.10.3.2.5. Phenotyping](docs/inFormCellAnalysisMultipassPhenotype.md#510325-phenotyping "Title")
    - [5.10.3.2.6. Export](docs/inFormCellAnalysisMultipassPhenotype.md#510326-export "Title")
- [5.10.4. Saving Project for the inForm Cell Analysis® JHU Processing Farm](docs/SavingProjectsfortheinFormCellAnalysisJHUProcessingFarm.md#5104-saving-projects-for-the-inform-cell-analysis-jhu-processing-farm "Title")
  - [5.10.4.1. Description](docs/SavingProjectsfortheinFormCellAnalysisJHUProcessingFarm.md#51041-description "Title")
  - [5.10.4.2. Instructions](docs/SavingProjectsfortheinFormCellAnalysisJHUProcessingFarm.md#51042-instructions "Title")
- [5.10.5. Setting up the Virtual Machines for inForm](docs "Title")
- [5.10.6. Adding Slides to the inForm Queue](docs "Title")
- [5.10.7. Processing inForm Cell Analysis Tasks](BatchProcessing "Title")
