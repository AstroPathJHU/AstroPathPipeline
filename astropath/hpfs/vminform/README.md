# 5.8. VMinForm
## 5.8.1. Description
After the images are corrected by the ```image correction``` module and the ```flatw_path```s are created([5.7.](../imagecorrection/README.md#57-image-correction "Title")), the inForm® (Akoya Biosciences®) classification algorithms can be developed. The *AstroPathPipeline* uses the so-called 'multipass' method for cell classification. This is a novel method of using inForm Cell Analysis®, developed to improve sensitivity and specificty of classication algorithms (described in detail here [5.8.3](docs/inFormMultipassPhenotype.md#583-inform-multipass-phenotype "Title")). 

Once algorithms are created we process the algorithms through the queued based system. Instructions for setting up the virtual machines and processing queue as well as launching the tasks can also be found there. To use this system, each algorithm should be saved according to the protocol in [5.8.4.](docs/SavingProjectsfortheinFormJHUProcessingFarm.md#584-saving-projects-for-the-inform-jhu-processing-farm "Title") to ensure proper export settings. Next, add the slides to the queue according to the directions described in [5.8.5.](docs/AddingSlidestotheinFormQueue.md#585-adding-slides-to-the-inform-queue). Afterward slides will be processed through inForm®.

## 5.8.2. Contents
- [5.8.3. inForm® Multipass Phenotyping](docs/inFormMultipassPhenotype.md#583-inform-multipass-phenotype "Title")
  - [5.8.3.1. Description](docs/inFormMultipassPhenotype.md#5831-description "Title")
  - [5.8.3.2. Instructions](docs/inFormMultipassPhenotype.md#5832-instructions "Title")
    - [5.8.3.2.1. Getting Started](docs/inFormMultipassPhenotype.md#58321-getting-started "Title")
    - [5.8.3.2.2. Core Icons to Remember](docs/inFormMultipassPhenotype.md#58322-core-icons-to-remember "Title")
    - [5.8.3.2.3. Segment Tissue](docs/inFormMultipassPhenotype.md#58323-segment-tissue "Title")
    - [5.8.3.2.4. Adaptive Cell Segmentation](docs/inFormMultipassPhenotype.md#58324-adaptive-cell-segmentation "Title")
    - [5.8.3.2.5. Phenotyping](docs/inFormMultipassPhenotype.md#58325-phenotyping "Title")
    - [5.8.3.2.6. Export](docs/inFormMultipassPhenotype.md#58326-export "Title")
- [5.8.4. Saving Project for the inForm® JHU Processing Farm](docs/SavingProjectsfortheinFormJHUProcessingFarm.md#584-saving-projects-for-the-inform-jhu-processing-farm "Title")
  - [5.8.4.1. Description](docs/SavingProjectsfortheinFormJHUProcessingFarm.md#5841-description "Title")
  - [5.8.4.2. Instructions](docs/SavingProjectsfortheinFormJHUProcessingFarm.md#5842-instructions "Title")
- [5.8.5. Adding Slides to the inForm Queue](docs/AddingSlidestotheinFormQueue.md#585-adding-slides-to-the-inform-queue)
- [5.8.6. Evaluating inForm® Phenotype QC Output for the *AstroPath Pipeline*](docs/EvaluatinginFormPhenotypeQCOutputfortheAstroPathPipeline.md#586-evaluating-inform-phenotype-qc-output-for-the-astropath-pipeline)
- [5.8.7. Processing inForm® Tasks](docs/ProcessinginFormTasks.md#587-proccessing-inform-tasks)
  - [5.8.7.1. Description](docs/ProcessinginFormTasks.md#5871-description)
  - [5.8.7.2. Important Definitions](docs/ProcessinginFormTasks.md#5872-important-definitions)
  - [5.8.7.3. Instructions](docs/ProcessinginFormTasks.md#5873-instructions)
    - [5.8.7.3.1. Setting up the Virtual Machines for inForm®](docs/ProcessinginFormTasks.md#58731-setting-up-the-virtual-machines-for-inform)
    - [5.8.7.3.2. Running the VMinForm Module](docs/ProcessinginFormTasks.md#58732-running-the-vminform-module)
  - [5.8.7.4. Workflow](docs/ProcessinginFormTasks.md#5874-workflow) 
