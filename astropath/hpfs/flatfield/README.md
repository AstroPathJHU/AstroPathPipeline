# 5.5. Flatfield

## 5.5.1. Description
The `flatfield` subworkflow provides routines to create meanimages for single samples, find the sets of samples that should be combined into flatfield correction models (`meanimagecomparison`), and create those flatfield correction models (`batchflatfield`). It also provides some code to double-check the efficacy of applying the determined flatfield corrections within a cohort. 

See [here](../../scans/docs/Definitions.md#43-definitions) for definitions of the terms in `<angle brackets>`.

## 5.5.2. Contents
- [5.5.3. Mean Image](docs/MeanImage.md#553-mean-image)
  - [5.5.3.1. Instructions to Run via *AstroPath Pipeline* Workflow](docs/MeanImage.md#5531-instructions-to-run-via-astropath-pipeline-workflow "Title")
  - [5.5.3.2. Instructions to Run Standalone via Python Package](docs/MeanImage.md#5532-instructions-to-run-standalone-via-python-package "Title")
  - [5.5.3.3. Instructions to Run Standalone Version *0.0.1*](docs/MeanImage.md#5533-instructions-to-run-standalone-version-001 "Title")
- [5.5.4. Mean Image Comparison](docs/MeanImageComparison.md#554-mean-image-comparison "Title")
  - [5.5.4.1. Instructions to Run via *AstroPath Pipeline* Workflow](docs/MeanImageComparison.md#5541-instructions-to-run-via-astropath-pipeline-workflow "Title")
  - [5.5.4.2. Instructions to Run Standalone via Python Package](docs/MeanImageComparison.md#5542-instructions-to-run-standalone-via-python-package "Title")
- [5.5.5. Batch Flatfield](docs/Batchflatfield.md#555-batch-flatfield "Title")
  - [5.5.5.1. Instructions to Run via *AstroPath Pipeline* Workflow](docs/Batchflatfield.md#5551-instructions-to-run-via-astropath-pipeline-workflow "Title")
  - [5.5.5.2. Instructions to Run Standalone via Python Package](docs/Batchflatfield.md#5552-instructions-to-run-standalone-via-python-package "Title")
  - [5.5.5.3. Instructions to Run Standalone Version *0.0.1*](docs/Batchflatfield.md#5553-instructions-to-run-standalone-version-001 "Title")
- [5.5.6. Batch Flatfield Tests](docs/Batchflatfield.md#556-batch-flatfield-tests "Title")
