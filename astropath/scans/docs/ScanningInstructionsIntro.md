# 4.4 Scanning Instructions Intro
Before scanning it is important to set up the ```<Spath>```. This folder is created on the scanning computer where the slides are scanned into. The folders are usually labeled *Clinical_Specimen_XX*, where the *XX* indicates a numeric value or unique lettering. Examples of these scanning folders incude *Clinical_Specimen_2* and *CLinical_Specimen_BMS_01*. In the JHU processing pipeline, this folder is backed up every night, using commercially available software, to a network server with significant storage capacity (~80TB). Once slide scans are completed they can be deleted from the local computer, the rest of the processing takes place on the network server location. In this way, the local computers never run in storage issues. *NOTE*: The fully qualified path for this scanning folder on the network server is designated as the ```<Spath>```. 

After a new batch of slides are stained, they should be added to a *Specimen_Table_N.xlsx* file located in each ```<Spath>```, described in detail below in [4.3.1](#431-specimen_table "Title"). As part of adding slides to this table, the slides will be given a unique de-identified name for scanning, the ```SampleName```. Tips on these names are included in [4.3.2](#432-samplenames-patient--or-m-numbers "Title"). The most important aspect of this convention is to avoid the use of spaces and special characters. For each batch, a control tma should also be stained and scanned. The scanning and naming of this slide is also very important for pipeline ingestion and is defined below in [4.3.3](#433-control-tma-conventions "Title"). *Note*: The control tmas do not go into the *Specimen_Table_N* and are found based on their **naming**.

Once added to the *SpecimenTable.xlsx*, slides can be scanned with 20% overlap according to the protocol laid out in [4.3.4](#434-whole-slide-scanning "Title"). In order for successful processing of the slides, it is very important that this procedure is adhered to correctly. After slides are scanned, the user should manually verify that all images were scanned completed properly and add a *BatchID.txt* file to the successful ```Scan``` directory. This initiates the slide transfer process in the pipeline, additional details on this step are defined in [4.3.5](#435-batchids "Title"). It is also important to create the *Batch_BB.csv* and *MergeConfig_BB.csv* files for processing to continue successfully. Each staining batch defined should have a separate set of these tables. Information on these files can be found in [4.3.6](#436-batch-tables "Title") and [4.3.7](#437-mergeconfig-tables "Title"), respectively. 

Again due to length this documentation is written in sections and linked to here in a table of contents. Sections are added in sequence in which they should be completed and should be read as such.

## 4.4.1. Contents
   - [4.4.2. Specimen Table](docs/scanning/SpecimenTable.md)
   - [4.4.3. SampleNames (Patient # or M Numbers)](docs/scanning/SampleNames.md)
   - [4.4.4. Control TMA Conventions](docs/scanning/ControlTMAConventions.md)
   - [4.4.5. Whole Slide Scanning Introduction](docs/scanning/WholeSlideScanning.md)
     - [4.4.5.1. Contents](docs/scanning/WholeSlideScanning.md/#4451-contents)
     - [4.4.5.2. Preparing 20% Overlap](docs/scanning/Preparing20Overlap.md)
     - [4.4.5.3. Scan the *Control TMA*](docs/scanning/ScantheControlTMA.md)
     - [4.4.5.4. Scan the Whole Slide Specimens](docs/scanning/ScantheWholeSlideSpecimens.md)
     - [4.4.5.5. Creating a Whole Slide Scanning Plan](docs/scanning/CreatingaWholeSlideScanningPlan.md)
     - [4.4.5.6. Important Scanning Notes](docs/scanning/ImportantScanningNotes.md)
   - [4.4.6. BatchIDs](docs/scanning/BatchIDs.md)
   - [4.4.7. Batch Tables](docs/scanning/BatchTables.md)
   - [4.4.8. Merge Configuration Tables](docs/scanning/MergeConfigTables.md)
