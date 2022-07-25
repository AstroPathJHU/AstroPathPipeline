# 4.4.3. SampleNames (Patient # or CS Numbers)
When slides are cut, they may be labeled with a patients medical record number (MRN) or other unique patient number. In order for the slide to be processed through the pipeline, it must be given a de-identified number. These are the so called ```SampleName``` (in past versions ‘M’ numbers or ```Patient #```s). Because slide naming conventions maybe different across groups or institutions, consistence in these ```SampleName```s  can be difficult to keep up. To avoid any issues in processing the ```SampleName```s are converted to the ```SlideID```s as part of the intial steps in the AstroPathPipeline.

Previously for these ```SampleName```'s, each cohort received its own alphabetical key at JHU. Currently we are using:
-	*M* for melanoma
-	*L* for Lung
-	*MA* for melanoma stain 2

Currently for these ```SampleName```'s, each cohort receives its own Clinical Specimen Project Number. Currently we are using:
-	To determine the next Clinical Specimen Number go to \\\BKI04\astropath_processing
-	Open **AstropathConfig.csv**
-	Looking in the **Project** column, add one to the highest number to get the new Clinical Specimen number

Within a cohort every slide stained will receive a new numeric value attached to the ```SampleName```s, the order of these do not matter. Examples for the first four slides stained of a cohort may be:
- *CS23_1*, *CS23_2*, *CS23_3*, *CS23_4*

If a Batch fails during staining, these slides should still receive a ```SampleName``` and show up in the *Specimen_Table.xlsx*. 
