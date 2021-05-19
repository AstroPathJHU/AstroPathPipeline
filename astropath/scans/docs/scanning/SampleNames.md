# 4.4.3. SampleNames (Patient # or M Numbers)
When slides are cut, they may be labeled with a patients medical record number (MRN) or other unique patient number. In order for the slide to be processed through the pipeline, it must be given a de-identified number. These are the so called ```SampleName``` (in past versions ‘M’ numbers or ```Patient #```s). Because slide naming conventions maybe different across groups or institutions, consistence in these ```SampleName```s  can be difficult to keep up. To avoid any issues in processing the ```SampleName```s are converted to the ```SlideID```s as part of the intial steps in the AstroPathPipeline.

For these ```SampleName```'s, each cohort receives its own alphabetical key at JHU. Currently we are using:
-	*M* for melanoma
-	*L* for Lung
-	*MA* for melanoma stain 2

Within a cohort every slide stained will receive a new numeric value attached to the ```SampleName```s, the order of these do not matter. Examples for the first four slides stained of a cohort may be:
- *M1*, *M2*, *M3*, *M4*
- *M9*, *M100*, *M42*, *M35_3*

If a Batch fails these slides should still receive a ```SampleName``` and show up in the *Speciment_Table_N.xlsx*. 
