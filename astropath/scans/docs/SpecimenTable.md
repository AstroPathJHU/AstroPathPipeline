# 4.4.1. Specimen_Table
The specimen table is used to intialize the slides into the pipeline. This table also serves as the link between the names on the slides, the ```Specimen #```s, and the de-identified ```SampleNames```s (refer to [4.4.1](#441-identification-definitions "Title") for a description of the variable names). The specimen table should be labeled *Specimen_Table_N.xlsx*, where *N* indicates the same unique specifier on the *Clinical_Specimen_N* folder. This file should always be contained in the ```<Spath>``` location with the original slide scans. If the ```Specimen #```s on the slides are pathology numbers or other medical identifiers, as is the case at Johns Hopkins, this location should be HIPAA compliant. The file has the following columns:
```
Patient #, Specimen #, Cut Data, Level, Batch ID, Stain Date, Scan Date
```
- ```Patient #```: this is the ```SampleName``` defined in [4.2.1](#421-identification-definitions "Title")
- ```Specimen #```: this the specimen number that is used to identify the patient clinical information
- ```Cut Date```: the date that the slide was cut from the tissue block
- ```Level```: the cut number from the sections cut on that date
- ```Batch ID```: the staining batch the slides was stained with, more details below in [4.3.5](#435-batchids “Title”).
- ```Stain Date```: the date the slide was stained on 
- ```Scan Date```: the date the slide started HPF scanning
