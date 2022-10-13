# 4.4.5.4. Scan the Whole Slide Specimens
If everything checks out scan the specimens as follows:
1. Take exposures (create a single protocol) for all of the specimen multiplex slides 
    - The name of the protocol should be *Multiplex_MMDDYYYY* 
    - and should be saved to the corresponding ```<spath>``` folder
2.	Create a task list (.csv format) for the scanning
    - It should be named Multiplex_MMDDYYYY.csv
3.	Load the slide carrier with the slides
    - Clean the slides using a Kimwipe to remove dust and excess mounting medium
    - Load the carrier from bottom to top so that the first slide corresponds to Slot 1 on the Task list
4.	Scan the whole slide overview on the microscope
5.	Create the annotation file for the [Whole Slide Overlap imagery](CreatingaWholeSlideScanningPlan.md)
6.	Acquire HPFs on the microscope
7.	If using a network storage server, wait for data to backup
    - Manually check that the image files are in both the local and backup locations
      - Open two windows explorers, one for the local and one for the backup location
      - Check that the same number of im3s exist in both locations 
      - Sort by ‘Size’ in the windows explorer to be sure that all of the images are the same size
        - If there is a file with 0 bytes, check for M# duplicate file. This is the only time variation in field size is allowed, see ‘Notes’ section below for details on how to handle M# files.
      - Check that the annotation.xml modified data are the same
8. Check that the data scanned properly
   - Open the overview scans in *Phenochart(C)* and check that at least 95% of the fields correctly scanned
      - Never retake single HPFs, if more than 5% of fields failed restart the entire HPF set. 
      - By making the image files 'extra large icons' a quick visual inspection can be performed of the slides. Here check for two things:
        - sometimes the microscope gets mixed up and scans the wrong part of the tissue, usually this creates a number of empty fields but can be hard to catch 
          - the AstroPath Pipeline code has been modfied to handle such cases so this is not a breaking issue but does cause a loss of data.
        - sometimes the microscope will automatic focus will fail and fields will be blurry
        - In both cases rescan the HPFs
9. If using a network server for microscope backup, delete the local copy of the data 
10.	Add the *BatchID.txt*, described above, to the proper ‘Scan’ folder
