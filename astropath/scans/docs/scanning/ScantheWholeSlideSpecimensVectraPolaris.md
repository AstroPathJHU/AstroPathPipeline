# 4.4.5.5. Scan the Whole Slide Specimens (Vectra Polaris)
If everything checks out scan the specimens as follows for the Vectra Polaris
1. Take exposures (create a single protocol) for all of the clinical specimen multiplex slides 
    - The name of the protocol should be *BatchX_MM.DD.YYYY* 
    - Save to the corresponding ```<spath>``` folder
1.	On the **Slide Log Template**
	1.	Enter the Clinical_Specimen_XX study in the column **Study**. **XX** should match the next available Project Number
	1.	Enter the appropriate scanning protocol for each slide in the column **Protocol**
	1.	Select **Scan Only** in the **Task** column
	1.	In box **G1** enter a name to help you identify the setup
	1.	Click the **Save** icon on the spreadsheet
1.	On the computer desktop double-click the icon **Polaris_setup**
1.	In the Vectra Polaris start menu, click **Scan Slides**
1.	Click **Load Setup**, select the setup that matches the information you entered in box **G1**, and click **Load**.
1.	Click **Scan** at the top of the screen.
1.	Watch the Polaris until you see **Performing fluorescent whole slide scan using protocol XXX**.
1.	Load the slide carrier with the slides
    - Clean the slides using a Kimwipe to remove dust and excess mounting medium
    - Load the carriers so that the first slide corresponds to Slot 1 on the Task list
1.	Scan the whole slide overview on the microscope
1.	Create the annotation file for the [Whole Slide Overlap imagery](CreatingaWholeSlideScanningPlan.md)
1.	Acquire HPFs on the microscope
1.	If using a network storage server, wait for data to backup
    - Manually check that the image files are in both the local and backup locations
      - Open two windows explorers, one for the local and one for the backup location
      - Check that the same number of im3s exist in both locations 
      - Sort by ‘Size’ in the windows explorer to be sure that all of the images are the same size
        - If there is a file with 0 bytes, check for M# duplicate file. This is the only time variation in field size is allowed, see ‘Notes’ section below for details on how to handle M# files.
      - Check that the annotation.xml modified data are the same
1. Check that the data scanned properly
   - Open the overview scans in *Phenochart* and check that at least 95% of the fields correctly scanned
      - Never retake single HPFs, if more than 5% of fields failed restart the entire HPF set
      - If a HPF fails in the region of interest, rescan the slide
      - By making the image files 'extra large icons' a quick visual inspection can be performed of the slides. Here check for two things:
        - sometimes the microscope gets mixed up and scans the wrong part of the tissue, usually this creates a number of empty fields but can be hard to catch 
          - the AstroPath Pipeline code has been modfied to handle such cases so this is not a breaking issue but does cause a loss of data.
        - sometimes the microscope will automatic focus will fail and fields will be blurry
        - In both cases rescan the HPFs
1. If you need to rescan a slide:
   - Go into the folder of the slide you need to rescan
   - Create a new folder
   - Name the folder ScanX where X is the number of the last scan folder plus 1
   - Go into the failed scan folder and copy everything except for the **MSI** folder and both annotation files
   - Copy into the new Scan folder
   - On the qptiff, change ScanX to match the folder name
   - Open the qptiff and redo the ROI
1. If using a network server for microscope backup, delete the local copy of the data 
1.	Add the *BatchID.txt*, described above, to the proper ‘Scan’ folder
