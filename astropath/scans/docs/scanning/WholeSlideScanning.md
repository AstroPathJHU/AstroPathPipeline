# 4.4.5. Whole Slide Scanning
This section describes the methods for whole slide scanning. The directions assume that the user has some working knowledge of how the Akoya multiplex microscopes function. While a skilled user of the Akoya platform would be able to easily acquire whole slide imagery, these directions provide nuances that must be adhered to for successful slide processing in the AstroPath Platform. Failure to adhere to these guidelines may crash the pipeline and render whole slide imagery incompatible with the AstroPath Platform. This section is started with directions for configuring the [20% field overlap](#4341-preparing-20-overlap "Title"). This is followed up with steps taken to scan\ evaluate the [control tmas](#4342-scan-the-control-tma "Title") and [whole slide samples](#4343-scan-the-whole-slide-specimens "Title"). Again these instructions assume that the user has working knowledge of the Akoya platform and should not be used in lieu of proper microscope training. Next, a brief description on how to create a [whole slide scanning plan](#4344-creating-a-whole-slide-scanning-plan "Title") is provided. Finally, [key notes and specifications](#4345-important-scanning-notes "Title") are provided for scanning. These notes are particularly important and should be reviewed carefully.

## 4.4.5.1. Contents

#### 4.4.5.1. Preparing 20% overlap
As part of the AstroPath Pipeline, slides must be scanned with fields that overlap by 20%. We use this overlap to measure and/ or correct for errors in imagery, including warping, illumination, segmentation, and positioning. This is step should be performed on each computer that will be used to create scanning plans in *Phenochart(C)*. The changes made are permant and should only need to be performed once for a computer.
1. Install the lastest version of *Phenochart(C)*
2. Be sure that *Phenochart(C)* is shut down
3. Go to *C:\ProgramData\PerkinElmer\Phenochart\* (Please note that ProgramData is usually a hidden folder)
4. Copy and paste the *Phenochart.config* file with one that does 20% overlap (a version of this is located in this folder)
5. Check ‘ROIs with overlap’ as on under the settings in *Phenochart(C)*

#### 4.4.5.2. Scan the *Control TMA*
1. Take references for the microscope
2. When a batch is finished staining we first check that the slides are designated in the *Specimen_Table_N* ([4.3.1](#431-specimen_table “Title”)).
   - If the slide is designated in the *Specimen_Table_N*; check that the ‘M’ numbers are written on the slides and that they are the correct numbers
   - If the slide is not designated in the *Specimen_Table_N* insert the required information and give the slides ```SampleName```s corresponding to the next available number (again values should not be repeated even if a batch fails)
3. Set up the scanning exposures (the ‘Protocol’) for the TMA in the proper ```<spath>``` folder
   - Name the protocol and the task list-> *Control_TMA_XXXX_YYY*
     - XXXX -> the TMA ID (ie. 1372)
     - YYY -> the TMA cut number (ie. 126)
     - Old scanning protocols will be located in any Clinical_Specimen folder under Protocols 
4. Name the control TMA slide -> *Control_TMA_XXXX_ZZ_MM.DD.YYYY*
   - XXXX is the TMA ID
   - ZZ is the TMA cut number
   - MM.DD.YYYY is the month, day, and year of when staining finished
5. Scan the whole slide overview on the microscope
6. Create an annotation scanning plan in Phenochart
   - Open the TMA in  Phenochart
   - Click ‘Login’ at the upper left-hand corner
     - Type your name or initials
    - Click ‘TMA’ in the upper middle of the screen
      - Change the plan so that the bottom right TMA is [1,1]
    - Change ‘core size’ to 1.5 mm
    - Change the ‘Imaging Size’ to ‘3x3’ Image
    - Draw the rectangle around all cores for Phenochart to locate the cores
    - To do this click and hold near one of the corners and drag across to the opposite corner (a search box should be displayed by Phenochart)
    - If a core is missing right click the location of the core and select ‘Add missing core’ then select the proper core
    - If the cores are mislabeled you will have to start over and draw a new box (select ‘clear grid’ at the top to start over)
    - Center the squares around the expected center of the Cores
      - To center the cores click and hold ‘Ctrl’ 
      - With the mouse click the center of one of the cores and move it
      - If much of the core is missing, still center around what would be the expected center
        - Because we do not correct for image distortion effects in the TMAs, if one core is missing half its tissue we still want to image the other half in the same way we would regularly
      - Once you are satisfied click ‘Accept Grid’ and close Phenochart
7. Select the ‘Acquire MSI fields’ task for the TMA on the microscope and scan the slide
8. Do the quality control of the TMA
   - Open three cores from a previous Control TMA in inForm
     - Be sure that the TMA cores came from the same cohort and from a Batch that worked 
     - Example Cores for methods would include:
   - Open the same three cores from the current TMA in inForm
     - The ‘Core’ coordinates should be the same between batches
   - Open an Algorithm with a current library
   - Prepare all images with the library 
   - Compare the counts and the stain quality between the old and the new TMA
     - Click on the icon with the box next to a mouse pointer to compare the counts between the old and new  TMAs
     - It is easiest to write down the Opal and its respective counts range on a sheet of paper to refer to when switching between old and new versions
     - The colors are always as follows
       - Opal520 – Green
       - Opal540 – Yellow
       - Opal570 – Red
       - Opal620 – Orange
       - Opal650 – Cyan
       - Opal690 – Magenta 
     - Look to see if general patterns are the same and if intensities vary
     - It is usually easy to see differences visually and use counts as a sanity check for your eyes
     - Because the of the similarities in the cores and stains within a cohort the variation between slides should be minimal
9. Give the TMA a *BatchID.txt* file and give it the next corresponding ```BatchID```
10. Fill in the ```BatchID``` on the *Specimen_Table_N.xlsx* if it is not already present

#### 4.4.5.3. Scan the Whole Slide Specimens
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
5.	Create the annotation file for the [Whole Slide Overlap imagery](#4344-creating-a-whole-slide-scanning-plan "Title")
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

#### 4.4.5.4. Creating a Whole Slide Scanning Plan
1.	Take lower power overview scan as normal
2.	Open Phenochart and select ROI
3.	Circle the entire tissue, click OK on dialog box that appears
4.	Delete empty fields around and inside tissue boundaries

#### 4.4.5.5. Important Scanning Notes
1.	To scan multiple pieces of tissue:
    - if they are close enough that there is asymmetric overlap between tissues include them in the same ROI drawing
    - If there is no overlap between fields; circle the ROIs separately 
2. Include the entire tissue: folds, necrosis, etc. 
3. If a large number of HPFs fail in slide scanning, rescan the QPtiff to create a Scan2 (or next Scan number) folder, draw a new ROI, and start the scan over (do not delete Scan1 folder)
   - This is a way to track for errors, remove human interaction with the data, and reduce issues that may come up due to multiple scans
   - It is possible to manually create a Scan2 folder and copy the QPtiff from Scan1 (renaming all files to Scan2), if you do this be sure to delete the annotations.xml file in Scan2 before drawing a new ROI
   - Much of our codes look to the .xml files to determine the expected number of .im3 files
4.	If only a few HPFs fail do not rescan them
5.	If it is a core biopsy and two cuts have been placed on the same slide; only select HPF annotation for one
6.	Some files may end up with an ‘_M#.im3’ designation at the end of the file name. This means that the HPF was scanned twice. The duplicate is labeled with the ‘M2’ designation, but the initial scan has no designation and has 0 bytes. Do not delete either file. This is issue is dealt with in the automated pipeline.
