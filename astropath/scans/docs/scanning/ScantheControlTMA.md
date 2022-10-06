# 4.4.5.3. Scan the *Control TMA* on Vectra Polaris
1. Take references for the microscope
2. When a batch is finished staining we first check that the slides are designated in the *Specimen_Table* ([4.3.1](#431-specimen_table “Title”)).
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
   - Click **Login** at the upper left-hand corner
     - Type your name or initials
    - Click **ROI** in the upper middle of the screen
    - Draw the rectangle around all cores for Phenochart to locate the cores
    - To do this click and hold near one of the corners and drag across to the opposite corner (a search box should be displayed by Phenochart)
    - Delete any fields that are completely missing tissue. If you are unsure if you should delete the field, leave it for scanning.
    - To delete fields you can either:
      - Hold down **Ctrl** on the keyboard and left-click on empty fields
      - Right-click on the field and select **Delete**
7. Select the **Acquire MSI fields** task for the TMA on the microscope and scan the slide
8. Do the quality control of the TMA
   - Open images from three tonsil cores from a previous Control TMA in inForm
     - Be sure that the TMA cores came from the same cohort and from a Batch that worked 
   - Open the fields that most closely match the three cores from the current TMA in inForm
   - Open the Algorithm with a current library, created for each Clinical Specimen Project
   - Prepare all images with the library 
   - Compare the counts and the stain quality between the old and the new TMA
     - Click on the icon with the box next to a mouse pointer to compare the counts between the old and new  TMAs
     - It is easiest to write down the Opal and its respective counts range on a sheet of paper to refer to when switching between old and new versions
     - The colors are always as follows
       - DAPI - Blue
       - Opal480 - Pink
       - Opal520 – Green
       - Opal540 – Yellow
       - Opal570 – Red
       - Opal620 – Orange
       - Opal650 – Cyan
       - Opal690 – Magenta
       - Opal780 - White 
     - Look to see if general patterns are the same and if intensities vary
     - It is usually easy to see differences visually and use counts as a sanity check for your eyes
     - Because the of the similarities in the cores and stains within a cohort the variation between slides should be minimal
9. Give the TMA a *BatchID.txt* file and give it the next corresponding ```BatchID```
10. Fill in the ```BatchID``` on the *Specimen_Table.xlsx* if it is not already present
