# 4.4.5.6. Important Scanning Notes
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
