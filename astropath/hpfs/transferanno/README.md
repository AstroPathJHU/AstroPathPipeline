# 5.12. Transfer Annotations
## 5.12.1. Description
To improve analysis capability in the database slides should be hand annotated by a pathologist using the HALO software. In this section directions for creating the annotations, exporting them, and transferring them to the proper locations for database upload are included. Importantly the database upload includes using a specific set of layers, layer order, and names provided in a data dictionary below. Adding additional layers is possible but layer names must be consistent and submitted for approval prior to database upload. These requirements are in place to so that data checks and verfication can be performed allowing seemless database injestion and subsequent analyses.

## 5.12.2. Creating HALO Annotations for the *AstroPath Pipeline*
### 5.12.2.1. Prepare the HALO Project
1. Open the HALO application
2. Create a new HALO project folder. 
   - Open the *Studies* tab
   - Right click on the desired folder destinatio
   - Create a new project folder
3. Map the *Network Drive*
   - Right click on *MyComputer*
   - Click on *Add Network Drive*
4. Add *.qptiff* images to the HALO project folder
   - Click on *File*
   - Click on *Open Images*
   - Navigate to the ```<Scan>``` folder for each case
     - under ```<Dpath>\<Dname>\<SlideID>\<im3>```
   - select the *.qptiff* Akoya whole slide scan
   - select the desired halo project folder to input the images
### 5.12.2.2. Create and Edit Annotation Layers
1. Create a layer
   - Double click on the *.qptiff* to be annotated in HALO
   - Open the HALO *Annotations* tab
   - Under *Layer Actions* select *New Layer*
     - *Layer 1* should appear
   - Add as many new layers as needed
2. Edit a layer names and colors
   - Make sure the appropriate layer is selected in the drop down menu 
   - Rename and change the color of the layers
   - Be sure to follow the suggested layer conventions
      - Use *Layer Actions* 
      - **Be careful in naming each layer consistently across all specimens and are in order**
3. Use the pen annotation mode to create polygons. Use the exclusion tool (*scissors*) to exclude areas of analysis
   - Regions to exclude would be things like:
     - tears in tissue
     - regions with *edge effect*
   - Polygons can be edited by retracing the orginial polygon.
   - Polygons can also be deleted, grown, or shrunk by right clicking the polygon and selecting the appropriate option.
   - Polygons can also be copied from one layer to annotation
4. Save the annotations (*crtl + s*)
5. View settings can be adjusted to highlight specific fluorescence channels to assist in the annotation process

### 5.12.2.3. Annotation Layer Data Dictionary


*NOTE*: Not all layers listed needed to be created but if you need to make additional layers for your project, start with Layer 6. Record your layer number and names below and send those back to the TME.

### 5.12.3. Exporting Annotations
Once all cases have been annotated the annotations need to be exported.
1. Open the *Studies* tab
2. Select all specimens with annotations to be exported
3. Right click on the files
4. Select *Export and Advanced*
5. Uncheck the default options
6. Check the *Annotations* box
7. Select the desired destination folder
8. Click *Generate*

### 5.12.4. Transfer Annotations to the BKI Server
### 5.12.4.1. Description
A simple set of batch files have been written to assist in dissemination of the annotations from the *Export* folder to the respective ```<Dpath>\<Dname>\<SlideID>\<im3>\<Scan>``` folders of each slide for data base upload. The script also renames the annotation files, adding *.xml* to the file extension and removing the unique annotation ID added to the file names by HALO. Running the batch file is very simple and directions follows.

### 5.12.4.2. Instructions
1. Locate the location of the exported annotations.
   - The folder will be inside the *desired destination folder* selected in [5.12.3. Step 7](#5123-exporting-annotations "Title")
   - Usually this folder will be labeled as *HALO archive YYYY-MM-DD <HALO version>* 
     - Where the exported date replaces *YYYY-MM-DD* and the current HALO version number replaces*<HALO version>*
   - The annotations themselves are located in an *Images* subfolder, this is the folder we are looking for but we just need the *archive* folder. Record the fully qualified path up to and including to this folder.
     - E.g. *\\HALO1\user\Halo archive 2021-04-20 12-22 - v3.1.1076*
2. Download the github *AstroPathJHU\AstroPathPipeline* repo
3. Open a command prompt
   - For Windows in the *Start Menu Search Bar* type *cmd* and it should be the first option
4. Enter: 
``` */AstroPathPipeline/astropath/hpfs/transferanno/TransferAnno "<source>" "<destination>" ```
   - ```*/```: The path to where the repo was downloaded
     - An example of the full command where the repo was downloaded to the desktop would be: *c:\users\username\Desktop\AstroPathPipeline/astropath/hpfs/transferanno/TransferAnno*
   - ```<source>```: the path recorded in step 1
     - E.g. *\\HALO1\user\Halo archive 2021-04-20 12-22 - v3.1.1076*
   - ```<destination>```: the project directory (referred to as ```<Dpath>\<Dname>``` in the *AstroPathPipeline* documentation
     - E.g. *\\bki07\Clinical_Specimen_12*
     
