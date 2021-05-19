# 4.4.7. Batch Tables
For each ```BatchID``` a *batch table* should be created, this file records the staining\ reagent information for that batch including lot numbers and antibody-opal pairs. These tables should be added to a *Batch* folder in the ```<Dpath>\<Dname>``` folder The full directory structure for the ```<Dname>``` folder can be found in [4.5](#45-directory-organization "Title"). Typically the person who performs the slide staining for a batch has the best access to this information. As such they should fill out this document and add it to the appropriate location as possible after staining to avoid delays. 

The *batch tables* should be named *Batch_BB.xlsx*, replacing the *BB* with the repective ```BatchID```. The tables should contain the following columns:
```
BatchID, OpalLot, Opal, OpalDilution, Target, Compartment, AbClone, AbLot, AbDilution
```
New and ambigious columns\ variables are described below:
- ```Target[string]```: name if the antigen the applied antibody is targeting
   - name does not need to be very technical by should unique
   - the name should also be used for:
      - the opal labels in inForm
      - the positive phenotype for each marker
      - the folder for each of the separate inForm outputs
      - **Exception**: the tumor marker (also designated in ImageQA)
        - For this marker, use 'Tumor' to desgnate the output folder
        - Optionally: use 'Tumor' when desgniating that antibody in inForm. 
        - Must use same name for phenotype and opal namings
- ```Compartment```: The cell compartment from the inForm tables to use when loading in the database. Options include ```EntireCell```, ```Membrane```, ```Nucleus```, and ```Cytoplasm```
*NOTE*: When adding to the OpalLot, Opal, AbClone, and ABLot columns in excel spreadsheets be sure to add a single quote (') before the value, to specify that the column is a string and not a number. Always use the *1to* designation for concentrations.

There should be a row for each stain applied to the batch. The stains should start with DAPI then added in increasing opal order. An example table from the *Vectra 3.0* microscope is shown below:

![Figure 1 Image](../www/Fig1.PNG)
