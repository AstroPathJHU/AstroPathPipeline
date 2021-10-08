# 4.8. Transfer Daemon

## 4.8.1. Description
This code is used to transfer and reorganize the whole slide clinical specimen scans from the ```<Spath>``` server to the ```<Dpath>``` servers once they have been successfully acquired. The code also renames the slides from their ```SampleName```s, defined on the vectra microscope, to their unqiue ```SlideID```s defined in the *AstropathAPIDdef* file, created by and described in the documentation for the APID generator code. The transfer process for a specimen is initiated by the addition of a *BatchID.txt* file, created manually, to a successful ```<ScanNN>``` directory. These files should contain the numeric batch id, defined in the project *Specimen_Table.xlsx*. The code loops through each project defined in the *AstropathCohortsProgress.csv* and transfers new directories, the code can optionally delete the source files and can run with or without data compression. On error the code will send an email to a provided email address. Further description of the ```BatchID```s can be found [here](../docs/scanning/BatchIDs.md/#446-batchids), definitions for other variables can be found [here](../docs/Definitions.md#431-identification-definitions), and descriptions of various *AstroPath* files can be found [here](../docs/AstroPathProcessingDirectoryandInitializingProjects.md#451-astropath_processing-directory).

## 4.8.2. Important Definitions

1. We use path specifiers to shorten descriptions, further description of these paths can be found in the additional documentation [here](../docs/Definitions.md#432-path-definitions):
   - ```<Mpath>```: the main path for all the astropath processing *.csv* configuration files; the current location of this path is *\\\\bki04\astropath_processing*
   - ```<Dname>```: the data name or the name of the clinical specimen folder
   - ```<Dpath>```: the data or destination path
      - this is the path to the project's data on the bki servers
   - ```<Spath>```: the source path to the project's data
   - ```<Cpath>```: the compressed path for the project's data
   *NOTE*: the ```<path>``` variables do not contain the ```<Dname>```

## 4.8.3. Instructions
For python download the repository and install the transferdeamon. Then launch using:

```TransferDaemon.py <Mpath> <email_on_error> [<source_file_handling>] [<logical_compression>] [<quiet>] [<version>] [<debug>]```

- ```<Mpath>```: should contain the ***AstropathCohortsProgress.csv***, ***AstropathPaths.csv***, and the ***AstropathCohorts.csv*** files
  - description of these files can be found [here](../docs/AstroPathProcessingDirectoryandInitializingProjects.md#451-astropath_processing-directory)
- ```<email_on_error>```: a valid email address to send error information to
- ```[<source_file_handling>]```: Optional argument; should be one of three options:
  - ```HYBRID```: Default; Follow CSV protocol but leave folders with DoNotDelete.txt file
  - ```AUTOMATIC```: DELETE ALL FILES under the CSV protocol
  - ```MANUAL```: Delete any folders without the DoNotDelete.txt file
- ```[<logical_compression>]```: Optional argument; enter "-no_compress" if files are not to be compressed
- ```[<quiet>]```: Optional argument; enter "-q" if no output is to be sent to command window
- ```[<version>]```: Optional argument; enter "-v" to get the version number of the code
- ```[<debug>]```: Optional argument; enter -d to run in debug mode. Here the code outputs more descriptive information and requires addition of "\\" when attempting to access server locations
  
## 4.8.4. Workflow
### 4.8.4.1. Initial Transfer
The first step the Demon takes is to transfer the files from a given folder to the designated spot on BKI04. During the transfer process, all files which contain the SampleName are changed to contain the new SlideID generated in ASTgen.py. The script copies files from ```<Spath>\<Dname>\<SampleName>\<scan>``` to ```<Dpath>\<Dname>\<SlideID>\<im3>\<scan>```. 

In the case of duplicate im3 files with the ‘_M#’ extension, the code processes them in the following way:
  1.	Finds the highest <filename>_M#.im3 file
  2.	Removes all other <filename>.im3 or <filename>_M#.im3 files
  3.	Renames the highest <filename>_M#.im3 file to <filename>.im3

### 4.8.4.2. MD5 Check
After the transfer process, the code then makes a copy of the local annotations.xml file for the specimen. The unchanged version has a ‘-original’ tag and the new version has the SampleName changed to the new SlideID in the metadata. 
It then creates ‘CheckSums.txt’ files for both folders and compares the hash values of two. This allows the Demon to know if the files were corrupted during transfer.

### 4.8.4.3. Compression Into Backup
If all files match, the script then begins the compression process. The files are compressed from the ```<Spath>\<Dname>\<SampleName>``` to BKI03 for cold storage at ```<Cpath>\<Dname>\<SlideID>``` (compresses to about half the size of the original). The Demon uses parallel processing to move multiple copies at once for increased speed.

### 4.8.4.4. Source File Handling
The original scan will then be deleted if the user has chosen to do so (```<source_file_handling>```). If the source files are not deleted, the script re-checks the directories for changes in files using MD5 hashes.

## 4.8.5. Notes

If the source folder has less files than the destination folder, the Demon will interpret this as if the source folder was not successfully deleted in the transfer process and it will delete that source folder. If you do not want the source folder to be deleted either… 
- Clear the destination and compressed folders or… 
- Do not delete files from the source folder, only add them.

There is a ‘transfer.log’ file within a ‘logfiles’ folder in each of the destination folders on BKI04 that records which specimens have been processed from that Clinical Specimen folder. There is a more detailed transfer.log file for each specimen in ```<Dpath>\<Dname>\<SlideID>\<logfiles>```.

The Code is meant to be run all the time and just waits for files to be ready. You can also choose to just run it whenever you need to. It will not slow the machine down if it is just waiting with nothing to do. You can hit ctrl + c to end the code or close out the console. 

** If you stop the code and it has not completed the case it was working on, you can delete the partially transferred file, the Checksums.txt file, and the compressed files and restart. However, the script should be able to account for any incongruities in the transfer process. **
