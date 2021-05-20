Creating the stitching workflow

Step 1. 
	Unpack the flatw im3 fiels into raw format
	For each sample execute the following batch command
		c:\mcode\Im3Tools\doShredPath E:\Clinical_Specimen im3\flatw F:\DAPI M1_1
	where M1_1 should be substituted with the name of each sample.
	The resulting files will have the .raw extension, and they will contain the
	flat-fielded and warped images. Each sample will have its own subdirectory
	in F:\DAPI.

Step 2.
	Extract the DAPI layer (520nm) from the .raw files. In order to do this, start
	MATLAB, go into the directory \\BKI02\C$\mcode\matlab\stitch5 and execute
		extractRawLayer('c:\dapi','*',1);