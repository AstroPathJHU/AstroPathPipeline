%%--------------------------------------------
%% running the image flatwarp workflow
%%--------------------------------------------

INTRODUCTION

	The steps below create the flatwarp workflow. It assumes that the original
	im3 files for the whole sample are located under a root. The im3 files should
	be in the path "\root\sample\im3\Scan*\MSI\*.im3". We will need to specify a 
	staging directory, where all the temporary fiels will go. In the examples below
	this will be "Z:\test". 
	
	The temporary files will be written into "Z:\test\sample". We will also create 
	a binary DAPI layer to be used for the image alignments, with an extension of 
	*.fw01. This will go into the staging directory.
	
	The final output will be another set of .im3 files, which will go back into the
	main directory tree of the sample, to "\root\sample\im3\flatw\*.im3".

STEPS

	1.  Make sure that the *_M2.im3 files are properly resolved. Remove the original,
		and rename the _M2 by running
	 	
			fixM2 X:\Clinical_Specimen M2_1
	
	2. 	Extract the image bytes from the original im3 files. This will generate .raw
		files from the original image bytes, plus a .imm files with metadata. 
		Watch for using Scan1 or Scan2 in the second argument.
		
			shredPath X:\Clinical_Specimen Y:\new M2_1
	
	3.	Run the flatwarp matlab procedure. This will take the .raw files, applies
		the image warping, and the flat field corrections, and writes the result
		with a .fw extension.
	
			flatwPath Y:\new M2_1
			
	4.	Write the new im3 files with the modified content. the result goes back into
		the main im3 directory, into the flatw subfolder as modified .im3 images which
		can now be processed by the InForm package.
	
			injectPath X:\Clinical_Specimen Y:\new M2_1
			
	5.	We need to generate the DAPI layer from the .fw files. We extract the first 
		layer with an .fw01 extension by running
			
			extractLayer Y:\new M2_1 1
		
	6.	Next we need to do some cleanup. Delete all the unnecessary files (.raw and .imm) 
		from the staging directory.
		
			cleanPath Y:\new M2_1

	All these steps have been aggregated into a single script that will perform the whole 
	workflow on a single sample. You can run this as

			doOneSample X:\Cinical_Specimen Y:\flatw M3_1


		

			
		
			
