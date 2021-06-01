CALIBRATION

Alex Szalay, Baltimore, 2018-11-04

	The code in this directory takes the calibration images
    and derives the mean flux in each of the markers. In order
    to do this we create a binary mask to separate the background
    pixels and only use the inside of the mask to compute fluxes.
			
	We use the component images in each of hte Control slides,
    located in the project directory, in \inform_data\Component_Tiffs\* 
    subdirectory. A single layer image size is 4028 x 3008, made up
    by a 3x3 mosaic of tiles, with a 2 pixel overlap. 

    The whole processing is automated, and it is executed by running

        C=runCalibration(root[,opt]);

    This step requires various files to be present:
        Subdirectories for the Control sets:
            <project>\Control_TMA_<tma>_<ctrl>_<date>
        The TIFF images within the Control sets:
            <project>\<ctrlset>\inform_data\Component_Tiffs\*.tif
        TMA core information: 
            <project>\Ctrl\Control_TMA_info.xlsx
        Batch information:
            <project>\Batch\MergeConfig_<batchid>.csv
        
    The results will be written into
        Detailed information by core
            <project>\Ctrl\dbload\project<prno>-cores.csv
            <project>\Ctrl\dbload\project<prno>-fluxes.csv
            <project>\Ctrl\dbload\project<prno>-batch.csv

Analysis:

    The computations will bo done during the MergeDB process,
    executing 
        Q1.1-CtrlTables.sql
        Q1.2-Stats.sql
        Q1.3-Outputs.sql
    and the results will be written into the y:\ctrl<project>\csv
    directory.

Plotting the results:
    The plotAll() command will generate all the diagnostic plots:
        1. rawflux
        2. normFlux
        3. flatFlux
        4. meanFlux
        5. boxplot by batch/tissue
        6. boxplot vs batch


	
		
		

	