# A script to run a quick test run of a flatfield

#imports
from ..flatfield.flatfield_producer import FlatfieldProducer
from ..flatfield.config import FLATFIELD_FILE_NAME_STEM, FILE_EXT, IMG_DTYPE_OUT, SMOOTHED_CORRECTED_MEAN_IMAGE_FILE_NAME_STEM
from ..utilities.img_file_io import getImageHWLFromXMLFile, getRawAsHWL
from ..utilities.misc import cd
import numpy as np
import pathlib, glob, os

#some constants
folder = pathlib.Path(__file__).parent
samp = 'M21_1'
rawfile_top_dir = folder/'data'/'raw'
rawfile_ext = '.Data.dat'
dbload_top_dir = folder/'data'
workingdir_name = 'flatfield_test_for_jenkins'
working_dir = folder/workingdir_name
working_dir.mkdir(exist_ok=True)

sample_names_to_run = [samp]
filepaths_to_run = None
with cd(os.path.join(rawfile_top_dir,samp)) :
	filepaths_to_run = glob.glob(f'*{rawfile_ext}')


    #begin by figuring out the background thresholds per layer by looking at the HPFs on the tissue edges
    if not args.skip_masking :
        if args.threshold_file_dir is not None :
            ff_producer.readInBackgroundThresholds(args.threshold_file_dir)
        elif args.dbload_top_dir is not None :
            ff_producer.findBackgroundThresholds(all_filepaths,args.n_threads)
    if args.mode in ['make_flatfield', 'apply_flatfield'] :
        
        if args.mode=='make_flatfield' :
            #make the flatfield image
            ff_producer.makeFlatField()
        if args.mode=='apply_flatfield' :
            
        #save the plots, etc.
        ff_producer.writeOutInfo()
    flatfield_logger.info('All Done!')

#get the image dimensions from the .xml file
dims=getImageHWLFromXMLFile(rawfile_top_dir,samp)
#make the FlatfieldProducer Object
print('Initializing FlatfieldProducer')
ff_producer = FlatfieldProducer(dims,sample_names_to_run,filepaths_to_run,dbload_top_dir,workingdir_name,False)
#write out the text file of all the raw file paths that will be run
ff_producer.writeFileLog()
#mask and stack images together
ff_producer.stackImages(1,0.0,0,True)
#make the flatfield image
ff_producer.makeFlatField()
#apply the flatfield to the same image stack
ff_producer.applyFlatField(os.path.join(workingdir_name,f'{FLATFIELD_FILE_NAME_STEM}{FILE_EXT}'))
#this should result in 1s everywhere in the smoothed corrected mean image file
scmi=getRawAsHWL(os.path.join(workingdir_name,f'{SMOOTHED_CORRECTED_MEAN_IMAGE_FILE_NAME_STEM}{FILE_EXT}'),dims[0],dims[1],dims[2],IMG_DTYPE_OUT)
np.testing.assert_equal(scmi,np.ones(dims,dtype=IMG_DTYPE_OUT))
print('Done!')
