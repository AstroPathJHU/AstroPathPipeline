# A script to run a quick test run of a flatfield

#imports
from ..flatfield.flatfield_producer import FlatfieldProducer
from ..flatfield.utilities import flatfield_logger
#from ..flatfield.config import CONST
from ..utilities.img_file_io import getImageHWLFromXMLFile
from ..utilities.misc import cd
import pathlib, glob, os, shutil

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
	filepaths_to_run = [os.path.join(rawfile_top_dir,samp,fn) for fn in glob.glob(f'*{rawfile_ext}')]

flatfield_logger.info('Starting test run....')
#get the image dimensions from the .xml file
dims=getImageHWLFromXMLFile(rawfile_top_dir,samp)
#make the FlatfieldProducer Object
ff_producer = FlatfieldProducer(dims,sample_names_to_run,filepaths_to_run,dbload_top_dir,working_dir,False)
#write out the text file of all the raw file paths that will be run
ff_producer.writeFileLog('filepath_log.txt')
##mask and stack images together
#ff_producer.stackImages(1,0.0,0,True)
##make the flatfield image
#ff_producer.makeFlatField()
##save the plots, etc.
#ff_producer.writeOutInfo()
##apply the flatfield to the same image stack
#ff_producer.applyFlatField(os.path.join(working_dir,f'{CONST.FLATFIELD_FILE_NAME_STEM}{CONST.FILE_EXT}'))
#remove what was made
flatfield_logger.info('Removing working directory....')
shutil.rmtree(working_dir,ignore_errors=True)
flatfield_logger.info('All Done!')
