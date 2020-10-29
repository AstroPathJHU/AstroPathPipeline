# A script to run a quick test run of a flatfield

#imports
from ..flatfield.flatfield_producer import FlatfieldProducer
from ..flatfield.utilities import flatfield_logger, FlatfieldSlideInfo
from ..flatfield.config import CONST
from ..utilities.misc import cd
import pathlib, glob, os, shutil

#some constants
folder = pathlib.Path(__file__).parent
slide_ID = 'M21_1'
rawfile_top_dir = folder/'data'/'raw'
rawfile_ext = '.Data.dat'
root_dir = folder/'data'
workingdir_name = 'flatfield_test_for_jenkins'
working_dir = folder/workingdir_name
working_dir.mkdir(exist_ok=True)

slides_to_run = [FlatfieldSlideInfo(slide_ID,rawfile_top_dir,root_dir)]
filepaths_to_run = None
with cd(os.path.join(rawfile_top_dir,slide_ID)) :
	filepaths_to_run = [os.path.join(rawfile_top_dir,slide_ID,fn) for fn in glob.glob(f'*{rawfile_ext}')]

flatfield_logger.info('Starting test run....')
#make the FlatfieldProducer Object
ff_producer = FlatfieldProducer(slides_to_run,filepaths_to_run,working_dir,True,True)
#write out the text file of all the raw file paths that will be run
ff_producer.writeFileLog('filepath_log.txt')
#mask and stack images together
ff_producer.stackImages(1,0.0,0,True)
#make the flatfield image
ff_producer.makeFlatField()
#save the plots, etc.
ff_producer.writeOutInfo()
#apply the flatfield to the same image stack
ff_producer.applyFlatField(os.path.join(working_dir,f'{CONST.FLATFIELD_FILE_NAME_STEM}_{os.path.basename(os.path.normpath(working_dir))}{CONST.FILE_EXT}'))
#remove what was made
flatfield_logger.info('Removing working directory....')
shutil.rmtree(working_dir,ignore_errors=True)
flatfield_logger.info('All Done!')
