#imports
from .config import CONST
from ...shared.sample import ReadRectanglesIm3FromXML, WorkflowSample
from ...utilities.config import CONST as UNIV_CONST
import pathlib

#some file-scope constants

class MeanImageSample(ReadRectanglesIm3FromXML,WorkflowSample) :
    """
    Main class to handle creating the meanimage for a slide
    """

    #################### PUBLIC FUNCTIONS ####################

    def __init__(self,*args,workingdir=pathlib.Path(CONST.MEANIMAGE_DIRNAME),et_offset_file=None,skip_masking=False,**kwargs) :
        super().__init__(*args,**kwargs)
        #set the path to the working directory based on a kwarg
        self._workingdirpath = workingdir
        #if it was just a name, set it to a directory with that name in the default location (im3 subdir)
        if self._workingdirpath.name==str(self._workingdirpath) :
            self._workingdirpath = self.im3folder / workingdir
        self._workingdirpath.mkdir(parents=True,exist_ok=True)
        #set some other runmode variables
        self._et_offset_file = et_offset_file
        self._skip_masking = skip_masking
        self.image_masking_folder = self._workingdirpath / CONST.IMAGE_MASKING_SUBDIR_NAME if not self._skip_masking else None

    #################### PROPERTIES ####################

    @property
    def inputfiles(self,**kwargs) :
        print('RUNNING INPUTFILES')
        return super().inputfiles(**kwargs) + [
            *(r.imagefile for r in self.rectangles),
        ]
    @property
    def getoutputfiles(self) :
        print('RUNNING GETOUTPUTFILES')
        output_files = []
        output_files.append(self._workingdirpath / CONST.FIELDS_USED_CSV_FILENAME)
        output_files.append(self._workingdirpath / CONST.METADATA_SUMMARY_STACKED_IMAGES_CSV_FILENAME)
        return output_files


    #################### CLASS METHODS ####################

    @classmethod
    def makeargumentparser(cls):
        p = super().makeargumentparser()
        p.add_argument('--filetype',choices=['raw','flatWarp'],default='raw',
                        help=f'Whether to use "raw" files (extension {UNIV_CONST.RAW_EXT}, default) or "flatWarp" files (extension {UNIV_CONST.FLATW_EXT})')
        p.add_argument('--workingdir', type=pathlib.Path, default=pathlib.Path(CONST.MEANIMAGE_DIRNAME),
                        help=f'path to the working directory (default: new subdirectory called "{CONST.MEANIMAGE_DIRNAME}" in the slide im3 directory)')
        g = p.add_mutually_exclusive_group(required=True)
        g.add_argument('--exposure_time_offset_file',
                        help='''Path to the .csv file specifying layer-dependent exposure time correction offsets for the slides in question
                        [use this argument to apply corrections for differences in image exposure time]''')
        g.add_argument('--skip_exposure_time_correction', action='store_true',
                        help='Add this flag to entirely skip correcting image flux for exposure time differences')
        p.add_argument('--skip_masking', action='store_true',
                        help='''Add this flag to entirely skip masking out the background regions of the images as they get added
                        [use this argument to completely skip the background thresholding and masking]''')
        return p
    @classmethod
    def initkwargsfromargumentparser(cls, parsed_args_dict):
        return {
            **super().initkwargsfromargumentparser(parsed_args_dict),
            'filetype': parsed_args_dict.pop('filetype'), 
            'workingdir': parsed_args_dict.pop('workingdir'),
            'et_offset_file': None if parsed_args_dict.pop('skip_exposure_time_correction') else parsed_args_dict.pop('exposure_time_offset_file'),
            'skip_masking': parsed_args_dict.pop('skip_masking')
        }
    @classmethod
    def defaultunits(cls) :
        return "fast"
    @classmethod
    def logmodule(cls) : 
        return "meanimage"
    @classmethod
    def workflowdependencies(cls) :
        return super().workflowdependencies()

#################### FILE-SCOPE FUNCTIONS ####################

def main(args=None) :
    MeanImageSample.runfromargumentparser(args)

if __name__=='__main__' :
    main()