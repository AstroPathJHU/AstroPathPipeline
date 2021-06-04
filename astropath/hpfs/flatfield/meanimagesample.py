#imports
from .rectangle import RectangleCorrectedIm3MultiLayer
from .config import CONST
from ...shared.sample import ReadRectanglesIm3FromXML, WorkflowSample
from ...utilities.img_file_io import LayerOffset
from ...utilities.tableio import readtable
from ...utilities.config import CONST as UNIV_CONST
import numpy as np
import matplotlib.pyplot as plt
import pathlib

#some file-scope constants

class MeanImageSample(ReadRectanglesIm3FromXML,WorkflowSample) :
    """
    Main class to handle creating the meanimage for a slide
    """

    #################### PUBLIC FUNCTIONS ####################

    def __init__(self,*args,workingdir=pathlib.Path(CONST.MEANIMAGE_DIRNAME),et_offset_file=None,skip_masking=False,**kwargs) :
        #initialize the parent classes
        super().__init__(*args,**kwargs)
        #set some runmode variables
        self.__et_offset_file = et_offset_file
        self.__skip_masking = skip_masking
        #set the path to the working directory based on a kwarg
        self.__workingdirpath = workingdir
        #if it was just a name, set it to a directory with that name in the default location (im3 subdir)
        if self.__workingdirpath.name==str(self.__workingdirpath) :
            self.__workingdirpath = self.im3folder / workingdir
        self.__workingdirpath.mkdir(parents=True,exist_ok=True)

    def initrectangles(self) :
        """
        Init Rectangles with additional transformations for exposure time differences after getting median exposure times and offsets
        """
        super().initrectangles()
        #find the median exposure times
        slide_exp_times = np.zeros(shape=(len(self.rectangles),self.nlayers)) 
        for ir,r in enumerate(self.rectangles) :
            slide_exp_times[ir,:] = r.allexposuretimes
        med_ets = np.median(slide_exp_times,axis=0)
        #read the exposure time offsets
        offsets = self.__read_exposure_time_offsets()
        #add the exposure time correction to every rectangle's transformations
        for r in self.rectangles :
            r.add_exposure_time_correction_transformation(med_ets,offsets)

    def run(self) :
        """
        Main "run" function to be looped when entire cohorts are run
        """
        r = self.rectangles[0]
        with r.using_image() as im :
            print(f'rectangle {r.n} shape = {im.shape} : )')
            #plt.imshow(im[0,:,:])
            #plt.show()
        print('explicitly exited context')

    #################### CLASS VARIABLES + PROPERTIES ####################

    multilayer = True
    rectangletype = RectangleCorrectedIm3MultiLayer
    
    @property
    def inputfiles(self,**kwargs) :
        print(f'RUNNING INPUTFILES')
        return super().inputfiles(**kwargs) + [
            *(r.imagefile for r in self.rectangles),
        ]
    @property
    def getoutputfiles(self) :
        print(f'RUNNING GETOUTPUTFILES')
        output_files = []
        output_files.append(self.__workingdirpath / CONST.FIELDS_USED_CSV_FILENAME)
        output_files.append(self.__workingdirpath / CONST.METADATA_SUMMARY_STACKED_IMAGES_CSV_FILENAME)
        return output_files
    @property
    def image_masking_folder(self) :
        self.__workingdirpath / CONST.IMAGE_MASKING_SUBDIR_NAME if not self.__skip_masking else None


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

    #################### PRIVATE HELPER FUNCTIONS ####################

    def __read_exposure_time_offsets(self) :
        """
        Read in the offset factors for exposure time corrections from the file defined by command line args
        """
        self.logger.info(f'Copying exposure time offsets from file {self.__et_offset_file}')
        layer_offsets_from_file = readtable(self.__et_offset_file,LayerOffset)
        offsets_to_return = []
        for ln in range(1,self.nlayers+1) :
            this_layer_offset = [lo.offset for lo in layer_offsets_from_file if lo.layer_n==ln]
            if len(this_layer_offset)==1 :
                offsets_to_return.append(this_layer_offset[0])
            elif len(this_layer_offset)==0 :
                warnmsg = f'WARNING: LayerOffset file {self.__et_offset_file} does not have an entry for layer {ln}'
                warnmsg+=  ', so that offset will be set to zero!'
                self.logger.warning(warnmsg)
                offsets_to_return.append(0)
            else :
                raise ValueError(f'ERROR: more than one entry found in LayerOffset file {self.__et_offset_file} for layer {ln}!')
        return offsets_to_return

#################### FILE-SCOPE FUNCTIONS ####################

def main(args=None) :
    MeanImageSample.runfromargumentparser(args)

if __name__=='__main__' :
    main()