#imports
from .meanimage import MeanImage
from .rectangle import RectangleCorrectedIm3MultiLayer
from .plotting import plot_tissue_edge_rectangle_locations
from .config import CONST
from ...shared.sample import ReadRectanglesOverlapsIm3FromXML, WorkflowSample
from ...shared.overlap import Overlap
from ...utilities.img_file_io import LayerOffset
from ...utilities.tableio import readtable, writetable
from ...utilities.misc import cd, MetadataSummary
from ...utilities.config import CONST as UNIV_CONST
import numpy as np
import matplotlib.pyplot as plt
import pathlib

#some file-scope constants
DEFAULT_N_THREADS = 10

class MeanImageSample(ReadRectanglesOverlapsIm3FromXML,WorkflowSample) :
    """
    Main class to handle creating the meanimage for a slide
    """

    #################### PUBLIC FUNCTIONS ####################

    def __init__(self,*args,workingdir=pathlib.Path(UNIV_CONST.MEANIMAGE_DIRNAME),et_offset_file=None,skip_masking=False,n_threads=DEFAULT_N_THREADS,**kwargs) :
        #initialize the parent classes
        super().__init__(*args,**kwargs)
        self.__workingdirpath = workingdir
        #if the workingdir arg was just a name, set it to a directory with that name in the default location (im3 subdir)
        if self.__workingdirpath.name==str(self.__workingdirpath) :
            self.__workingdirpath = self.im3folder / workingdir
        self.__workingdirpath.mkdir(parents=True,exist_ok=True)
        #set some other variables
        self.__et_offset_file = et_offset_file
        self.__n_threads = n_threads
        #start up the meanimage
        self.__meanimage = MeanImage(skip_masking)
        #set up the list of output files to add to as the code runs (though some will always be required)
        self.__output_files = []
        self.__output_files.append(self.__workingdirpath / f'{self.SlideID}-{CONST.MEAN_IMAGE_BIN_FILE_NAME_STEM}')
        self.__output_files.append(self.__workingdirpath / f'{self.SlideID}-{CONST.SUM_IMAGES_SQUARED_BIN_FILE_NAME_STEM}')
        self.__output_files.append(self.__workingdirpath / f'{self.SlideID}-{CONST.STD_ERR_OF_MEAN_IMAGE_BIN_FILE_NAME_STEM}')
        self.__output_files.append(self.__workingdirpath / CONST.FIELDS_USED_CSV_FILENAME)
        self.__output_files.append(self.__workingdirpath / CONST.METADATA_SUMMARY_STACKED_IMAGES_CSV_FILENAME)

    def initrectangles(self) :
        """
        Init Rectangles with additional transformations for exposure time differences after getting median exposure times and offsets
        (only if exposure time corrections aren't being skipped)
        """
        super().initrectangles()
        self.__med_ets = None
        if self.__et_offset_file is None :
            return
        #find the median exposure times
        slide_exp_times = np.zeros(shape=(len(self.rectangles),self.nlayers)) 
        for ir,r in enumerate(self.rectangles) :
            slide_exp_times[ir,:] = r.allexposuretimes
        self.__med_ets = np.median(slide_exp_times,axis=0)
        #read the exposure time offsets
        offsets = self.__read_exposure_time_offsets()
        #add the exposure time correction to every rectangle's transformations
        for r in self.rectangles :
            r.add_exposure_time_correction_transformation(self.__med_ets,offsets)

    def run(self) :
        """
        Main "run" function to be looped when entire cohorts are run
        """
        #figure out the background thresholds to use if images will be masked
        background_thresholds = None if self.__meanimage.skip_masking else self.__get_background_thresholds()
        #loop over all rectangle images and add them to the stack
        #create and write out the final mask stack, mean image, and std. error of the mean image

    #################### CLASS VARIABLES + PROPERTIES ####################

    multilayer = True
    rectangletype = RectangleCorrectedIm3MultiLayer
    overlaptype = Overlap
    nclip = UNIV_CONST.N_CLIP
    
    @property
    def inputfiles(self,**kwargs) :
        print(f'RUNNING INPUTFILES')
        return super().inputfiles(**kwargs) + [
            *(r.imagefile for r in self.rectangles),
        ]
    @property
    def getoutputfiles(self) :
        print(f'RUNNING GETOUTPUTFILES')
        return self.__output_files
    @property
    def image_masking_folder(self) :
        self.__workingdirpath / CONST.IMAGE_MASKING_SUBDIR_NAME


    #################### CLASS METHODS ####################

    @classmethod
    def makeargumentparser(cls):
        p = super().makeargumentparser()
        p.add_argument('--filetype',choices=['raw','flatWarp'],default='raw',
                        help=f'Whether to use "raw" files (extension {UNIV_CONST.RAW_EXT}, default) or "flatWarp" files (extension {UNIV_CONST.FLATW_EXT})')
        p.add_argument('--workingdir', type=pathlib.Path, default=pathlib.Path(UNIV_CONST.MEANIMAGE_DIRNAME),
                        help=f'path to the working directory (default: new subdirectory called "{UNIV_CONST.MEANIMAGE_DIRNAME}" in the slide im3 directory)')
        g = p.add_mutually_exclusive_group(required=True)
        g.add_argument('--exposure_time_offset_file',
                        help='''Path to the .csv file specifying layer-dependent exposure time correction offsets for the slides in question
                        [use this argument to apply corrections for differences in image exposure time]''')
        g.add_argument('--skip_exposure_time_correction', action='store_true',
                        help='Add this flag to entirely skip correcting image flux for exposure time differences')
        p.add_argument('--skip_masking', action='store_true',
                        help='''Add this flag to entirely skip masking out the background regions of the images as they get added
                        [use this argument to completely skip the background thresholding and masking]''')
        p.add_argument('--n_threads', type=int, default=DEFAULT_N_THREADS,
                        help=f'Number of threads to use for parallelized portions of the code (default={DEFAULT_N_THREADS})')
        return p
    @classmethod
    def initkwargsfromargumentparser(cls, parsed_args_dict):
        return {
            **super().initkwargsfromargumentparser(parsed_args_dict),
            'filetype': parsed_args_dict.pop('filetype'), 
            'workingdir': parsed_args_dict.pop('workingdir'),
            'et_offset_file': None if parsed_args_dict.pop('skip_exposure_time_correction') else parsed_args_dict.pop('exposure_time_offset_file'),
            'skip_masking': parsed_args_dict.pop('skip_masking'),
            'n_threads': parsed_args_dict.pop('n_threads'),
        }
    @classmethod
    def defaultunits(cls) :
        return "fast"
    @classmethod
    def logmodule(cls) : 
        return "meanimage"
    @classmethod
    def workflowdependencyclasses(cls):
        return super().workflowdependencyclasses()

    #################### PRIVATE HELPER FUNCTIONS ####################

    def __read_exposure_time_offsets(self) :
        """
        Read in the offset factors for exposure time corrections from the file defined by command line args
        """
        self.logger.info(f'Copying exposure time offsets for {self.SlideID} from file {self.__et_offset_file}')
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

    def __get_background_thresholds(self) :
        """
        Return the background thresholds for each image layer, either reading them from an existing file 
        or calculating them from the rectangles on the edges of the tissue
        """
        #first check the working directory for the background threshold file
        threshold_file_path = self.__workingdirpath / f'{self.SlideID}-{CONST.BACKGROUND_THRESHOLD_TEXT_FILE_NAME_STEM}'
        #if it's not in the working directory, check in the slide's meanimage directory (if it exists)
        if not threshold_file_path.is_file() :
            other_threshold_file_path = self.root / f'{self.SlideID}' / f'{UNIV_CONST.IM3_DIR_NAME}' / f'{UNIV_CONST.MEANIMAGE_DIRNAME}' 
            other_threshold_file_path = other_threshold_file_path / f'{self.SlideID}-{CONST.BACKGROUND_THRESHOLD_TEXT_FILE_NAME_STEM}'
            if other_threshold_file_path.is_file() :
                threshold_file_path = other_threshold_file_path
        #read the values from the files or find them from the tissue edge rectangles
        if threshold_file_path.is_file() :
            return self.__get_background_thresholds_from_file(threshold_file_path)
        else :
            return self.__find_background_thresholds_from_tissue_edge_images(threshold_file_path)

    def __get_background_thresholds_from_file(self,threshold_file_path) :
        """
        Return the list of background thresholds found in a given file
        """
        background_thresholds_to_return = []
        self.logger.info(f'Reading background thresholds for {self.SlideID} from file {threshold_file_path}')
        with open(threshold_file_path,'r') as tfp :
            all_lines = [l.rstrip() for l in tfp.readlines()]
            for line in all_lines :
                try :
                    background_thresholds_to_return.append(int(line))
                except ValueError :
                    pass
        if not len(background_thresholds_to_return)==self.nlayers :
            errmsg = f'ERROR: number of background thresholds read from {threshold_file_path} is not equal to the number of image layers!'
            errmsg+= f' (read {len(background_thresholds_to_return)} values but there are {self.nlayers} image layers)'
            raise ValueError(errmsg)
        return background_thresholds_to_return

    def __find_background_thresholds_from_tissue_edge_images(self,threshold_file_path) :
        """
        Find, write out, and return the list of optimal background thresholds found from the set of images located on the edges of the tissue
        """
        #add the output file(s) that will be created to the list
        self.__output_files.append(self.__workingdirpath / f'{self.SlideID}-{CONST.BACKGROUND_THRESHOLD_TEXT_FILE_NAME_STEM}')
        self.__output_files.append(self.__workingdirpath / f'{self.SlideID}-{CONST.METADATA_SUMMARY_THRESHOLDING_IMAGES_CSV_FILENAME}')
        #get the list of rectangles that are on the edge of the tissue, plot their locations, and save a summary of their metadata
        tissue_edge_rects = [r for r in self.rectangles if len(self.overlapsforrectangle(r.n))<8]
        plot_tissue_edge_rectangle_locations(self.rectangles,tissue_edge_rects,self.root,self.SlideID,
                                             self.__workingdirpath / CONST.THRESHOLDING_SUMMARY_PDF_FILENAME.replace('.pdf','_plots'))
        edge_rect_ts = [r.t for r in tissue_edge_rects]
        with cd(self.__workingdirpath) :
            writetable(self.__workingdirpath / f'{self.SlideID}-{CONST.METADATA_SUMMARY_THRESHOLDING_IMAGES_CSV_FILENAME}',
                      [MetadataSummary(self.SlideID,self.Project,self.Cohort,self.microscopename,str(min(edge_rect_ts)),str(max(edge_rect_ts)))])
        #find the optimal thresholds for each tissue edge image and make plots of the thresholds found in each layer

        #write out the data table of all the thresholds found
        #make some plots and collect them in a .pdf file
        #return the list of background thresholds

#################### FILE-SCOPE FUNCTIONS ####################

def main(args=None) :
    MeanImageSample.runfromargumentparser(args)

if __name__=='__main__' :
    main()