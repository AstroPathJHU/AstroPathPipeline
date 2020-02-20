#imports
from .warpset import WarpSet
from .alignmentset import AlignmentSet, Rectangle
from .overlap import Overlap
from .readtable import readtable,writetable
import os

#global variables
OVERLAP_FILE_EXT   = '_overlap.csv'
RECTANGLE_FILE_EXT = '_rect.csv'
IM3_EXT='.im3'
RAW_EXT='.raw'
WARP_EXT='.camWarp_layer'

class FittingError(Exception) :
    """
    Class for errors encountered during fitting
    """
    pass

class WarpFitter :
    """
    Main class for fitting a camera matrix and distortion parameters to a set of images based on the results of their alignment
    """
    def __init__(self,samplename,rawfile_dir,metafile_dir,working_dir,overlaps=-1,warp=None,nlayers=35,layers=[1]) :
        """
        samplename   = name of the microscope data sample to fit to ("M21_1" or equivalent)
        rawfile_dir  = path to directory containing multilayered ".raw" files
        metafile_dir = path to directory containing "dbload" metadata files (assuming at least a "rect.csv" and "overlap.csv")
        working_dir  = path to some local directory to store files produced by the WarpFitter
        overlaps     = list of (or two-element tuple of first/last) #s (n) of overlaps to use for evaluating quality of alignment 
                       (default=-1 will use all overlaps)
        warp         = CameraWarp object whose optimal parameters will be determined (if None a new default CameraWarp will be created)
        nlayers      = # of layers in raw images (default=35)
        layers       = list of image layer numbers (indexed starting at 1) to consider in the warping/alignment (default=[1])
        """
        #store the directory paths
        self.samp_name = samplename
        self.rawfile_dir=rawfile_dir
        self.metafile_dir=metafile_dir
        self.working_dir=working_dir
        self.init_dir = os.getcwd()
        #setup the working directory and the lists of overlaps and rectangles
        self.overlaps = None; self.rectangles = None #will be populated by the helper function call on the next line
        self.__setupWorkingDirectory(overlaps)
        #get the list of raw file paths
        self.rawfile_paths = [os.path.join(self.rawfile_dir,fn.replace(IM3_EXT,RAW_EXT)) for fn in [r.file for r in self.rectangles]]
        #make the warpset object to use
        self.warpset = WarpSet(warp,self.rawfile_paths,nlayers,layers)
        #make the alignmentset object to use
        self.alignset = AlignmentSet(os.path.join(*([os.sep]+self.metafile_dir.split(os.sep)[:-2])),self.working_dir,self.samp_name,interactive=True)

    # helper function to create the working directory and create/write out lists of overlaps and rectangles
    def __setupWorkingDirectory(self,overlaps) :
        #read all the overlaps in this sample's metadata and reduce to what will actually be used
        all_overlaps = readtable(os.path.join(self.metafile_dir,self.samp_name+OVERLAP_FILE_EXT),Overlap)
        if overlaps==-1 :
            self.overlaps = all_overlaps 
        elif isinstance(overlaps,tuple) :
            self.overlaps = all_overlaps[overlaps[0]-1:overlaps[1]]
        elif isinstance(overlaps,list) :
            self.overlaps = [all_overlaps[i-1] for i in overlaps]
        else :
            raise FittingError(f'Cannot recognize overlap choice from overlaps={overlaps} (must be a tuple, list, or -1)')
        #read all the rectangles and store the relevant ones
        all_rectangles = readtable(os.path.join(self.metafile_dir,self.samp_name+RECTANGLE_FILE_EXT),Rectangle)
        self.rectangles = [r for r in all_rectangles if r.n in ([o.p1 for o in self.overlaps]+[o.p2 for o in self.overlaps])]
        #create the working directory and write to it the new metadata .csv files
        if not os.path.isdir(self.working_dir) :
            os.mkdir(self.working_dir)
        os.chdir(self.working_dir)
        writetable(os.path.join(self.working_dir,self.samp_name+OVERLAP_FILE_EXT),self.overlaps)
        writetable(os.path.join(self.working_dir,self.samp_name+RECTANGLE_FILE_EXT),self.rectangles)
        os.chdir(self.init_dir)