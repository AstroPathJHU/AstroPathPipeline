#imports
from .warpset import WarpSet
from .alignmentset import AlignmentSet, Rectangle
from .overlap import Overlap
from .tableio import readtable,writetable
import numpy as np, scipy
import os, logging

#global variables
OVERLAP_FILE_EXT   = '_overlap.csv'
RECTANGLE_FILE_EXT = '_rect.csv'
IM3_EXT='.im3'
RAW_EXT='.raw'
WARP_EXT='.camWarp_layer'
DEF_PAR_BOUNDS = {
    'cx':(400,945),
    'cy':(300,710),
    'fx':(39600,40400),
    'fy':(39600,40400),
    'k1':(-75.,75.),
    'k2':(-150000.,150000.),
    'p1':(-0.75,0.75),
    'p2':(-0.6,0.6)
}

class FittingError(Exception) :
    """
    Class for errors encountered during fitting
    """
    pass

class WarpFitter :
    """
    Main class for fitting a camera matrix and distortion parameters to a set of images based on the results of their alignment
    """
    def __init__(self,samplename,rawfile_dir,metafile_dir,working_dir,overlaps=-1,warp=None,n=1344,m=1004,nlayers=35,layers=[1]) :
        """
        samplename   = name of the microscope data sample to fit to ("M21_1" or equivalent)
        rawfile_dir  = path to directory containing multilayered ".raw" files
        metafile_dir = path to directory containing "dbload" metadata files (assuming at least a "rect.csv" and "overlap.csv")
        working_dir  = path to some local directory to store files produced by the WarpFitter
        overlaps     = list of (or two-element tuple of first/last) #s (n) of overlaps to use for evaluating quality of alignment 
                       (default=-1 will use all overlaps)
        warp         = CameraWarp object whose optimal parameters will be determined (if None a new default CameraWarp will be created)
        n            = image width (pixels)
        m            = image height (pixels)
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
        self.overlaps, self.rectangles = self.__setupWorkingDirectory(overlaps)
        #get the list of raw file paths
        self.rawfile_paths = [os.path.join(self.rawfile_dir,fn.replace(IM3_EXT,RAW_EXT)) for fn in [r.file for r in self.rectangles]]
        #make the warpset object to use
        if warp is not None :
            self.warpset = WarpSet(warp=warp,rawfiles=self.rawfile_paths,nlayers=nlayers,layers=layers)
        else :
            self.warpset = WarpSet(n=n,m=m,rawfiles=self.rawfile_paths,nlayers=nlayers,layers=layers)
        #make the alignmentset object to use
        self.alignset = self.__initializeAlignmentSet()

    #################### PUBLIC FUNCTIONS ####################

    def loadRawFiles(self) :
        """
        Load the raw files into the warpset, warp/save them, and load them into the alignment set 
        """
        self.warpset.loadRawImageSet(self.rawfile_paths)
        self.warpset.warpLoadedImageSet()
        os.chdir(self.working_dir)
        if not os.path.isdir(os.path.join(self.working_dir,self.samp_name)) :
            os.mkdir(self.samp_name)
        os.chdir(os.path.join(self.working_dir,self.samp_name))
        self.warpset.writeOutWarpedImageSet()
        os.chdir(self.init_dir)
        self.alignset.getDAPI(filetype='camWarpDAPI')

    def doFit(self,par_bounds=None,fix_cxcy=False,fix_fxfy=False,fix_k1k2=False,fix_p1p2=False,max_radial_warp=25.,max_tangential_warp=25.) :
        """
        Fit the cameraWarp model to the loaded dataset
        par_bounds = dictionary of alternate parameter bounds for differential_evolution keyed by name ('cx','cy','fx','fy','k1','k2','p1','p2')
                     can be some parameters or all, will just overwrite the defaults with those supplied
        fix_*      = set True to fix groups of parameters
        max_*_warp = values to use for max warp amount constraints (set to -1 to remove constraints)
        """
        #make the iteration counter and the lists of costs/warp amounts
        self.minfunc_calls=0
        self.costs=[]
        self.max_radial_warps=[]
        self.max_tangential_warps=[]
        #silence the AlignmentSet logger
        logger = logging.getLogger('align')
        logger.setLevel(logging.WARN)
        #build the list of parameter bounds
        parameter_bounds = self.__getParameterBoundsList(par_bounds,fix_cxcy,fix_fxfy,fix_k1k2,fix_p1p2)
        print(len(parameter_bounds))
        #get the list of constraints
        constraints = self.__getConstraints(fix_k1k2,fix_p1p2,max_radial_warp,max_tangential_warp)
        #get the list to use to mask fixed parameters in the minimization functions
        self.par_mask = self.__getParameterMask(fix_cxcy,fix_fxfy,fix_k1k2,fix_p1p2)
        #call differential_evolution
        result=scipy.optimize.differential_evolution(
            self._evalCamWarpOnAlignmentSet,
            parameter_bounds,
            constraints=constraints
            )
        #calculate/print results
        print(result)

    #################### FUNCTIONS FOR USE WITH MINIMIZATION ####################

    # !!!!!! For the time being, these functions don't correctly describe dependence on k3, k4, k5, or k6 !!!!!!

    def _evalCamWarpOnAlignmentSet(self,pars) :
        return 1
        #self.minfunc_calls+=1
        ##first fix the parameter list so the warp functions always see vectors of the same length
        ##pars = rescalePars(pars)
        ##update the warp with the new parameters
        ##fixedpars = [p for p in pars]+[0.,0.]
        ##fixedpars = [p for p in pars[:2]]+[40000.,40000.]+[p for p in pars[-2:]]+[0.,0.]
        #warpset.updateCameraParams(pars)
        ##warpset.updateCameraParams(fixedpars)
        #warpset.warp.printParams()
        ##then warp the images
        #os.chdir(warpedfile_dir)
        #warpset.warpLoadedImageSet()
        #os.chdir(init_dir)
        ##reload the (newly-warped) images into the alignment set
        #alignSet.updateRectangleImages(warpset.warped_images,'.raw')
        ##align the images 
        #cost = alignSet.align()
        ##add to the lists to plot
        #costs.append(cost)
        #rad_warps.append(warpset.warp.maxRadialDistortAmount(pars))
        #tan_warps.append(warpset.warp.maxTangentialDistortAmount(pars))
        ##print/return the cost from the alignment
        ##print(f'  Call {fcall} cost={cost} (radial warp={rad_warps[-1]:.02f}, tangential warp={tan_warps[-1]:.02f})')
        #print(f'  Cost={cost} (radial warp={rad_warps[-1]:.02f}, tangential warp={tan_warps[-1]:.02f})')
        #return cost

    def _maxRadialDistortAmountJacobianForConstraint(self,pars) :
        warpresult = np.array(self.warpset.warp.maxRadialDistortAmountJacobian(pars))
        return (warpresult[self.par_mask]).tolist()

    def _maxTangentialDistortAmountJacobianForConstraint(self) :
        warpresult = np.array(self.warpset.warp.maxTangentialDistortAmountJacobian(pars))
        return (warpresult[self.par_mask]).tolist()

    #################### PRIVATE HELPER FUNCTIONS ####################

    # helper function to create the working directory and create/write out lists of overlaps and rectangles
    def __setupWorkingDirectory(self,overlaps) :
        #read all the overlaps in this sample's metadata and reduce to what will actually be used
        all_overlaps = readtable(os.path.join(self.metafile_dir,self.samp_name+OVERLAP_FILE_EXT),Overlap)
        if overlaps==-1 :
            olaps = all_overlaps 
        elif isinstance(overlaps,tuple) :
            olaps = all_overlaps[overlaps[0]-1:overlaps[1]]
        elif isinstance(overlaps,list) :
            olaps = [all_overlaps[i-1] for i in overlaps]
        else :
            raise FittingError(f'Cannot recognize overlap choice from overlaps={overlaps} (must be a tuple, list, or -1)')
        #read all the rectangles and store the relevant ones
        all_rectangles = readtable(os.path.join(self.metafile_dir,self.samp_name+RECTANGLE_FILE_EXT),Rectangle)
        rects = [r for r in all_rectangles if r.n in ([o.p1 for o in olaps]+[o.p2 for o in olaps])]
        #create the working directory and write to it the new metadata .csv files
        if not os.path.isdir(self.working_dir) :
            os.mkdir(self.working_dir)
        os.chdir(self.working_dir)
        writetable(os.path.join(self.working_dir,self.samp_name+OVERLAP_FILE_EXT),olaps)
        writetable(os.path.join(self.working_dir,self.samp_name+RECTANGLE_FILE_EXT),rects)
        os.chdir(self.init_dir)
        return olaps, rects

    # helper function to create and return a new alignmentSet object that's set up to run on the identified set of images/overlaps
    def __initializeAlignmentSet(self) :
        a = AlignmentSet(os.path.join(*([os.sep]+self.metafile_dir.split(os.sep)[:-2])),self.working_dir,self.samp_name,interactive=True)
        a.overlaps=self.overlaps
        a.rectangles=self.rectangles
        return a

    #helper function to make the list of parameter bounds for fitting
    def __getParameterBoundsList(self,par_bounds,fix_cxcy,fix_fxfy,fix_k1k2,fix_p1p2) :
        parorder=['cx','cy','fx','fy','k1','k2','p1','p2']
        #overwrite the default with anything that was supplied
        bounds_dict = DEF_PAR_BOUNDS
        if par_bounds is not None :
            for name in par_bounds.keys() :
                if name in DEF_PAR_BOUNDS.keys() :
                    bounds_dict[name] = par_bounds[name]
                else :
                    raise FittingError(f'Parameter "{name}"" in supplied dictionary of bounds not recognized!')
        #remove any parameters that will be fixed
        to_remove = []
        if fix_cxcy :
            to_remove+=['cx','cy']
        if fix_fxfy :
            to_remove+=['fx','fy']
        if fix_k1k2 :
            to_remove+=['k1','k2']
        if fix_p1p2 :
            to_remove+=['p1','p2']
        for parkey in to_remove :
            if parkey in bounds_dict.keys() :
                del bounds_dict[parkey]
        #print info about the parameters that will be used
        fixed_par_string=''
        for name in to_remove :
            fixed_par_string+=name+', '
        print(f'Will fit with {len(bounds_dict.keys())} parameters ({fixed_par_string[:-2]} fixed).')
        #return the ordered list of parameters
        return [bounds_dict[name] for name in parorder if name in bounds_dict.keys()]

    #helper function to make the constraints
    def __getConstraints(self,fix_k1k2,fix_p1p2,max_radial_warp,max_tangential_warp) :
        constraints = []
        names_to_print = []
        #if k1 and k2 are being fit for, and max_radial_warp is defined, add the max_radial_warp constraint
        if (not fix_k1k2) and max_radial_warp!=-1 :
            constraints.append(scipy.optimize.NonlinearConstraint(
                self.warpset.warp.maxRadialDistortAmount,
                -1.*max_radial_warp,
                max_radial_warp,
                jac=self._maxRadialDistortAmountJacobianForConstraint,
                hess=scipy.optimize.BFGS()
                )
            )
            names_to_print.append(f'max radial warp={max_radial_warp} pixels')
        #if k1 and k2 are being fit for, and max_tangential_warp is defined, add the max_tangential_warp constraint
        if (not fix_p1p2) and max_tangential_warp!=-1 :
            constraints.append(scipy.optimize.NonlinearConstraint(
                self.warpset.warp.maxTangentialDistortAmount,
                -1.*max_tangential_warp,
                max_tangential_warp,
                jac=self._maxTangentialDistortAmountJacobianForConstraint,
                hess=scipy.optimize.BFGS()
                )
            )
            names_to_print.append(f'max tangential warp={max_tangential_warp} pixels')
        #print the information about the constraints
        if len(constraints)==0 :
            print('No constraints will be applied')
            return ()
        else :
            constraintstring = 'Will apply constraints: '
            for ntp in names_to_print :
                constraintstring+=ntp+', '
            print(constraintstring[:-2]+'.')
        #return the list of constraints
        if len(constraints)==1 : #if there's only one it doesn't get passed as a list
            return constraints[0]
        else :
            return constraints

    #helper function to get a masking list for the parameters from the warp functions to deal with fixed parameters
    def __getParameterMask(self,fix_cxcy,fix_fxfy,fix_k1k2,fix_p1p2) :
        mask = [True,True,True,True,True,True,True,True]
        if fix_cxcy :
            mask[0]=False
            mask[1]=False
        if fix_fxfy :
            mask[2]=False
            mask[3]=False
        if fix_k1k2 :
            mask[4]=False
            mask[5]=False
        if fix_p1p2 :
            mask[6]=False
            mask[7]=False
        return mask
