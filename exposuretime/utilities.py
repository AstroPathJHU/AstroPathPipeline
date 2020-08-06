#imports
import numpy as np, matplotlib.pyplot as plt
import os, logging, dataclasses, cv2

#set up the logger
et_fit_logger = logging.getLogger("exposure_time_fitter")
et_fit_logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(message)s    [%(funcName)s, %(asctime)s]"))
et_fit_logger.addHandler(handler)

#helper function to make sure necessary directories exist and that other arguments are valid
def checkArgs(args) :
    #rawfile_top_dir/[sample] must exist
    rawfile_dir = os.path.join(args.rawfile_top_dir,args.sample)
    if not os.path.isdir(rawfile_dir) :
        raise ValueError(f'ERROR: rawfile directory {rawfile_dir} does not exist!')
    #metadata top dir must exist
    if not os.path.isdir(args.metadata_top_dir) :
        raise ValueError(f'ERROR: metadata_top_dir argument ({args.metadata_top_dir}) does not point to a valid directory!')
    #metadata top dir dir must be usable to find a metafile directory
    metafile_dir = os.path.join(args.metadata_top_dir,args.sample,'im3','xml')
    if not os.path.isdir(metafile_dir) :
        raise ValueError(f'ERROR: metadata_top_dir ({args.metadata_top_dir}) does not contain "[sample name]/im3/xml" subdirectories!')
    #make sure the flatfield file exists
    if not os.path.isfile(args.flatfield_file) :
        raise ValueError(f'ERROR: flatfield_file ({args.flatfield_file}) does not exist!')
    #create the working directory if it doesn't already exist
    if not os.path.isdir(args.workingdir_name) :
        os.mkdir(args.workingdir_name)
    #make sure the layers argument makes sense
    if len(args.layers)<1 :
    	raise ValueError(f'ERROR: layers argument {args.layers} must have at least one layer number (or -1)!')
    #make sure the overlaps argument makes sense
    if len(args.overlaps)<1 :
        raise ValueError(f'ERROR: overlaps argument {args.overlaps} must have at least one overlap number (or -1)!')

#helper class to hold a rectangle's rawfile key, raw image, and index in a list of Rectangles 
@dataclasses.dataclass(eq=False, repr=False)
class UpdateImage :
    rawfile_key          : str
    raw_image            : cv2.UMat
    rectangle_list_index : int

#helper class to hold the pre- and post-fit details of overlaps
@dataclasses.dataclass
class OverlapFitResult :
    n            : int
    p1           : int
    p2           : int
    tag          : int
    p1_et        : float
    p2_et        : float
    et_diff      : float
    npix         : int
    prefit_cost  : float
    postfit_cost : float

@dataclasses.dataclass
class LayerOffset :
    layer_n    : int
    offset     : float
    final_cost : float

#helper class for comparing overlap image exposure times
class OverlapWithExposureTimes :

    def __init__(self,olap,p1et,p2et,max_exp_time,cutimages,raw_p1im=None,raw_p2im=None) :
        self.olap = olap
        self.p1et = p1et
        self.p2et = p2et
        self.et_diff = self.p2et-self.p1et
        self.max_exp_time = max_exp_time
        whole_p1_im, whole_p2_im = self.olap.shifted
        w=min(whole_p1_im.shape[1],whole_p2_im.shape[1])
        h=min(whole_p1_im.shape[0],whole_p2_im.shape[0])
        SLICES = {1:np.index_exp[:int(0.5*h),:int(0.5*w)],
                  2:np.index_exp[:int(0.5*h),int(0.25*w):int(0.75*w)],
                  3:np.index_exp[:int(0.5*h),int(0.5*w):],
                  4:np.index_exp[int(0.25*h):int(0.75*h),:int(0.5*w)],
                  6:np.index_exp[int(0.25*h):int(0.75*h),int(0.5*w):],
                  7:np.index_exp[int(0.5*h):,:int(0.5*w)],
                  8:np.index_exp[int(0.5*h):,int(0.25*w):int(0.75*w)],
                  9:np.index_exp[int(0.5*h):,int(0.5*w):]
                }
        if cutimages :
            self.p1_im = whole_p1_im[SLICES[self.olap.tag]]
            self.p2_im = whole_p2_im[SLICES[self.olap.tag]]
        else :
            self.p1_im = whole_p1_im
            self.p2_im = whole_p2_im
        self.raw_p1im = raw_p1im
        self.raw_p2im = raw_p2im
        self.npix = self.p1_im.shape[0]*self.p1_im.shape[1]
        self.raw_npix = self.raw_p1im.shape[0]*self.raw_p1im.shape[1]

    def getCostAndNPix(self,offset,raw=False) :
        if offset<0 : #then don't correct the images
            corr_p1 = self.raw_p1im if raw else self.p1_im
            corr_p2 = self.raw_p2im if raw else self.p2_im
        else :
            corr_p1, corr_p2 = self.__getCorrectedImages(offset,raw)
        npix = self.raw_npix if raw else self.npix
        return np.sum(np.abs(corr_p2-corr_p1)), npix

    def getFitResult(self,best_fit_offset) :
        oc,onp = self.getCostAndNPix(-1)
        self.orig_cost = oc/onp
        cc,cnp = self.getCostAndNPix(best_fit_offset)
        self.corr_cost = cc/cnp
        return OverlapFitResult(self.olap.n,self.olap.p1,self.olap.p2,self.olap.tag,self.p1et,self.p2et,self.et_diff,self.npix,self.orig_cost,self.corr_cost)

    def saveComparisonImages(self,best_fit_offset,filename_stem) :
        orig_overlay = np.clip(np.array([self.p1_im, self.p2_im, 0.5*(self.p1_im+self.p2_im)]).transpose(1, 2, 0) / 1000., 0., 1.)
        corr_p1im, corr_p2im = self.__getCorrectedImages(best_fit_offset,False)
        corr_overlay = np.clip(np.array([corr_p1im, corr_p2im, 0.5*(corr_p1im+corr_p2im)]).transpose(1, 2, 0) / 1000., 0., 1.)
        f,ax = plt.subplots(1,2,figsize=(2*6.4,4.6))
        ax[0].imshow(orig_overlay)
        ax[0].set_title(f'overlap {self.olap.n} original (cost={self.orig_cost:.3f})')
        ax[1].imshow(corr_overlay)
        ax[1].set_title(f'overlap {self.olap.n} corrected (cost={self.corr_cost:.3f})')
        plt.savefig(f'{filename_stem}_offset={best_fit_offset:.3f}_clipped_and_smoothed.png')
        plt.close()
        if self.raw_p1im is not None and self.raw_p2im is not None :
            orig_overlay = np.clip(np.array([self.raw_p1im, self.raw_p2im, 0.5*(self.raw_p1im+self.raw_p2im)]).transpose(1, 2, 0) / 1000.,0.,1.)
            corr_p1im, corr_p2im = self.__getCorrectedImages(best_fit_offset,True)
            corr_overlay = np.clip(np.array([corr_p1im, corr_p2im, 0.5*(corr_p1im+corr_p2im)]).transpose(1, 2, 0) / 1000.,0.,1.)
            f,ax = plt.subplots(1,2,figsize=(2*6.4,4.6))
            ax[0].imshow(orig_overlay)
            ax[0].set_title(f'raw overlap {self.olap.n} original')
            ax[1].imshow(corr_overlay)
            ax[1].set_title(f'raw overlap {self.olap.n} corrected')
            plt.savefig(f'{filename_stem}_offset={best_fit_offset:.3f}_raw_and_whole.png')
            plt.close()

    def __getCorrectedImages(self,offset,raw) :
        if raw and (self.raw_p1im is not None) and (self.raw_p2im is not None) :
            corr_p1 = np.where((self.raw_p1im-offset)>0,offset+(1.*self.max_exp_time/self.p1et)*(self.raw_p1im-offset),self.raw_p1im)
            corr_p2 = np.where((self.raw_p2im-offset)>0,offset+(1.*self.max_exp_time/self.p2et)*(self.raw_p2im-offset),self.raw_p2im)
        else :    
            corr_p1 = np.where((self.p1_im-offset)>0,offset+(1.*self.max_exp_time/self.p1et)*(self.p1_im-offset),self.p1_im)
            corr_p2 = np.where((self.p2_im-offset)>0,offset+(1.*self.max_exp_time/self.p2et)*(self.p2_im-offset),self.p2_im)
        return corr_p1, corr_p2
