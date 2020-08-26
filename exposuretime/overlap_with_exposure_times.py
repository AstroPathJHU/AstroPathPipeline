#imports
import numpy as np, matplotlib.pyplot as plt
from .utilities import ExposureTimeOverlapFitResult
from .config import CONST
from ..utilities.img_file_io import correctImageLayerForExposureTime

#################### FILE-SCOPE HELPER FUNCTIONS ####################

#helper function to calculate the L1 fitting cost from the raw images given a dark current offset
def costFromImages(p1im,p2im,p1et,p2et,maxet,offset) :
    if offset>=0. :
        corrp1 = correctImageLayerForExposureTime(p1im,p1et,maxet,offset)
        corrp2 = correctImageLayerForExposureTime(p2im,p2et,maxet,offset)
    else : #if the given offset is negative don't correct the images
        corrp1 = p1im
        corrp2 = p2im
    return np.sum(np.abs(corrp1-corrp2))

#helper class for comparing overlap image exposure times
class OverlapWithExposureTimes :

    #################### PROPERTIES ####################

    @property
    def raw_p1im(self) :
        return self._raw_p1_im
    @raw_p1im.setter
    def raw_p1im(self,rp1i) :
        self._raw_p1_im = rp1i
    @property
    def raw_p2im(self) :
        return self._raw_p2_im
    @raw_p2im.setter
    def raw_p2im(self,rp2i) :
        self._raw_p2_im = rp2i
    @property
    def raw_npix(self) :
        if self.raw_p1im is not None and self.raw_p2im is not None :
            return self.raw_p1im.shape[0]*self.raw_p1im.shape[1]
        else :
            return self.npix

    #################### PUBLIC FUNCTIONS ####################

    def __init__(self,olap,p1et,p2et,max_exp_time,cutimages,offset_bounds) :
        self.n    = olap.n
        self.p1   = olap.p1
        self.p2   = olap.p2
        self.tag  = olap.tag
        self.p1et = p1et
        self.p2et = p2et
        self.et_diff = self.p2et-self.p1et
        self.max_exp_time = max_exp_time
        p1_im, p2_im = self.__getp1p2Images(olap,cutimages)
        self.fit_pars = self.__getFitParameters(p1_im,p2_im,offset_bounds)
        self.uncorrected_cost = costFromImages(p1_im,p2_im,self.p1et,self.p2et,self.max_exp_time,-1.)
        self.npix = p1_im.shape[0]*p1_im.shape[1]
        self._raw_p1_im=None
        self._raw_p2_im=None

    def getCostAndNPix(self,offset) :
        """
        Calculate the cost at a given offset from the polynomial fit parameterization
        """
        cost = 0. 
        for i in range(CONST.OVERLAP_COST_POLYFIT_DEG+1) :
            cost+=self.fit_pars[i]*(offset**(CONST.OVERLAP_COST_POLYFIT_DEG-i))
        return cost, self.npix

    def getFitResult(self,best_fit_offset) :
        """
        Get this overlap's fit result object
        """
        self.orig_cost = self.uncorrected_cost/self.npix
        cc,cnp = self.getCostAndNPix(best_fit_offset)
        self.corr_cost = cc/cnp
        return ExposureTimeOverlapFitResult(self.n,self.p1,self.p2,self.tag,self.p1et,self.p2et,self.et_diff,self.npix,self.orig_cost,self.corr_cost)

    def saveComparisonImages(self,best_fit_offset,filename_stem) :
        """
        If this overlap has raw images set, save a plot of the pre- and post-correction overlays
        """
        if (self.raw_p1im is None) or (self.raw_p2im is None) :
            return
        orig_overlay = np.clip(np.array([self.raw_p1im, self.raw_p2im, 0.5*(self.raw_p1im+self.raw_p2im)]).transpose(1, 2, 0) / 1000.,0.,1.)
        corr_p1im = correctImageLayerForExposureTime(self.raw_p1im,self.p1et,self.max_exp_time,best_fit_offset)
        corr_p2im = correctImageLayerForExposureTime(self.raw_p2im,self.p2et,self.max_exp_time,best_fit_offset)
        corr_overlay = np.clip(np.array([corr_p1im, corr_p2im, 0.5*(corr_p1im+corr_p2im)]).transpose(1, 2, 0) / 1000.,0.,1.)
        f,ax = plt.subplots(1,2,figsize=(2*6.4,4.6))
        ax[0].imshow(orig_overlay)
        ax[0].set_title(f'overlap {self.n} original (cost={self.orig_cost:.3f})')
        ax[1].imshow(corr_overlay)
        ax[1].set_title(f'overlap {self.n} corrected (cost={self.corr_cost:.3f})')
        plt.savefig(f'{filename_stem}_offset={best_fit_offset:.3f}.png')
        plt.close()

    #################### PRIVATE HELPER FUNCTIONS ####################

    #helper function to get the two overlap images
    def __getp1p2Images(self,olap,cutimages) :
        whole_p1_im, whole_p2_im = olap.shifted
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
            p1_im = whole_p1_im[SLICES[self.tag]]
            p2_im = whole_p2_im[SLICES[self.tag]]
        else :
            p1_im = whole_p1_im
            p2_im = whole_p2_im
        return p1_im,p2_im

    #helper function to get the cost parameterization
    def __getFitParameters(self,p1_im,p2_im,offset_bounds) :
        offsets = []; costs = []
        for o in range(offset_bounds[0],offset_bounds[1]+1) :
            offsets.append(o)
            costs.append(costFromImages(p1_im,p2_im,self.p1et,self.p2et,self.max_exp_time,o))
        return np.polyfit(offsets,costs,CONST.OVERLAP_COST_POLYFIT_DEG)
