#imports
import numpy as np, matplotlib.pyplot as plt
from .utilities import ExposureTimeOverlapFitResult
from .config import CONST
from ..utilities.img_file_io import correctImageLayerForExposureTime
import methodtools

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
        self.offsets = np.linspace(offset_bounds[0],offset_bounds[1],CONST.OVERLAP_COST_PARAMETERIZATION_N_POINTS)
        self.costs   = [costFromImages(p1_im,p2_im,self.p1et,self.p2et,self.max_exp_time,o) for o in self.offsets]
        self.uncorrected_cost = costFromImages(p1_im,p2_im,self.p1et,self.p2et,self.max_exp_time,-1.)
        self.npix = p1_im.shape[0]*p1_im.shape[1]
        self._raw_p1_im=None
        self._raw_p2_im=None

    def getCostAndNPix(self,offset) :
        """
        Calculate the cost at a given offset by interpolating between points
        """
        index_below, index_above = self.__getLeftSideIndex(offset)
        x1, y1, slope = self.__getLowerPointAndSlope(index_below)
        interp_cost = slope*(offset-x1)+y1
        return interp_cost, self.npix

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
        f,ax = plt.subplots(1,3,figsize=(3*6.4,4.6))
        ax[0].imshow(orig_overlay)
        ax[0].set_title(f'overlap {self.n} original (cost={self.orig_cost:.3f})')
        ax[1].imshow(corr_overlay)
        ax[1].set_title(f'overlap {self.n} corrected (cost={self.corr_cost:.3f})')
        offsets = list(range(self.offset_bounds[0],self.offset_bounds[1]))
        raw_image_costs = []; smoothed_clipped_image_costs = []
        for o in self.offsets :
            raw_image_costs.append(costFromImages(self.raw_p1im,self.raw_p2im,self.p1et,self.p2et,self.max_exp_time,o)/self.raw_npix)
            pcost,npix = self.getCostAndNPix(o)
            smoothed_clipped_image_costs.append(pcost/npix)
        ax[2].plot(offsets,raw_image_costs,linewidth=2,label='cost per pixel from raw images')
        ax[2].plot(offsets,smoothed_clipped_image_costs,linewidth=2,label='cost from smoothed/clipped images')
        ax[2].plot([best_fit_offset,best_fit_offset],[0.8*y for y in ax[2].get_ylim()],linewidth=2,color='k',label=f'best fit offset ({best_fit_offset:.3f})')
        ax[2].set_title(f'overlap {self.n} (tag={self.tag}) w/ exp. time diff. = {self.et_diff:.3f}')
        ax[2].set_xlabel('offset')
        ax[2].set_ylabel('avg. cost per pixel')
        ax[2].legend(loc='best')
        plt.savefig(f'{filename_stem}_offset={best_fit_offset:.3f}.png')
        plt.close()

    #################### PRIVATE HELPER FUNCTIONS ####################

    #helper function to return the (offset,cost) at a particular index and the slope of the line connecting it to the point to its right
    @methodtools.lru_cache()
    def __getLowerPointAndSlope(self,index_below) :
        x1 = self.offsets[index_below]; x2 = self.offsets[index_below+1]
        y1 = self.costs[index_below];   y2 = self.costs[index_below+1]
        slope = (y2-y1)/(x2-x1)
        return x1, y1, slope

    #helper function to return the last index in the offset list that's to the left of a given offset
    @methodtools.lru_cache()
    def __getLeftSideIndex(self,offset) :
        below = 0 
        while self.offsets[below]<offset and below<len(self.offsets)-1:
            below+=1
        return below

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
