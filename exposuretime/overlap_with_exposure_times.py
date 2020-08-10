#imports
import numpy as np, matplotlib.pyplot as plt
from .utilities import ExposureTimeOverlapFitResult

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
        self._raw_p2_im = rp1i
    @property
    def raw_npix(self) :
        if self.raw_p1im is not None and self.raw_p2im is not None :
            return self.raw_p1im.shape[0]*self.raw_p1im.shape[1]
        else :
            return self.npix

    #################### PUBLIC FUNCTIONS ####################

    def __init__(self,olap,p1et,p2et,max_exp_time,cutimages) :
        self.n    = olap.n
        self.p1   = olap.p1
        self.p2   = olap.p2
        self.tag  = olap.tag
        self.p1et = p1et
        self.p2et = p2et
        self.et_diff = self.p2et-self.p1et
        self.max_exp_time = max_exp_time
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
            self.p1_im = whole_p1_im[SLICES[self.tag]]
            self.p2_im = whole_p2_im[SLICES[self.tag]]
        else :
            self.p1_im = whole_p1_im
            self.p2_im = whole_p2_im
        self.npix = self.p1_im.shape[0]*self.p1_im.shape[1]
        self._raw_p1_im=None
        self._raw_p2_im=None

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
        return ExposureTimeOverlapFitResult(self.n,self.p1,self.p2,self.tag,self.p1et,self.p2et,self.et_diff,self.npix,self.orig_cost,self.corr_cost)

    def saveComparisonImages(self,best_fit_offset,filename_stem) :
        orig_overlay = np.clip(np.array([self.p1_im, self.p2_im, 0.5*(self.p1_im+self.p2_im)]).transpose(1, 2, 0) / 1000., 0., 1.)
        corr_p1im, corr_p2im = self.__getCorrectedImages(best_fit_offset,False)
        corr_overlay = np.clip(np.array([corr_p1im, corr_p2im, 0.5*(corr_p1im+corr_p2im)]).transpose(1, 2, 0) / 1000., 0., 1.)
        f,ax = plt.subplots(1,2,figsize=(2*6.4,4.6))
        ax[0].imshow(orig_overlay)
        ax[0].set_title(f'overlap {self.n} original (cost={self.orig_cost:.3f})')
        ax[1].imshow(corr_overlay)
        ax[1].set_title(f'overlap {self.n} corrected (cost={self.corr_cost:.3f})')
        plt.savefig(f'{filename_stem}_offset={best_fit_offset:.3f}_clipped_and_smoothed.png')
        plt.close()
        if self.raw_p1im is not None and self.raw_p2im is not None :
            orig_overlay = np.clip(np.array([self.raw_p1im, self.raw_p2im, 0.5*(self.raw_p1im+self.raw_p2im)]).transpose(1, 2, 0) / 1000.,0.,1.)
            corr_p1im, corr_p2im = self.__getCorrectedImages(best_fit_offset,True)
            corr_overlay = np.clip(np.array([corr_p1im, corr_p2im, 0.5*(corr_p1im+corr_p2im)]).transpose(1, 2, 0) / 1000.,0.,1.)
            f,ax = plt.subplots(1,2,figsize=(2*6.4,4.6))
            ax[0].imshow(orig_overlay)
            ax[0].set_title(f'raw overlap {self.n} original')
            ax[1].imshow(corr_overlay)
            ax[1].set_title(f'raw overlap {self.n} corrected')
            plt.savefig(f'{filename_stem}_offset={best_fit_offset:.3f}_raw_and_whole.png')
            plt.close()

    #################### PRIVATE HELPER FUNCTIONS ####################

    def __getCorrectedImages(self,offset,raw) :
        if raw and (self.raw_p1im is not None) and (self.raw_p2im is not None) :
            corr_p1 = np.where((self.raw_p1im-offset)>0,offset+(1.*self.max_exp_time/self.p1et)*(self.raw_p1im-offset),self.raw_p1im)
            corr_p2 = np.where((self.raw_p2im-offset)>0,offset+(1.*self.max_exp_time/self.p2et)*(self.raw_p2im-offset),self.raw_p2im)
        else :    
            corr_p1 = np.where((self.p1_im-offset)>0,offset+(1.*self.max_exp_time/self.p1et)*(self.p1_im-offset),self.p1_im)
            corr_p2 = np.where((self.p2_im-offset)>0,offset+(1.*self.max_exp_time/self.p2et)*(self.p2_im-offset),self.p2_im)
        return corr_p1, corr_p2
