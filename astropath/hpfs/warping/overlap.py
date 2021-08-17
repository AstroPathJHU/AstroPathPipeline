#imports
import time
import numpy as np, uncertainties as unc
from ...utilities import units
from ...slides.align.overlap import AlignmentOverlap
from ...slides.align.computeshift import computeshift, mse, shiftimg

class AlignmentOverlapForWarping(AlignmentOverlap) :
    """
    A streamlined version of an AlignmentOverlap to use in fitting for warping patterns
    """

    def myupdaterectangles(self, rectangles):
        super().updaterectangles(rectangles)
        self.__calculate_image_variables()

    @property
    def images(self) :
        return self.__images
    @property
    def cutimages(self) :
        return self.__cutimages
    @property
    def overlap_shape(self) :
        return self.__overlap_shape
    @property
    def overlap_npix(self) :
        return self.__overlap_npix

    def __calculate_image_variables(self) :
        """
        Recalculate a bunch of properties of the overlap images/dimensions/etc. when the rectangle images are set
        or when they change. The point is to try and do these calculations as infrequently as possible.
        Reading the images is what takes the longest
        """
        self.__images = self.rectangles[0].image,self.rectangles[1].image
        if (not hasattr(self,f'_{self.__class__.__name__}__imageshape')) or self.__images[0].shape!=self.__imageshape :
                self.__imageshape = self.__images[0].shape
                self.__cutimage_slices = tuple(self.cutimageslices)
                self.__overlap_shape = (self.__cutimage_slices[0][0].stop-self.__cutimage_slices[0][0].start,
                                        self.__cutimage_slices[0][1].stop-self.__cutimage_slices[0][1].start)
                self.__overlap_npix = self.__overlap_shape[0]*self.__overlap_shape[1]
        self.__cutimages = self.__images[0][self.__cutimage_slices[0]], self.__images[1][self.__cutimage_slices[1]]
