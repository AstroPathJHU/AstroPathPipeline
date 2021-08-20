#imports
from ...slides.align.overlap import AlignmentOverlap

class AlignmentOverlapForWarping(AlignmentOverlap) :
    """
    A streamlined version of an AlignmentOverlap to use in fitting for warping patterns
    """

    def __post_init__(self,*args,**kwargs) :
        super().__post_init__(*args,**kwargs)
        self.__images = None
        self.__cutimages = None
        self.__overlap_shape = None
        self.__overlap_npix = None

    def myupdaterectangles(self, rectangles):
        super().updaterectangles(rectangles)
        self.__calculate_image_variables()

    @property
    def images(self) :
        if self.__images is None :
            self.__calculate_image_variables()
        return self.__images
    @property
    def cutimages(self) :
        if self.__cutimages is None :
            self.__calculate_image_variables()
        return self.__cutimages
    @property
    def overlap_shape(self) :
        if self.__overlap_shape is None :
            self.__calculate_image_variables()
        return self.__overlap_shape
    @property
    def overlap_npix(self) :
        if self.__overlap_npix is None :
            self.__calculate_image_variables()
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
