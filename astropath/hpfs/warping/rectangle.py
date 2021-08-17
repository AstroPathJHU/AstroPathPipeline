#imports
from ...shared.rectangle import RectangleCorrectedIm3SingleLayer
from ...slides.align.rectangle import AlignmentRectangleBase

class AlignmentRectangleForWarping(RectangleCorrectedIm3SingleLayer,AlignmentRectangleBase) :
    """
    Rectangles that are used to fit warping
    """
    def __post_init__(self,*args,**kwargs) :
        super().__post_init__(*args,use_mean_image=False,**kwargs)
        self.__single_image = None
    @property
    def image(self) :
        if self.__single_image is None :
            self.__single_image = super().image
        return self.__single_image
    @image.setter
    def image(self,im) :
        self.__single_image = im
