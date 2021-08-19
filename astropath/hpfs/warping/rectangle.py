#imports
from ...shared.rectangle import RectangleCorrectedIm3SingleLayer
from ...slides.align.rectangle import AlignmentRectangleBase

class AlignmentRectangleForWarping(RectangleCorrectedIm3SingleLayer,AlignmentRectangleBase) :
    """
    Rectangles that are used to fit warping; streamlined to hold images in memory
    """

    def __post_init__(self,*args,**kwargs) :
        super().__post_init__(*args,use_mean_image=False,**kwargs)
        self.__single_image = None
    
    @property
    def image(self) :
        if self.__single_image is None :
            self.__single_image = super().image
            if len(self.__single_image.shape)==3 :
                if self.__single_image.shape[2]==1 :
                    self.__single_image = self.__single_image[:,:,0]
                else :
                    errmsg = f'ERROR: {self.__class__.__name__} found an image of shape {self.__single_image.shape}'
                    raise ValueError(errmsg)
            elif len(self.__single_image.shape)!=2 :
                errmsg = f'ERROR: {self.__class__.__name__} found an image of shape {self.__single_image.shape}'
                raise ValueError(errmsg)
        return self.__single_image
    
    @image.setter
    def image(self,im) :
        self.__single_image = im
