#imports
from ..alignment.alignmentset import AlignmentSetFromXML
from ..alignment.rectangle import AlignmentRectangle
from ..baseclasses.rectangle import RectangleTransformationBase
from ..utilities.img_correction import correctImageLayerForExposureTime, correctImageLayerWithFlatfield
from ..utilities.img_file_io import getExposureTimesByLayer
from ..utilities.config import CONST as UNIV_CONST
import os

class CorrectForExposureTime(RectangleTransformationBase):
    def __init__(self, et, med_et, offset):
        self.__exp_time = et
        self.__med_et = med_et
        self.__offset = offset
    def transform(self, originalimage):
        return correctImageLayerForExposureTime(originalimage,self.__exp_time,self.__med_et,self.__offset)

class ApplyFlatfield(RectangleTransformationBase):
    def __init__(self, flatfield):
        self.__flatfield = flatfield
    def transform(self, originalimage):
        return correctImageLayerWithFlatfield(originalimage,self.__flatfield)

class RectangleForWarping(AlignmentRectangle):
    def __user_init__(self, *args, rtd, root_dir, slide_ID, number_of_layers, med_et, offset, flatfield, transformations=None, **kwargs):
        super().__user_init__(*args, transformations=None, **kwargs)
        exp_time = None
        if transformations is None: transformations = []
        if (med_et is not None) and (offset is not None) :
            rfp = os.path.join(rtd,slide_ID,self.file.replace(UNIV_CONST.IM3_EXT,UNIV_CONST.RAW_EXT))
            try :
                exp_time = (getExposureTimesByLayer(rfp,number_of_layers,root_dir))[self.layer-1]
            except Exception :
                try :
                    exp_time = (getExposureTimesByLayer(rfp,number_of_layers,rtd))[self.layer-1]
                except Exception as e :
                    raise ValueError(f'Could not find rectangle exposure time for raw file {rfp} in root dir {root_dir} or rawfile top dir {rtd}! \n Exception: {e}')
            transformations.append(CorrectForExposureTime(exp_time,med_et,offset))
        if flatfield is not None:
            transformations.append(ApplyFlatfield(flatfield))
        super().__user_init__(*args, transformations=transformations, **kwargs)
        

class AlignmentSetForWarping(AlignmentSetFromXML):
    def __init__(self, *args, med_et, offset, flatfield, **kwargs):
        self.__med_et = med_et
        self.__offset = offset
        self.__flatfield = flatfield
        super().__init__(*args, use_mean_image=False, **kwargs)

    rectangletype = RectangleForWarping

    @property
    def rectangleextrakwargs(self):
        return {
            **super().rectangleextrakwargs,
            "rtd": self.root2,
            "root_dir": self.root,
            "slide_ID": self.samp.SlideID,
            "number_of_layers": self.nlayers,
            "med_et": self.__med_et,
            "offset": self.__offset,
            "flatfield": self.__flatfield,
        }
