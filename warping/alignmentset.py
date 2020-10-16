#imports
from .config import CONST
from ..alignment.alignmentset import AlignmentSetFromXML
from ..alignment.rectangle import AlignmentRectangle
from ..baseclasses.rectangle import RectangleTransformationBase
from ..utilities.img_correction import correctImageLayerForExposureTime, correctImageLayerWithFlatfield
from ..utilities.img_file_io import getExposureTimesByLayer
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
    def __init__(self, *args, rtd, mtd, samp, number_of_layers, med_et, offset, flatfield, transformations=None, **kwargs):
        super().__init__(*args, transformations=None, use_mean_image=False, **kwargs)
        exp_time = None
        if transformations is None: transformations = []
        if (med_et is not None) and (offset is not None) :
            rfp = os.path.join(rtd,samp,self.file.replace(CONST.IM3_EXT,CONST.RAW_EXT))
            try :
                exp_time = (getExposureTimesByLayer(rfp,number_of_layers,mtd))[self.layer-1]
            except Exception :
                try :
                    exp_time = (getExposureTimesByLayer(rfp,number_of_layers,rtd))[self.layer-1]
                except Exception as e :
                    raise ValueError(f'Could not find rectangle exposure time for raw file {rfp} in metadata top dir {mtd} or rawfile top dir {rtd}! \n Exception: {e}')
            transformations.append(CorrectForExposureTime(exp_time,med_et,offset))
        if flatfield is not None:
            transformations.append(ApplyFlatfield(flatfield))
        super().__init__(*args, transformations=transformations, use_mean_image=False, **kwargs)
        

class AlignmentSetForWarping(AlignmentSetFromXML):
    def __init__(self, *args, med_et, offset, flatfield, **kwargs):
        self.__med_et = med_et
        self.__offset = offset
        self.__flatfield = flatfield
        super().__init__(*args, **kwargs)

    rectangletype = RectangleForWarping

    @property
    def rectangleextrakwargs(self):
        return {
            **super().rectangleextrakwargs,
            "rtd": self.root2,
            "mtd": self.root,
            "samp": self.samp.SlideID,
            "number_of_layers": self.nlayers,
            "med_et": self.__med_et,
            "offset": self.__offset,
            "flatfield": self.__flatfield,
        }
