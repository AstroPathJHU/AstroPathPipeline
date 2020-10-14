#imports
from .config import CONST
from ..alignment.alignmentset import AlignmentSetFromXML
from ..alignment.rectangle import AlignmentRectangle
from ..baseclasses.rectangle import RectangleTransformationBase
from ..utilities.img_correction import correctImageLayerForExposureTime, correctImageLayerWithFlatfield
from ..utilities.img_file_io import getExposureTimesByLayer
import numpy as np
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
        return correctImageLayerWithFlatfield(originalimage,flatfield)

class RectangleForWarping(AlignmentRectangle):
    @property
    def exp_time(self):
        return self.__exp_time
    def __init__(self, *args, rtd, mtd, samp, nlayers, layer, med_et, offset, flatfield, transformations=None, **kwargs):
        if transformations is None: transformations = []
        self.__exp_time = None
        if (rtd is not None) and (mtd is not None) and (samp is not None) and (nlayers is not None) and (layer is not None) 
            and (med_et is not None) and (offset is not None) :
            transformations.append(CorrectForExposureTime(self.exp_time,med_et,offset))
        if flatfield is not None:
            transformations.append(ApplyFlatfield(flatfield))
        super().__init__(*args, transformations=transformations, use_mean_image=False, **kwargs)
        rfp = os.path.join(rtd,samp,self.file.replace(CONST.IM3_EXT,CONST.RAW_EXT))
        try :
            self.__exp_time = getExposureTimesByLayer(rfp,nlayers,mtd)
        except Exception :
            try :
                self.__exp_time = getExposureTimesByLayer(rfp,nlayers,rtd)
            except Exception :
                raise ValueError(f'Could not find rectangle exposure time for raw file {rfp} in metadata top dir {mtd} or rawfile top dir {rtd}!')

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
            "samp": self.samp,
            "nlayers": self.nlayers,
            "layer":self.layer,
            "med_et": self.__med_et,
            "offset": self.__offset,
            "flatfield": self.__flatfield,
        }
