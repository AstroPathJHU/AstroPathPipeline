import numpy as np
from ..alignment.alignmentset import AlignmentSetFromXML
from ..alignment.rectangle import AlignmentRectangle
from ..baseclasses.rectangle import RectangleTransformationBase
from ..utilities.img_correction import correctImageLayerForExposureTime, correctImageLayerWithFlatfield

class CorrectForExposureTime(RectangleTransformationBase):
    def __init__(self, med_et, offset):
        self.__med_et = med_et
        self.__offset = offset
    def transform(self, originalimage):
        return correctImageLayerForExposureTime(originalimage,exp_time,self.__med_et,self.__offset)

class ApplyFlatfield(RectangleTransformationBase):
    def __init__(self, flatfield):
        self.__flatfield = flatfield
    def transform(self, originalimage):
        return correctImageLayerWithFlatfield(originalimage,flatfield)

class RectangleForWarping(AlignmentRectangle):
    def __init__(self, *args, med_et, offset, flatfield, transformations=None, **kwargs):
        if transformations is None: transformations = []
        if (med_et is not None) and (offset is not None) :
            transformations.append(CorrectForExposureTime(med_et,offset))
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
            "median_et": self.__med_et,
            "offset": self.__offset,
            "flatfield": self.__flatfield,
        }
