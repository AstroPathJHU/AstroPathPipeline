#imports
from ...shared.rectangletransformation import RectangleTransformationBase
from ...slides.align.alignsample import AlignSampleFromXML
from ...slides.align.rectangle import AlignmentRectangle
from ...utilities.img_file_io import smoothImageWorker
from ...utilities.img_correction import correctImageLayerWithFlatfield

class ApplyFlatfield(RectangleTransformationBase):
    def __init__(self, flatfield):
        self.__flatfield = flatfield
    def transform(self, originalimage):
        return correctImageLayerWithFlatfield(originalimage,self.__flatfield)

class SmoothImage(RectangleTransformationBase):
    def __init__(self, smoothsigma):
        self.__smoothsigma = smoothsigma
    def transform(self, originalimage):
        return smoothImageWorker(originalimage, self.__smoothsigma)

class RectangleForExposureTime(AlignmentRectangle):
    def __post_init__(self, *args, flatfield, smoothsigma, transformations=None, **kwargs):
        if transformations is None: transformations = []
        if flatfield is not None:
            transformations.append(ApplyFlatfield(flatfield))
        if smoothsigma is not None:
            transformations.append(SmoothImage(smoothsigma))
        super().__post_init__(*args, transformations=transformations, **kwargs)

class AlignSampleForExposureTime(AlignSampleFromXML):
    def __init__(self, *args, flatfield, smoothsigma, **kwargs):
        self.__flatfield = flatfield
        self.__smoothsigma = smoothsigma
        super().__init__(*args, use_mean_image=False, **kwargs)

    rectangletype = RectangleForExposureTime

    @property
    def rectangleextrakwargs(self):
        return {
            **super().rectangleextrakwargs,
            "flatfield": self.__flatfield,
            "smoothsigma": self.__smoothsigma
        }
