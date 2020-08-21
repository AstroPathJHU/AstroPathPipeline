import numpy as np
from ..alignment.alignmentset import AlignmentSetFromXML
from ..alignment.rectangle import AlignmentRectangle
from ..utilities.img_file_io import smoothImageWorker

class RectangleForExposureTime(AlignmentRectangle):
  def __init__(self, *args, flatfield, smoothsigma, **kwargs):
    self.__flatfield = flatfield
    self.__smoothsigma = smoothsigma
    super().__init__(*args, **kwargs)
  def transformimage(self, originalimage):
    image = np.rint(originalimage / self.__flatfield).astype(np.uint16)
    if self.__smoothsigma is not None:
      image = smoothImageWorker(image,self.__smoothsigma)
    return image

class AlignmentSetForExposureTime(AlignmentSetFromXML):
  def __init__(self, *args, flatfield, smoothsigma, **kwargs):
    self.__flatfield = flatfield
    self.__smoothsigma = smoothsigma
    super().__init__(*args, **kwargs)

  rectangletype = RectangleForExposureTime

  @property
  def rectangleextrakwargs(self):
    return {
      **super().rectangleextrakwargs,
      "flatfield": self.__flatfield,
      "smoothsigma": self.__smoothsigma
    }
