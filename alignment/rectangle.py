import numpy as np

from ..baseclasses.rectangle import RectangleTransformationBase, RectangleWithImage
from ..utilities.misc import dummylogger
from .flatfield import meanimage

class ApplyMeanImage(RectangleTransformationBase):
  def __init__(self, mean_image=None, logger=dummylogger):
    self.__meanimage = mean_image
    self.__allrectangles = None
    self.__logger = logger

  def setrectanglelist(self, allrectangles):
    self.__allrectangles = allrectangles

  def transform(self, originalimage):
    self.setmeanimage()
    for r in self.__allrectangles:
      r.setmeanimage(mean_image=self.__meanimage)
    img = np.empty_like(originalimage)
    img[:] = np.rint(originalimage / self.__meanimage.flatfield)
    return img

  def setmeanimage(self, mean_image=None):
    if self.__allrectangles is None:
      raise ValueError("Have to call setrectanglelist() before getting any images")
    if self.__meanimage is None:
      if mean_image is None:
        allimages = []
        n = len(self.__allrectangles)
        for i, r in enumerate(self.__allrectangles, start=1):
          self.__logger.info(f"loading rectangle {i}/{n}")
          with r.using_image(-2) as rawimage:
            allimages.append(rawimage)
        self.__logger.info("meanimage")
        mean_image = meanimage(allimages)
      self.__meanimage = mean_image

  @property
  def meanimage(self):
    return self.__meanimage

class AlignmentRectangle(RectangleWithImage):
  def __init__(self, *args, mean_image=None, use_mean_image=True, logger=dummylogger, transformations=None, **kwargs):
    if transformations is None: transformations = []
    if use_mean_image:
      self.__meanimagetransformation = ApplyMeanImage(mean_image=mean_image, logger=logger)
      transformations.append(self.__meanimagetransformation)
    super().__init__(*args, transformations=transformations, **kwargs)
    self.__rawimage = None
    self.__logger = logger

  def setrectanglelist(self, *args, **kwargs):
    self.__meanimagetransformation.setrectanglelist(*args, **kwargs)
  def setmeanimage(self, *args, **kwargs):
    self.__meanimagetransformation.setmeanimage(*args, **kwargs)
