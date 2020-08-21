import numpy as np

from ..baseclasses.rectangle import RectangleTransformImageBase, RectangleWithImage
from ..utilities.misc import dummylogger
from .flatfield import meanimage

class AlignmentRectangle(RectangleTransformImageBase):
  def __init__(self, *args, mean_image=None, keepraw=False, logger=dummylogger, **kwargs):
    super().__init__(originalrectangle=RectangleWithImage(*args, **kwargs))
    self.__allrectangles = None
    self.__meanimage = mean_image
    self.__rawimage = None
    self.__keeprawimage = False
    self.__logger = logger

  def setrectanglelist(self, allrectangles):
    self.__allrectangles = allrectangles

  def transformimage(self, originalimage):
    self.__setmeanimage()
    for r in self.__allrectangles:
      r.__setmeanimage(mean_image=self.__meanimage)
    img = np.empty_like(originalimage)
    img[:] = np.rint(originalimage / self.__meanimage.flatfield)
    self.__meanimage = None
    return img

  def __setmeanimage(self, mean_image=None):
    if self.__allrectangles is None:
      raise ValueError("Have to call setrectanglelist() before getting any images")
    if self.__meanimage is None:
      if mean_image is None:
        allimages = []
        n = len(self.__allrectangles)
        for i, r in enumerate(self.__allrectangles, start=1):
          self.__logger.info(f"loading rectangle {i}/{n}")
          with r.using_original_image() as rawimage:
            allimages.append(rawimage)
        self.__logger.info("meanimage")
        mean_image = meanimage(allimages)
      self.__meanimage = mean_image

  @property
  def layer(self):
    return self.originalrectangle.layer
