import numpy as np

from ..baseclasses.rectangle import RectangleWithImage
from ..utilities.misc import dummylogger
from .flatfield import meanimage

class AlignmentRectangle(RectangleWithImage):
  def __init__(self, *args, mean_image=None, keepraw=False, logger=dummylogger, **kwargs):
    super().__init__(*args, **kwargs)
    self.__allrectangles = None
    self.__meanimage = mean_image
    self.__rawimage = None
    self.__keeprawimage = False
    self.__logger = logger

  def setrectanglelist(self, allrectangles):
    self.__allrectangles = allrectangles

  def getimage(self):
    self.__setmeanimage()
    for r in self.__allrectangles:
      r.__setmeanimage(mean_image=self.__meanimage)
    img = np.empty_like(self.__rawimage)
    img[:] = np.rint(self.__rawimage / self.__meanimage.flatfield)
    if not self.__keeprawimage:
      self.__rawimage = None
    self.__meanimage = None
    return img

  def __getrawimage(self):
    if self.__rawimage is None:
      self.__rawimage = super().getimage()
    return self.__rawimage

  def __setmeanimage(self, mean_image=None):
    if self.__allrectangles is None:
      raise ValueError("Have to call setrectanglelist() before getting any images")
    if self.__meanimage is None:
      if mean_image is None:
        if mean_image is None:
          allimages = []
          n = len(self.__allrectangles)
          for i, r in enumerate(self.__allrectangles, start=1):
            self.__logger.info(f"loading rectangle {i}/{n}")
            allimages.append(r.__getrawimage())
          self.__logger.info("meanimage")
          mean_image = meanimage(allimages)
      self.__meanimage = mean_image

  @property
  def rawimage(self):
    self.__keeprawimage = True
    return self.__getrawimage()
