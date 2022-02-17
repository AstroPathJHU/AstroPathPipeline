import abc, collections, contextlib, cv2, methodtools, more_itertools, numpy as np, sklearn.decomposition
try:
  contextlib.nullcontext
except AttributeError:
  import contextlib2 as contextlib

from ...shared.logging import dummylogger
from ...shared.rectangle import Rectangle, RectangleReadComponentTiffBase, RectangleReadComponentTiffMultiLayer, RectangleReadComponentTiffSingleLayer, RectangleReadIm3Base, RectangleReadIm3MultiLayer, RectangleReadIm3SingleLayer
from ...shared.rectangletransformation import RectangleTransformationBase
from ...utilities import units
from .flatfield import meanimage

class ApplyMeanImage(RectangleTransformationBase):
  """
  Rectangle transformation that divides the rectangle images by the
  mean image over all of the rectangles in the set.
  """
  def __init__(self, mean_image=None, logger=dummylogger):
    self.__meanimage = mean_image
    self.__allrectangles = None
    self.__logger = logger

  @property
  def meanimage(self):
    self.setmeanimage()
    return self.__meanimage

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
        with contextlib.ExitStack() as stack:
          allimages = []
          n = len(self.__allrectangles)
          for i, r in enumerate(self.__allrectangles, start=1):
            self.__logger.debug(f"loading rectangle {i}/{n}")
            rawimage = stack.enter_context(r.using_image_before_flatfield())
            allimages.append(rawimage)
          self.__logger.info("meanimage")
          mean_image = meanimage(allimages)
      self.__meanimage = mean_image

class AlignmentRectangleBase(Rectangle):
  """
  Rectangle that divides the image by the
  mean image over all of the rectangles in the set.
  """
  def __post_init__(self, *args, mean_image=None, use_mean_image=True, logger=dummylogger, **kwargs):
    if use_mean_image:
      self.__meanimagetransformation = ApplyMeanImage(mean_image=mean_image, logger=logger)
    else:
      self.__meanimagetransformation = None
    super().__post_init__(*args, **kwargs)
    self.__rawimage = None
    self.__logger = logger

  @property
  @abc.abstractmethod
  def imageloaderbeforeflatfield(self): pass
  @property
  @abc.abstractmethod
  def alignmentlayers(self): pass

  @methodtools.lru_cache()
  @property
  def alignmentimageloader(self):
    return TransformedImage(self.imageloaderbeforeflatfield, self.__meanimagetransformation)

  def setrectanglelist(self, *args, **kwargs):
    if self.__meanimagetransformation is not None:
      self.__meanimagetransformation.setrectanglelist(*args, **kwargs)
  def setmeanimage(self, *args, **kwargs):
    if self.__meanimagetransformation is not None:
      self.__meanimagetransformation.setmeanimage(*args, **kwargs)

  @property
  def meanimage(self):
    if self.__meanimagetransformation is None: return None
    self.setmeanimage()
    return self.__meanimagetransformation.meanimage

  def using_image_before_flatfield(self):
    return self.imageloaderbeforeflatfield.using_image()
  @property
  def image_before_flatfield(self):
    return self.imageloaderbeforeflatfield.image
  def using_alignment_image(self):
    return self.alignmentimageloader.using_image()
  @property
  def alignmentimage(self):
    return self.alignmentimageloader.image

class AlignmentRectangleIm3Base(AlignmentRectangleBase, RectangleReadIm3Base):
  def __init_subclass__(cls, *args, **kwargs):
    super().__init_subclass__(*args, **kwargs)
    if issubclass(cls, RectangleReadComponentTiffBase):
      raise ValueError("Alignment rectangle has to read im3 or component tiff, not both")
  @methodtools.lru_cache()
  @property
  def imageloaderbeforeflatfield(self):
    return self.im3loader
  @property
  def alignmentlayers(self): return self.layersim3

class AlignmentRectangleIm3MultiLayer(AlignmentRectangleBase, RectangleReadIm3MultiLayer):
  pass

class AlignmentRectangleIm3SingleLayer(AlignmentRectangleIm3Base, RectangleReadIm3SingleLayer):
  pass

class AlignmentRectangleComponentTiffBase(AlignmentRectangleBase, RectangleReadComponentTiffBase):
  def __init_subclass__(cls, *args, **kwargs):
    super().__init_subclass__(*args, **kwargs)
    if issubclass(cls, RectangleReadIm3Base):
      raise ValueError("Alignment rectangle has to read im3 or component tiff, not both")
  @methodtools.lru_cache()
  @property
  def imageloaderbeforeflatfield(self):
    return self.componenttiffloader

class AlignmentRectangleComponentTiffMultiLayer(AlignmentRectangleComponentTiffBase, RectangleReadComponentTiffMultiLayer):
  pass

class AlignmentRectangleComponentTiffSingleLayer(AlignmentRectangleComponentTiffBase, RectangleReadComponentTiffSingleLayer):
  pass
