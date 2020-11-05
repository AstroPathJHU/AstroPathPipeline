import collections, contextlib, cv2, more_itertools, numpy as np, sklearn.decomposition

from ..baseclasses.rectangle import RectangleFromOtherRectangle, RectangleProvideImage, RectangleTransformationBase, RectangleWithImage, RectangleWithImageBase, RectangleWithImageMultiLayer
from ..utilities import units
from ..utilities.misc import dummylogger
from .flatfield import meanimage

class ApplyMeanImage(RectangleTransformationBase):
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
            self.__logger.info(f"loading rectangle {i}/{n}")
            rawimage = stack.enter_context(r.using_image_before_flatfield())
            allimages.append(rawimage)
          self.__logger.info("meanimage")
          mean_image = meanimage(allimages)
      self.__meanimage = mean_image

class AlignmentRectangleBase(RectangleWithImageBase):
  def __init__(self, *args, mean_image=None, use_mean_image=True, logger=dummylogger, transformations=None, **kwargs):
    if transformations is None: transformations = []
    if use_mean_image:
      self.__meanimagetransformation = ApplyMeanImage(mean_image=mean_image, logger=logger)
      self.__meanimagetransformationindex = len(transformations)
      transformations.append(self.__meanimagetransformation)
    else:
      self.__meanimagetransformation = None
    super().__init__(*args, transformations=transformations, **kwargs)
    self.__rawimage = None
    self.__logger = logger

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
    if self.__meanimagetransformation is None: return contextlib.nullcontext()
    return self.using_image(self.__meanimagetransformationindex)
  @property
  def image_before_flatfield(self):
    return self.any_image(self.__meanimagetransformationindex)

class AlignmentRectangle(AlignmentRectangleBase, RectangleWithImage):
  pass

class AlignmentRectangleMultiLayer(AlignmentRectangleBase, RectangleWithImageMultiLayer):
  pass

class AlignmentRectangleProvideImage(AlignmentRectangleBase, RectangleProvideImage):
  def __init__(self, *args, layer, **kwargs):
    self.__layer = layer
    super().__init__(*args, **kwargs)
  @property
  def layer(self):
    return self.__layer

class ConsolidateBroadbandFilters(RectangleTransformationBase):
  def __init__(self, layershifts, broadbandfilters=None):
    self.__layershifts = layershifts
    self.__broadbandfilters = broadbandfilters

  def setbroadbandfilters(self, broadbandfilters):
    self.__broadbandfilters = broadbandfilters

  def transform(self, originalimage):
    if self.__broadbandfilters is None:
      raise ValueError("Have to call setbroadbandfilters first")
    shifted = collections.defaultdict(list)
    for layer, shift, filter in more_itertools.zip_equal(originalimage, self.__layershifts, self.__broadbandfilters):
      dx, dy = units.nominal_values(shift)
      shifted[filter].append(
        cv2.warpAffine(
          layer,
          np.array([[1, 0, dx], [0, 1, dy]]),
          flags=cv2.INTER_CUBIC,
          borderMode=cv2.BORDER_REPLICATE,
          dsize=layer.T.shape,
        )
      )

    shifted = {k: np.array(v) for k, v in shifted.items()}

    pca = sklearn.decomposition.PCA(n_components=1, copy=False)
    pcas = {
      filter:
      pca.fit_transform(
        layers
        .reshape(
          layers.shape[0], layers.shape[1]*layers.shape[2]
        ).T
      ).reshape(
        layers.shape[1], layers.shape[2]
      )
      for filter, layers in shifted.items()
    }

    return np.array(list(pcas.values()))

class RectanglePCAByBroadbandFilter(RectangleFromOtherRectangle):
  def __init__(self, *args, layershifts, transformations=None, **kwargs):
    if transformations is None: transformations = []
    self.__pcabroadbandtransformation = ConsolidateBroadbandFilters(layershifts=layershifts)
    transformations.append(self.__pcabroadbandtransformation)
    super().__init__(*args, transformations=transformations, **kwargs)
    self.__pcabroadbandtransformation.setbroadbandfilters(broadbandfilters=self.originalrectangle.broadbandfilters)

  def setrectanglelist(self, rectanglelist): pass
  def using_image_before_flatfield(self): return contextlib.nullcontext()
  @property
  def layers(self):
    return self.originalrectangle.layers
