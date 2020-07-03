import cv2, methodtools, numpy as np
from .badregions import BadRegionFinderRegionMean, BadRegionFinderWatershedSegmentation, logger

class TissueFoldFinderSimple(BadRegionFinderRegionMean):
  """
  Find blurry regions of the image, like in this article:
  https://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/
  And label those as bad.  Instead of just looking at the absolute
  variance, we take std/mean, so it's invariant under scaling the
  intensity.
  (but not invariant under scaling the dimensions...)
  """
  @methodtools.lru_cache()
  def __laplacestd(self):
    grid = self.evaluationgrid

    evaluatedlaplacestd = self.regionlaplacianstd(*grid)
    return self.makebiggrid(evaluatedlaplacestd, self.image.shape)

  @property
  def laplacestd(self): return self.__laplacestd()

  @methodtools.lru_cache()
  def __ratio(self):
    meannonzero = self.mean != 0
    maxratio = max(self.laplacestd[meannonzero] / self.mean[meannonzero])

    laplacestd = np.where(meannonzero, self.laplacestd, maxratio)
    mean = np.where(meannonzero, self.mean, 1)

    return laplacestd / mean

  @property
  def ratio(self): return self.__ratio()

  def badregions(self, *, threshold=0.15):
    return self.ratio<threshold

  def __regionlaplacianstd(self, y, x):
    xmin = x-self.blocksize//2
    ymin = y-self.blocksize//2
    xmax = xmin+self.blocksize
    ymax = ymin+self.blocksize

    xmin = max(xmin, 0)
    ymin = max(ymin, 0)

    laplace = self.laplacian[ymin:ymax, xmin:xmax]

    return laplace.std()

  def regionlaplacianstd(self, y, x):
    return np.vectorize(self.__regionlaplacianstd)(y, x)

class TissueFoldFinderByCell(BadRegionFinderWatershedSegmentation):
  """
  Building off of TissueFoldFinderSimple, instead of taking
  arbitrary regions we want to try to segment the image.
  The goal isn't to get exact cells, but just to pinpoint region
  boundaries within a few pixels.  Then we look at the laplacian
  around the border, and if the border is fuzzy we call it bad.
  """

  @methodtools.lru_cache()
  def cellquantification(self, **kwargs):
    segmented = self.segment(**kwargs)

    #segmented contains 1 for background, -1 for boundaries, and 2, 3, 4... for the different regions
    allboundaries = self.cell(-1, **kwargs)
    image = self.image
    laplacian = self.laplacian

    result = {}

    for i in np.unique(segmented):
      if i < 2: continue
      logger.info("processing cell #%d", i)

      #find the boundary of this region
      thisregion = self.cell(i, **kwargs)
      regionboundary = (cv2.distanceTransform((~thisregion).astype(np.uint8), cv2.DIST_L2, 5) <= 2) & allboundaries

      kernel = np.ones((5, 5), np.uint8)
      dilated = cv2.dilate(regionboundary.astype(np.uint8), kernel, iterations=1)

      result[i] = laplacian[dilated>0].std() / image[thisregion>0].mean()

    return result

  def badregions(self, *, threshold, **kwargs):
    cellquantification = self.cellquantification(**kwargs)

    result = np.zeros_like(self.image, dtype=bool)
    for k, v in cellquantification.items():
      logger.info("processing cell #%d", k)
      if v < threshold:
        result |= self.cell(k, **kwargs)

    return result
