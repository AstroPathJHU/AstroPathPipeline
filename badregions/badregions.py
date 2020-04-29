import abc, cv2, functools, logging, matplotlib.pyplot as plt, methodtools, numpy as np

logger = logging.getLogger("badregions")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(message)s, %(funcName)s, %(asctime)s"))
logger.addHandler(handler)

class BadRegionFinder(abc.ABC):
  def __init__(self, image):
    self.__image = image

  @property
  def image(self): return self.__image

  @abc.abstractmethod
  def badregions(self): pass
  def goodregions(self, *args, **kwargs):
    return ~self.badregions(*args, **kwargs)

  def show(self, *, alpha=1, saveas=None, **kwargs):
    imagepurple = np.transpose([self.image, self.image//2, self.image], (1, 2, 0))
    plt.imshow(imagepurple)

    badhighlight = np.array(
      [0*self.image+1, 0*self.image+1, 0*self.image, self.badregions(**kwargs)*alpha],
      dtype=float,
    ).transpose(1, 2, 0)
    plt.imshow(badhighlight)

    if saveas is None:
      plt.show()
    else:
      plt.savefig(saveas)
      plt.close()

  @methodtools.lru_cache()
  def __laplacian(self):
    return cv2.Laplacian(self.image, cv2.CV_64F)
  @property
  def laplacian(self):
    return self.__laplacian()

class BadRegionFinderLaplaceStd(BadRegionFinder):
  """
  Find blurry regions of the image, like in this article:
  https://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/
  And label those as bad.  Instead of just looking at the absolute
  variance, we take std/mean, so it's invariant under scaling the
  intensity.
  (but not invariant under scaling the dimensions...)
  """

  def __init__(self, image, *, blocksize=40, blockoffset=5):
    super().__init__(image)
    self.__blocksize = blocksize
    self.__blockoffset = blockoffset

  @property
  def blocksize(self): return self.__blocksize
  @property
  def blockoffset(self): return self.__blockoffset

  @property
  def __evaluationypoints(self):
    ysize, xsize = self.image.shape
    return range((ysize % self.blockoffset) // 2, ysize, self.blockoffset)
  @property
  def __evaluationxpoints(self):
    ysize, xsize = self.image.shape
    return range((xsize % self.blockoffset) // 2, xsize, self.blockoffset)

  @property
  def __evaluationgrid(self):
    return np.meshgrid(self.__evaluationypoints, self.__evaluationxpoints, indexing="ij")

  @methodtools.lru_cache()
  def __laplacestd(self):
    grid = self.__evaluationgrid

    evaluatedlaplacestd = self.regionlaplacianstd(*grid)
    return self.makebiggrid(evaluatedlaplacestd, self.image.shape)

  @property
  def laplacestd(self): return self.__laplacestd()

  @methodtools.lru_cache()
  def __mean(self):
    grid = self.__evaluationgrid

    evaluatedmean = self.regionmean(*grid)
    return self.makebiggrid(evaluatedmean, self.image.shape)

  @property
  def mean(self): return self.__mean()

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

  def makebiggrid(self, smallgridvalues, biggridshape):
    biggridvalues = np.ndarray(biggridshape)

    for iy, y in enumerate(self.__evaluationypoints):
      starty = (y + self.__evaluationypoints[iy-1])//2 if iy != 0 else 0
      endy = (y + self.__evaluationypoints[iy+1])//2 if iy != len(self.__evaluationypoints)-1 else biggridshape[0]
      for ix, x in enumerate(self.__evaluationxpoints):
        startx = (x + self.__evaluationxpoints[ix-1])//2 if ix != 0 else 0
        endx = (x + self.__evaluationxpoints[ix+1])//2 if ix != len(self.__evaluationxpoints)-1 else biggridshape[1]

        biggridvalues[starty:endy, startx:endx] = smallgridvalues[iy,ix]

    return biggridvalues

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

  def __regionmean(self, y, x):
    xmin = x-self.blocksize//2
    ymin = y-self.blocksize//2
    xmax = xmin+self.blocksize
    ymax = ymin+self.blocksize

    xmin = max(xmin, 0)
    ymin = max(ymin, 0)

    slc = self.image[ymin:ymax, xmin:xmax]
    return slc.mean()

  def regionmean(self, y, x):
    return np.vectorize(self.__regionmean)(y, x)

class BadRegionFinderWatershedSegmentation(BadRegionFinder):
  """
  Building off of BadRegionFinderLaplaceStd, instead of taking
  arbitrary regions we want to try to segment the image.
  The goal isn't to get exact cells, but just to pinpoint region
  boundaries within a few pixels.  Then we look at the laplacian
  around the border, and if the border is fuzzy we call it bad.

  Working from https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_watershed/py_watershed.html
  """

  def __init__(self, image):
    super().__init__(image)

  @methodtools.lru_cache()
  def segment(self, *, boundaryregionsize=5):
    gray = self.image
    img = np.transpose([self.image, self.image, self.image], (1, 2, 0))

    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_OTSU)

    # noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

    # sure background area
    sure_bg = cv2.dilate(opening,kernel,iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,boundaryregionsize,255,0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1

    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0

    markers = cv2.watershed(img, markers)

    return markers

  @methodtools.lru_cache()
  def cell(self, i, **kwargs):
    return self.segment(**kwargs) == i

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
    segmented = self.segment(**kwargs)
    cellquantification = self.cellquantification(**kwargs)

    result = np.zeros_like(self.image, dtype=bool)
    for k, v in cellquantification.items():
      logger.info("processing cell #%d", k)
      if v < threshold:
        result |= self.cell(k, **kwargs)

    return result
