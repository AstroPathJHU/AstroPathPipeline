import abc, cv2, logging, matplotlib.pyplot as plt, methodtools, numpy as np, skimage

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

  def show(self, *, alpha=1, saveas=None, plotstyling=lambda fig, ax: None, scale=0.2, **kwargs):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    imagepurple = np.transpose([self.image, self.image//2, self.image], (1, 2, 0))
    imagepurple = (imagepurple * scale).astype(np.uint16)
    plt.imshow(imagepurple)

    badhighlight = np.array(
      [0*self.image+1, 0*self.image+1, 0*self.image, self.badregions(**kwargs)*alpha],
      dtype=float,
    ).transpose(1, 2, 0)
    plt.imshow(badhighlight)

    plotstyling(fig, ax)

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

class BadRegionFinderEvaluateGrid(BadRegionFinder):

  def __init__(self, image, *, blocksize=40, blockoffset=5):
    super().__init__(image)
    self.__blocksize = blocksize
    self.__blockoffset = blockoffset

  @property
  def blocksize(self): return self.__blocksize
  @property
  def blockoffset(self): return self.__blockoffset

  @property
  def evaluationypoints(self):
    ysize, xsize = self.image.shape
    return range((ysize % self.blockoffset) // 2, ysize, self.blockoffset)
  @property
  def evaluationxpoints(self):
    ysize, xsize = self.image.shape
    return range((xsize % self.blockoffset) // 2, xsize, self.blockoffset)

  @property
  def evaluationgrid(self):
    return np.meshgrid(self.evaluationypoints, self.evaluationxpoints, indexing="ij")

  def makebiggrid(self, smallgridvalues, biggridshape):
    biggridvalues = np.ndarray(biggridshape)

    for iy, y in enumerate(self.evaluationypoints):
      starty = (y + self.evaluationypoints[iy-1])//2 if iy != 0 else 0
      endy = (y + self.evaluationypoints[iy+1])//2 if iy != len(self.evaluationypoints)-1 else biggridshape[0]
      for ix, x in enumerate(self.evaluationxpoints):
        startx = (x + self.evaluationxpoints[ix-1])//2 if ix != 0 else 0
        endx = (x + self.evaluationxpoints[ix+1])//2 if ix != len(self.evaluationxpoints)-1 else biggridshape[1]

        biggridvalues[starty:endy, startx:endx] = smallgridvalues[iy,ix]

    return biggridvalues

class BadRegionFinderRegionMean(BadRegionFinderEvaluateGrid):
  @methodtools.lru_cache()
  def __mean(self):
    grid = self.evaluationgrid

    evaluatedmean = self.regionmean(*grid)
    return self.makebiggrid(evaluatedmean, self.image.shape)

  @property
  def mean(self): return self.__mean()

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

class BadRegionFinderRegionStd(BadRegionFinderEvaluateGrid):
  @methodtools.lru_cache()
  def __std(self):
    grid = self.evaluationgrid

    evaluatedstd = self.regionstd(*grid)
    return self.makebiggrid(evaluatedstd, self.image.shape)

  @property
  def std(self): return self.__std()

  def __regionstd(self, y, x):
    xmin = x-self.blocksize//2
    ymin = y-self.blocksize//2
    xmax = xmin+self.blocksize
    ymax = ymin+self.blocksize

    xmin = max(xmin, 0)
    ymin = max(ymin, 0)

    slc = self.image[ymin:ymax, xmin:xmax]
    return slc.std()

  def regionstd(self, y, x):
    return np.vectorize(self.__regionstd)(y, x)

class BadRegionFinderSegmentation(BadRegionFinder):
  """
  Base class for bad region finders that work cell by cell
  """
  @abc.abstractmethod
  def segment(self, **kwargs): pass

  @methodtools.lru_cache()
  def cell(self, i, **kwargs):
    return self.segment(**kwargs) == i

  @abc.abstractmethod
  def cellquantification(self, i, **kwargs): pass

class BadRegionFinderWatershedSegmentation(BadRegionFinderSegmentation):
  """
  Working from https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_watershed/py_watershed.html
  """

  @methodtools.lru_cache()
  def segment(self, *, boundaryregionsize=5):
    gray = skimage.img_as_ubyte(self.image)
    img = np.transpose([gray, gray, gray], (1, 2, 0))

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
