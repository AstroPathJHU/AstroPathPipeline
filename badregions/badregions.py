import abc, cv2, functools, matplotlib.pyplot as plt, methodtools, numpy as np

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

class BadRegionFinderLaplaceStd(BadRegionFinder):
  """
  https://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/
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

    slc = self.image[ymin:ymax, xmin:xmax]
    laplace = cv2.Laplacian(slc, cv2.CV_64F)

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
