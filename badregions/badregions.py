"""
https://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/
"""

import cv2, functools, matplotlib.pyplot as plt, methodtools, numpy as np

class BadRegionFinder:
  def __init__(self, image, *, blocksize=40, blockoffset=5):
    self.__image = image
    self.__blocksize = blocksize
    self.__blockoffset = blockoffset

  @property
  def image(self): return self.__image
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

    evaluatedlaplacestd = regionlaplacianstd(self.image, *grid, width=self.blocksize)
    return makebiggrid(self.__evaluationypoints, self.__evaluationxpoints, evaluatedlaplacestd, self.image.shape)

  @property
  def laplacestd(self): return self.__laplacestd()

  @methodtools.lru_cache()
  def __mean(self):
    grid = self.__evaluationgrid

    evaluatedmean = regionmean(self.image, *grid, width=self.blocksize)
    return makebiggrid(self.__evaluationypoints, self.__evaluationxpoints, evaluatedmean, self.image.shape)

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

def makebiggrid(smallgridy, smallgridx, smallgridvalues, biggridshape):
  biggridvalues = np.ndarray(biggridshape)

  for iy, y in enumerate(smallgridy):
    starty = (y + smallgridy[iy-1])//2 if iy != 0 else 0
    endy = (y + smallgridy[iy+1])//2 if iy != len(smallgridy)-1 else biggridshape[0]
    for ix, x in enumerate(smallgridx):
      startx = (x + smallgridx[ix-1])//2 if ix != 0 else 0
      endx = (x + smallgridx[ix+1])//2 if ix != len(smallgridx)-1 else biggridshape[1]

      biggridvalues[starty:endy, startx:endx] = smallgridvalues[iy,ix]

  return biggridvalues

@functools.partial(np.vectorize, excluded=(0, 3, "image", "width"))
def regionlaplacianstd(image, y, x, width=40):
  xmin = x-width//2
  ymin = y-width//2
  xmax = xmin+width
  ymax = ymin+width

  xmin = max(xmin, 0)
  ymin = max(ymin, 0)

  slc = image[ymin:ymax, xmin:xmax]
  laplace = cv2.Laplacian(slc, cv2.CV_64F)

  return laplace.std()

@functools.partial(np.vectorize, excluded=(0, 3, "image", "width"))
def regionmean(image, y, x, width=40):
  xmin = x-width//2
  ymin = y-width//2
  xmax = xmin+width
  ymax = ymin+width

  xmin = max(xmin, 0)
  ymin = max(ymin, 0)

  slc = image[ymin:ymax, xmin:xmax]
  return slc.mean()
