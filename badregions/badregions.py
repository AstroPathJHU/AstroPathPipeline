"""
https://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/
"""

import cv2, functools, imageio, matplotlib.pyplot as plt, methodtools, numpy as np, scipy.interpolate

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
  def __laplacevariance(self):
    grid = self.__evaluationgrid

    evaluatedlaplacevariance = regionlaplacianvariance(self.image, *grid, width=self.blocksize)
    return makebiggrid(self.__evaluationypoints, self.__evaluationxpoints, evaluatedlaplacevariance, self.image.shape)

  @property
  def laplacevariance(self): return self.__laplacevariance()

  @methodtools.lru_cache()
  def __meansq(self):
    grid = self.__evaluationgrid

    evaluatedmeansq = regionmeansq(self.image, *grid, width=self.blocksize)
    return makebiggrid(self.__evaluationypoints, self.__evaluationxpoints, evaluatedmeansq, self.image.shape)

  @property
  def meansq(self): return self.__meansq()

  @methodtools.lru_cache()
  def __ratio(self):
    meansqnonzero = self.meansq != 0
    maxratio = max(self.laplacevariance[meansqnonzero] / self.meansq[meansqnonzero])

    laplacevariance = np.where(meansqnonzero, self.laplacevariance, maxratio)
    meansq = np.where(meansqnonzero, self.meansq, 1)

    return laplacevariance / meansq

  @property
  def ratio(self): return self.__ratio()

  def show(self, *, threshold=.02, alpha=0.4):
    imagepurple = np.transpose([self.image, self.image//2, self.image], (1, 2, 0))
    plt.imshow(imagepurple)

    badhighlight = np.transpose(np.array([0*self.image+1, 0*self.image+1, 0*self.image, np.where(self.ratio<threshold, alpha, 0.)]), (1, 2, 0))
    plt.imshow(badhighlight)

    plt.show()

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
def regionlaplacianvariance(image, y, x, width=40):
  xmin = x-width//2
  ymin = y-width//2
  xmax = xmin+width
  ymax = ymin+width

  xmin = max(xmin, 0)
  ymin = max(ymin, 0)

  slc = image[ymin:ymax, xmin:xmax]
  laplace = cv2.Laplacian(slc, cv2.CV_64F)

  return laplace.var()

@functools.partial(np.vectorize, excluded=(0, 3, "image", "width"))
def regionmeansq(image, y, x, width=40):
  xmin = x-width//2
  ymin = y-width//2
  xmax = xmin+width
  ymax = ymin+width

  xmin = max(xmin, 0)
  ymin = max(ymin, 0)

  slc = image[ymin:ymax, xmin:xmax]
  return slc.mean()**2
