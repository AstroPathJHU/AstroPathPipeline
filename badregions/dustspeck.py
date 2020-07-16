import cv2, matplotlib.pyplot as plt, numpy as np, scipy.ndimage
from ..flatfield.utilities import getImageArrayLayerHistograms, getLayerOtsuThresholdsAndWeights
from .badregions import BadRegionFinder

class DustSpeckFinder(BadRegionFinder):
  def badregions(self, *, dilatesize=None, statserodesize=None, showdebugplots=False):
    hist = getImageArrayLayerHistograms(self.image)
    thresholds, weights = getLayerOtsuThresholdsAndWeights(hist)
    threshold = thresholds[1]  #first one finds signal, second finds dust speck

    badregions = cv2.UMat((self.image > threshold).astype(np.uint8))
    if showdebugplots:
      print("image > threshold")
      plt.imshow(badregions.get())
      plt.show()

    ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    badregions = cv2.morphologyEx(badregions, cv2.MORPH_CLOSE, ellipse, borderType=cv2.BORDER_REPLICATE)
    if showdebugplots:
      print("after small close")
      plt.imshow(badregions.get())
      plt.show()
    badregions = cv2.morphologyEx(badregions, cv2.MORPH_OPEN,  ellipse, borderType=cv2.BORDER_REPLICATE)
    if showdebugplots:
      print("after small open")
      plt.imshow(badregions.get())
      plt.show()

    ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (100, 100))
    badregions = cv2.morphologyEx(badregions, cv2.MORPH_OPEN,  ellipse, borderType=cv2.BORDER_REPLICATE)
    if showdebugplots:
      print("after big open")
      plt.imshow(badregions.get())
      plt.show()
    badregions = cv2.morphologyEx(badregions, cv2.MORPH_CLOSE, ellipse, borderType=cv2.BORDER_REPLICATE)
    if showdebugplots:
      print("after big close")
      plt.imshow(badregions.get())
      plt.show()

    if dilatesize is not None:
      ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilatesize, dilatesize))
      badregions = cv2.dilate(badregions, ellipse)

    badregions = badregions.get().astype(bool)

    labeled, _ = scipy.ndimage.label(badregions)
    for i in np.unique(labeled):
      if i == 0: continue

      selection = (labeled==i).astype(np.uint16)
      if statserodesize is not None:
        ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (statserodesize, statserodesize))
        selection = cv2.erode(selection, ellipse)

      intensities = self.image[selection.astype(bool)]
      min, q01, q99, max = np.quantile(intensities, [0, 0.01, 0.99, 1])

      if q01 / q99 < 0.1:
        badregions[labeled == i] = False

    return badregions
