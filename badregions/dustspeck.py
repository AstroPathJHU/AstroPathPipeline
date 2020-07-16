import cv2, matplotlib.pyplot as plt, numpy as np, scipy.ndimage
from ..flatfield.utilities import getImageArrayLayerHistograms, getLayerOtsuThresholdsAndWeights
from .badregions import BadRegionFinder

class DustSpeckFinder(BadRegionFinder):
  def badregions(self, *, dilatesize=None, statserodesize=None, showdebugplots=False):
    hist = getImageArrayLayerHistograms(self.image)
    thresholds, weights = getLayerOtsuThresholdsAndWeights(hist)
    try:
      threshold = thresholds[1]  #first one finds signal, second finds dust speck
      weight = weights[1]
    except IndexError:
      weight = -1
    if weight <= 0:
      return np.zeros_like(self.image, dtype=bool)

    signalmask = self.image > threshold
    badregions = cv2.UMat(signalmask.astype(np.uint8))
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
    aftersmallcloseopen = badregions.get().astype(bool)

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
    aftersmallcloseopen_labeled = None

    for i in np.unique(labeled):
      if i == 0: continue

      thisregion = labeled == i
      if showdebugplots:
        plt.imshow(thisregion)

      fractionalsize = np.sum(thisregion) / thisregion.size
      print("fractional size", fractionalsize)
      if fractionalsize > 0.99:
        badregions[thisregion] = False
        continue

      selection = thisregion.astype(np.uint16)
      if statserodesize is not None:
        ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (statserodesize, statserodesize))
        selection = cv2.erode(selection, ellipse)
      intensities = self.image[selection.astype(bool)]
      min, q01, q99, max = np.quantile(intensities, [0, 0.01, 0.99, 1])
      if showdebugplots:
        print("quantiles", min, q01, q99, max, q01/q99)

      if q01 / q99 < 0.1:
        badregions[thisregion] = False
        continue

      ratio = np.sum(signalmask[thisregion]) / np.sum(thisregion)
      #the way this could be < 1 is if there are little holes closed up by
      #the small scale close
      #a few of those are ok because we don't want to be sensitive to noise
      #but if we have a lot of them that probably means this isn't a real dust speck
      if showdebugplots:
        print("ratio", ratio)
      if ratio < 0.95:
        badregions[thisregion] = False
        continue

      #make sure the region isn't just a remnant of a huge original region
      if aftersmallcloseopen_labeled is None:
        aftersmallcloseopen_labeled, _ = scipy.ndimage.label(aftersmallcloseopen)
      for j in np.unique(aftersmallcloseopen_labeled):
        if j == 0: continue
        thisoldregion = aftersmallcloseopen_labeled == j
        fractionalintersection = np.sum(thisoldregion & thisregion) / np.sum(thisoldregion | thisregion)
        print("fractional intersection", fractionalintersection)
        if fractionalintersection > 0.5:
          break
      else:
        badregions[thisregion] = False
        continue

    return badregions
