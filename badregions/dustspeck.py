import cv2, numpy as np, scipy.ndimage
from .badregions import BadRegionFinder

class DustSpeckFinder(BadRegionFinder):
  def badregions(self, *, sigma=101, threshold=500, openclosesizes=[100], dilatesize=0, statserodesize=0):
    blur = cv2.GaussianBlur(self.image, (sigma, sigma), sigma)
    badregions = cv2.UMat((blur > threshold).astype(np.uint8))

    for size in openclosesizes:
      ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
      badregions = cv2.morphologyEx(badregions, cv2.MORPH_CLOSE, ellipse, borderType=cv2.BORDER_REPLICATE)
      badregions = cv2.morphologyEx(badregions, cv2.MORPH_OPEN,  ellipse, borderType=cv2.BORDER_REPLICATE)

    if dilatesize:
      ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilatesize, dilatesize))
      badregions = cv2.dilate(badregions, ellipse)

    badregions = badregions.get().astype(bool)

    labeled, _ = scipy.ndimage.label(badregions)
    for i in np.unique(labeled):
      if i == 0: continue

      selection = (labeled==i).astype(np.uint16)
      if statserodesize:
        ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (statserodesize, statserodesize))
        selection = cv2.erode(selection, ellipse)

      intensities = self.image[selection.astype(bool)]
      min, q01, q99, max = np.quantile(intensities, [0, 0.01, 0.99, 1])

      if q01 / q99 < 0.1:
        badregions[labeled == i] = False

    return badregions
