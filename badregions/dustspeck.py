import cv2, numpy as np
from .badregions import BadRegionFinder

class DustSpeckFinder(BadRegionFinder):
  def badregions(self, *, sigma=101, threshold=500, openclosesizes=[100], dilatesize=0):
    blur = cv2.GaussianBlur(self.image, (sigma, sigma), sigma)
    badregions = cv2.UMat((blur > threshold).astype(np.uint8))

    for size in openclosesizes:
      ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
      badregions = cv2.morphologyEx(badregions, cv2.MORPH_CLOSE, ellipse, borderType=cv2.BORDER_REPLICATE)
      badregions = cv2.morphologyEx(badregions, cv2.MORPH_OPEN,  ellipse, borderType=cv2.BORDER_REPLICATE)

    if dilatesize:
      ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilatesize, dilatesize))
      badregions = cv2.dilate(badregions, ellipse)

    return badregions.get().astype(bool)
