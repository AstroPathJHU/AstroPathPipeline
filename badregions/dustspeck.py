import cv2, matplotlib.pyplot as plt, numpy as np, scipy.ndimage
from ..flatfield.utilities import getImageArrayLayerHistograms, getLayerOtsuThresholdsAndWeights
from .badregions import BadRegionFinder

class DustSpeckFinder(BadRegionFinder):
  def __init__(self, image, *args, **kwargs):
    self.fullimage = image
    super().__init__(image=image[0], *args, **kwargs)

  eigenvector = np.array([
    3.73880790e-01,  3.97652102e-01,  3.96508416e-01,  3.83417588e-01,
    3.49132453e-01,  3.02387811e-01,  2.63084761e-01,  2.24859324e-01,
    1.88104483e-01, -3.41723829e-02, -4.89441657e-02, -5.22999938e-02,
   -4.10684986e-02, -3.45748414e-02, -2.42433192e-02, -1.69204249e-02,
   -1.68248782e-02, -2.28126481e-02, -6.54554407e-02, -6.70447075e-02,
   -5.31272143e-02, -3.40825405e-02, -3.50213688e-02, -4.10613572e-02,
   -4.57104458e-02, -3.77583399e-03, -1.07586166e-02, -1.18092188e-04,
    1.02102564e-02,  7.18445463e-03, -3.97109615e-03, -1.50314166e-02,
    4.01981362e-03, -6.38929944e-03, -1.26212998e-02
  ])

  def badregions(self, *, threshold=2e-6, dilatesize=None, statserodesize=None, showdebugplots=False):
    im = self.fullimage
    transposed = im.transpose(1, 2, 0)
    scaled = transposed / np.sum(im, axis=(1, 2))
    projected = scaled @ self.eigenvector

    signalmask = projected > threshold
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

    """
    labeled, _ = scipy.ndimage.label(badregions)
    aftersmallcloseopen_labeled = None

    for i in np.unique(labeled):
      if i == 0: continue

      thisregion = labeled == i
      if showdebugplots:
        plt.imshow(thisregion)

      fractionalsize = np.sum(thisregion) / thisregion.size
      if showdebugplots:
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
        if showdebugplots:
          print("fractional intersection", fractionalintersection)
        if fractionalintersection > 0.75:
          break
      else:
        badregions[thisregion] = False
        continue
    """

    return badregions
