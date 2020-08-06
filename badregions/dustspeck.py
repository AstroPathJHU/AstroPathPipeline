import cv2, matplotlib.pyplot as plt, numpy as np, scipy.ndimage
from ..flatfield.utilities import getImageArrayLayerHistograms, getLayerOtsuThresholdsAndWeights
from .badregions import BadRegionFinder

class DustSpeckFinder(BadRegionFinder):
  def __init__(self, image, **kwargs):
    self.fullimage = image
    super().__init__(image=image[0], *args, **kwargs)

  eigenvector = np.array([
    3.74232724e-01,  3.97916625e-01,  3.96707054e-01,  3.83565923e-01,
    3.49349573e-01,  3.02937958e-01,  2.63828851e-01,  2.25467130e-01,
    1.88632897e-01, -3.22435638e-02, -4.60956243e-02, -4.91402069e-02,
   -3.84106161e-02, -3.22414045e-02, -2.29283253e-02, -1.81406213e-02,
   -1.99784469e-02, -2.58578890e-02, -6.00813643e-02, -6.16669897e-02,
   -4.87819064e-02, -3.30696926e-02, -3.75845706e-02, -4.45549348e-02,
   -4.79685955e-02, -3.46724381e-03, -8.92635498e-03, -1.40048653e-04,
    5.43894270e-03,  5.74610904e-04, -9.31201613e-03, -1.82879178e-02,
   -2.67289473e-03, -1.10337250e-02,
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
