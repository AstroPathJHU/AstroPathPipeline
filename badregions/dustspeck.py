import cv2, matplotlib.pyplot as plt, numpy as np, scipy.ndimage
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

    badregions = badregions.get().astype(bool)
    regions, nregions = scipy.ndimage.label(badregions)
    for r in range(1, nregions+1):
      thisregion = regions==r
      if np.sum(thisregion) < 5000:
        badregions[thisregion] = 0
      else: print(r, np.sum(thisregion))

    if showdebugplots:
      print("after choosing big regions")
      plt.imshow(badregions)
      plt.show()

    badregions = cv2.UMat(badregions.astype(np.uint8))
    ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
    badregions = cv2.morphologyEx(badregions, cv2.MORPH_OPEN,  ellipse, borderType=cv2.BORDER_REPLICATE)
    if showdebugplots:
      print("after medium open")
      plt.imshow(badregions.get())
      plt.show()

    badregions = badregions.get().astype(bool)
    regions, nregions = scipy.ndimage.label(badregions)
    for r in range(1, nregions+1):
      thisregion = regions==r
      if np.sum(thisregion) < 5000:
        badregions[thisregion] = 0
      else: print(r, np.sum(thisregion))

    if showdebugplots:
      print("after choosing big regions")
      plt.imshow(badregions)
      plt.show()

    badregions = cv2.UMat(badregions.astype(np.uint8))

    ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
    badregions = cv2.morphologyEx(badregions, cv2.MORPH_CLOSE,  ellipse, borderType=cv2.BORDER_REPLICATE)
    if showdebugplots:
      print("after medium close")
      plt.imshow(badregions.get())
      plt.show()

    if dilatesize is not None:
      ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilatesize, dilatesize))
      badregions = cv2.dilate(badregions, ellipse)

    badregions = badregions.get().astype(bool)

    return badregions
