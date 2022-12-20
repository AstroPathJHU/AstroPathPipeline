import cv2, matplotlib.patches, matplotlib.pyplot as plt, numpy as np, skimage.filters
from ...shared.tenx import TenXSampleBase

class TenXAnnoWarp(TenXSampleBase):
  @property
  def logmodule(self):
    return "tenxannowarp"

  def circle_subplot(self, spot):
    xc = float(spot.imageX / self.onepixel)
    yc = float(spot.imageY / self.onepixel)
    dia = float(spot.dia / self.onepixel)
    x1 = int(xc - dia)
    x2 = int(xc + dia)
    y1 = int(yc - dia)
    y2 = int(yc + dia)
    with self.using_wsi() as wsi:
      return wsi[y1:y2, x1:x2], (x1, y1, x2, y2)

  def drawcircle(self, spot):
    xc = float(spot.imageX / self.onepixel)
    yc = float(spot.imageY / self.onepixel)
    dia = float(spot.dia / self.onepixel)

    im, (x1, y1, x2, y2) = self.circle_subplot(spot)
    circle = matplotlib.patches.Circle((xc, yc), dia/2, alpha=0.3, color='r')
    ax.add_patch(circle)

  def findcircle(self, spot, draw=False):
    wsi, (x1, y1, x2, y2) = self.circle_subplot(spot)
    gray = cv2.cvtColor(wsi, cv2.COLOR_BGR2GRAY)
    bw = np.where(gray>120, 255, 0).astype(np.uint8)
    smooth = (skimage.filters.gaussian(bw, sigma=10, mode='nearest') * 255).astype(np.uint8)
    xc = float(spot.imageX / self.onepixel)
    yc = float(spot.imageY / self.onepixel)
    dia = float(spot.dia / spot.onepixel)
    circles = cv2.HoughCircles(
      gray,cv2.HOUGH_GRADIENT,1,20,
      param1=50,param2=50,minRadius=int(dia*.9//2),maxRadius=int(dia*1.1//2)
    )
    if circles is not None:
      circle = circles[0][0]
      circle[0] += x1
      circle[1] += y1
    else:
      circle = None

    if draw:
      fig, ax = plt.subplots()
      plt.imshow(
        wsi,
        #smooth,
        extent=(x1, x2, y2, y1),
      )
      patchkwargs = {
        #"alpha": 0.3,
        "fill": False,
        "linewidth": 2,
      }
      if circle is not None:
        ax.add_patch(matplotlib.patches.Circle((circle[0], circle[1]), circle[2], color='b', **patchkwargs))
      ax.add_patch(matplotlib.patches.Circle((xc, yc), dia/2, color='r', **patchkwargs))
      plt.show()

    return circle
