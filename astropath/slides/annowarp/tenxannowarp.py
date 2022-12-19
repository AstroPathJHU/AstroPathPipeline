import matplotlib.patches, matplotlib.pyplot as plt
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

    fig, ax = plt.subplots()
    im, (x1, y1, x2, y2) = self.circle_subplot(spot)
    plt.imshow(im, extent=(x1, x2, y2, y1))
    circle = matplotlib.patches.Circle((xc, yc), dia/2, alpha=0.3, color='r')
    ax.add_patch(circle)
