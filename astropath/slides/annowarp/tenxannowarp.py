from ...shared.tenx import TenXSampleBase

class TenXAnnoWarp(TenXSampleBase):
  @property
  def logmodule(self):
    return "tenxannowarp"

  def drawcircle(self, spot):
    xc, yc = spot.imageX, spot.imageY
    x1 = xc - spot.diameter
    x2 = xc + spot.diameter
    y1 = yc - spot.diameter
    y2 = yc + spot.diameter
    with self.using_wsi() as wsi:
      plt.imshow(wsi[x1:x2, y1:y2])
