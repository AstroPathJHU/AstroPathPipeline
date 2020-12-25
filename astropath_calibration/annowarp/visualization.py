import matplotlib.patches, matplotlib.pyplot as plt, numpy as np
from ..utilities import units

def showannotation(image, regions, *, qpscale, imagescale, xlim=(), ylim=(), vertices=None, figurekwargs={}, showplot=None, saveas=None):
  fig = plt.figure(**figurekwargs)
  xlim = (np.array(xlim) / units.onepixel(imagescale)).astype(float)
  ylim = (np.array(ylim) / units.onepixel(imagescale)).astype(float)
  ax = fig.add_subplot(1, 1, 1)
  plt.imshow(image)
  for region in regions:
    if vertices is None:
      polyvertices = region.poly.vertices
    else:
      polyvertices = [v for v in vertices if v.regionid == region.regionid]

    polygon = matplotlib.patches.Polygon(
      units.convertpscale(
        [[v.x, v.y] for v in polyvertices],
        qpscale,
        imagescale
      ) / units.onepixel(imagescale),
      alpha=0.5,
      color="r"
    )
    ax.add_patch(polygon)
  plt.xlim(*xlim)
  plt.ylim(*ylim)
  if showplot is None: showplot = saveas is None
  if showplot:
    plt.show()
  if saveas is not None:
    plt.savefig(saveas)
  if not showplot:
    plt.close()
