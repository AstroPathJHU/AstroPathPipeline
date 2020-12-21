import matplotlib.patches, matplotlib.pyplot as plt, numpy as np
from ..utilities import units

def showannotation(image, regions, *, qpscale, imagescale, xlim=(), ylim=(), figurekwargs={}, showplot=None, saveas=None):
  fig = plt.figure(**figurekwargs)
  xlim = (np.array(xlim) / units.onepixel(imagescale)).astype(float)
  ylim = (np.array(ylim) / units.onepixel(imagescale)).astype(float)
  ax = fig.add_subplot(1, 1, 1)
  plt.imshow(image)
  for region in regions:
    vertices = units.convertpscale([[v.x, v.y] for v in region.poly.vertices], qpscale, imagescale) / units.onepixel(imagescale)
    polygon = matplotlib.patches.Polygon(vertices, alpha=0.5, color="r")
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
