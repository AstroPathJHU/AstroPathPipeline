import matplotlib.pyplot as plt, numpy as np
from ...shared.polygon import SimplePolygon
from ...utilities import units

def showannotation(image, regions, *, imagescale, vertices=None, xlim=(), ylim=(), figurekwargs={}, showplot=None, saveas=None, alpha=0.5):
  """
  show an image with the annotations

  image: the wsi or qptiff image
  regions: the region objects from regions.csv
  vertices: a list of vertices (optional if the region has a polygon written)
  imagescale: scale of the image, e.g. apscale for the qptiff or pscale for the wsi
  xlim, ylim: optional limits in imagescale units (default: full range of the image)
  alpha: transparency of the annotation region (default: 0.5)
  figurekwargs: kwargs for plt.figure (default: {})
  showplot: should the function call plt.show()? (default: True if saveas is None otherwise False)
  saveas: filename to save the figure (default: None)
  """
  fig = plt.figure(**figurekwargs)
  xlim = (np.array(xlim) / units.onepixel(imagescale)).astype(float)
  ylim = (np.array(ylim) / units.onepixel(imagescale)).astype(float)
  ax = fig.add_subplot(1, 1, 1)
  plt.imshow(image)
  for region in regions:
    if vertices is None:
      poly = region.poly
    else:
      poly = SimplePolygon(vertices=[v for v in vertices if v.regionid == region.regionid], pscale=imagescale)

    polygon = poly.matplotlibpolygon(
      imagescale=imagescale,
      alpha=alpha,
      color="r",
    )
    ax.add_patch(polygon)
  plt.xlim(*xlim)
  plt.ylim(*ylim)
  if showplot is None: showplot = saveas is None
  if showplot:
    plt.show()
  if saveas is not None:
    fig.savefig(saveas)
  if not showplot:
    plt.close()
