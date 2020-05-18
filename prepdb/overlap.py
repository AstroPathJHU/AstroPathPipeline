import abc, dataclasses, matplotlib.pyplot as plt, networkx as nx, numpy as np, typing, uncertainties as unc

from .rectangle import rectangleoroverlapfilter as overlapfilter
from ..utilities import units
from ..utilities.misc import covariance_matrix, dataclass_dc_init, floattoint
from ..utilities.units.dataclasses import DataClassWithDistances, distancefield

@dataclasses.dataclass
class Overlap(DataClassWithDistances):
  pixelsormicrons = "microns"

  n: int
  p1: int
  p2: int
  x1: units.Distance = distancefield(pixelsormicrons=pixelsormicrons)
  y1: units.Distance = distancefield(pixelsormicrons=pixelsormicrons)
  x2: units.Distance = distancefield(pixelsormicrons=pixelsormicrons)
  y2: units.Distance = distancefield(pixelsormicrons=pixelsormicrons)
  tag: int
  layer: dataclasses.InitVar[float]
  pscale: dataclasses.InitVar[float]
  nclip: dataclasses.InitVar[float]
  rectangles: dataclasses.InitVar[float]
  readingfromfile: dataclasses.InitVar[bool] = False

  def __post_init__(self, layer, pscale, nclip, rectangles, readingfromfile=False):
    super().__post_init__(pscale=pscale, readingfromfile=readingfromfile)

    self.layer = layer
    self.nclip = nclip
    self.result = None

    p1rect = [r for r in rectangles if r.n==self.p1]
    p2rect = [r for r in rectangles if r.n==self.p2]
    if not len(p1rect) == len(p2rect) == 1:
      raise ValueError(f"Expected exactly one rectangle each with n={self.p1} and {self.p2}, found {len(p1rect)} and {len(p2rect)}")
    self.rectangles = p1rect[0], p2rect[0]

  @property
  def x1vec(self):
    return np.array([self.x1, self.y1])
  @property
  def x2vec(self):
    return np.array([self.x2, self.y2])

class OverlapCollection(abc.ABC):
  @abc.abstractproperty
  def overlaps(self): pass

  def overlapgraph(self, useexitstatus=False):
    g = nx.DiGraph()
    for o in self.overlaps:
      if useexitstatus and o.result.exit: continue
      g.add_edge(o.p1, o.p2, overlap=o)

    return g

  def nislands(self, *args, **kwargs):
    return nx.number_strongly_connected_components(self.overlapgraph(*args, **kwargs))

  @property
  def overlapsdict(self):
    return {(o.p1, o.p2): o for o in self.overlaps}

  @property
  def overlaprectangleindices(self):
    return frozenset(o.p1 for o in self.overlaps) | frozenset(o.p2 for o in self.overlaps)

  @property
  def selectoverlaprectangles(self):
    return overlapfilter(self.overlaprectangleindices)

class OverlapList(list, OverlapCollection):
  @property
  def overlaps(self): return self
