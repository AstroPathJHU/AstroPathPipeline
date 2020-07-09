import abc, dataclasses, networkx as nx, numpy as np, pathlib

from ..baseclasses.csvclasses import Constant
from ..utilities import units
from ..utilities.tableio import readtable
from ..utilities.units.dataclasses import DataClassWithDistances, distancefield
from .rectangle import Rectangle, RectangleCollection, RectangleList, rectangleoroverlapfilter, rectangleoroverlapfilter as overlapfilter

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
  readingfromfile: dataclasses.InitVar[bool]

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

  def islands(self, *args, **kwargs):
    return list(nx.strongly_connected_components(self.overlapgraph(*args, **kwargs)))

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

class RectangleOverlapCollection(RectangleCollection, OverlapCollection):
  def overlapgraph(self, *args, **kwargs):
    g = super().overlapgraph(*args, **kwargs)
    for r in self.rectangles:
      g.add_node(r.n)
    return g

class RectangleOverlapList(RectangleOverlapCollection):
  def __init__(self, rectangles, overlaps):
    self.__rectangles = rectangles
    self.__overlaps = overlaps

  @property
  def rectangles(self): return self.__rectangles
  @property
  def overlaps(self): return self.__overlaps

def rectangleoverlaplist_fromcsvs(dbloadfolder, *, selectrectangles=None, selectoverlaps=None, onlyrectanglesinoverlaps=False):
  dbload = pathlib.Path(dbloadfolder)
  samp = dbload.parent.name
  tmp = readtable(dbload/(samp+"_constants.csv"), Constant, extrakwargs={"pscale": 1})
  pscale = {_.value for _ in tmp if _.name == "pscale"}.pop()
  constants     = readtable(dbload/(samp+"_constants.csv"), Constant, extrakwargs={"pscale": pscale})
  constantsdict = {constant.name: constant.value for constant in constants}
  layer = constantsdict["layer"]
  nclip = constantsdict["nclip"]

  rectanglefilter = rectangleoroverlapfilter(selectrectangles)
  _overlapfilter = rectangleoroverlapfilter(selectoverlaps)
  overlapfilter = lambda o: _overlapfilter(o) and o.p1 in rectangles.rectangleindices and o.p2 in rectangles.rectangleindices

  rectangles  = readtable(dbload/(samp+"_rect.csv"), Rectangle, extrakwargs={"pscale": pscale})
  rectangles = RectangleList([r for r in rectangles if rectanglefilter(r)])
  overlaps  = readtable(dbload/(samp+"_overlap.csv"), Overlap, filter=lambda row: row["p1"] in rectangles.rectangleindices and row["p2"] in rectangles.rectangleindices, extrakwargs={"pscale": pscale, "layer": layer, "rectangles": rectangles, "nclip": nclip})
  overlaps = OverlapList([o for o in overlaps if overlapfilter(o)])
  if onlyrectanglesinoverlaps:
    oldfilter = rectanglefilter
    rectanglefilter = lambda r: oldfilter(r) and overlaps.selectoverlaprectangles(r)
    rectangles = [r for r in rectangles if rectanglefilter(r)]
  return RectangleOverlapList(rectangles, overlaps)

