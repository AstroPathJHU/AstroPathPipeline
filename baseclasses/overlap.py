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
  pscale: dataclasses.InitVar[float]
  nclip: dataclasses.InitVar[float]
  rectangles: dataclasses.InitVar[float]
  readingfromfile: dataclasses.InitVar[bool]

  def __post_init__(self, pscale, nclip, rectangles, readingfromfile=False):
    super().__post_init__(pscale=pscale, readingfromfile=readingfromfile)

    self.nclip = nclip
    self.result = None

    self.updaterectangles(rectangles)

  def updaterectangles(self, rectangles):
    p1rect = None; p2rect=None
    for r in rectangles :
      if (p1rect is not None) and (p2rect is not None) :
        break
      elif r.n==self.p1 :
        p1rect = r
      elif r.n==self.p2 :
        p2rect = r
    if (p1rect is None) or (p2rect is None):
      raise ValueError(f"Searched for rectangles with n=p1={self.p1} and n=p2={self.p2} but p1rect={p1rect} and p2rect={p2rect}")
    self.rectangles = p1rect, p2rect

  @property
  def layer(self):
    try:
      layers = [r.layer for r in self.rectangles]
    except KeyError:
      raise TypeError("Trying to get layer for overlap whose rectangles don't have a layer assigned")
    if layers[0] != layers[1]:
      raise ValueError(f"Rectangles have inconsistent layers: {layers}")
    return layers[0]

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

def rectangleoverlaplist_fromcsvs(dbloadfolder, *, selectrectangles=None, selectoverlaps=None, onlyrectanglesinoverlaps=False, layer_override=None, rectangletype=Rectangle, overlaptype=Overlap):
  dbload = pathlib.Path(dbloadfolder)
  samp = dbload.parent.name
  tmp = readtable(dbload/(samp+"_constants.csv"), Constant, extrakwargs={"pscale": 1})
  pscale = {_.value for _ in tmp if _.name == "pscale"}.pop()
  constants     = readtable(dbload/(samp+"_constants.csv"), Constant, extrakwargs={"pscale": pscale})
  constantsdict = {constant.name: constant.value for constant in constants}
  layer = constantsdict["layer"] if "layer" in constantsdict.keys() else layer_override
  if layer is None :
    raise RuntimeError(f"""ERROR: {samp}_constants.csv file in {dbload} does not have a layer variable, 
                           and layer_override in rectangleoverlaplist_fromcsvs is {layer_override}!""")
  nclip = constantsdict["nclip"]

  rectanglefilter = rectangleoroverlapfilter(selectrectangles)
  _overlapfilter = rectangleoroverlapfilter(selectoverlaps)
  overlapfilter = lambda o: _overlapfilter(o) and o.p1 in rectangles.rectangleindices and o.p2 in rectangles.rectangleindices

  rectangles  = readtable(dbload/(samp+"_rect.csv"), rectangletype, extrakwargs={"pscale": pscale})
  rectangles = RectangleList([r for r in rectangles if rectanglefilter(r)])
  overlaps  = readtable(dbload/(samp+"_overlap.csv"), overlaptype, filter=lambda row: row["p1"] in rectangles.rectangleindices and row["p2"] in rectangles.rectangleindices, extrakwargs={"pscale": pscale, "rectangles": rectangles, "nclip": nclip})
  overlaps = OverlapList([o for o in overlaps if overlapfilter(o)])
  if onlyrectanglesinoverlaps:
    oldfilter = rectanglefilter
    rectanglefilter = lambda r: oldfilter(r) and overlaps.selectoverlaprectangles(r)
    rectangles = [r for r in rectangles if rectanglefilter(r)]
  return RectangleOverlapList(rectangles, overlaps)
