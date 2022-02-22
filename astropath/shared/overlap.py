import abc, methodtools, networkx as nx, numpy as np, pathlib

from ..utilities import units
from ..utilities.tableio import readtable
from ..utilities.units.dataclasses import DataClassWithPscale, distancefield
from .csvclasses import constantsdict
from .rectangle import Rectangle, RectangleCollection, RectangleList, rectangleoroverlapfilter, rectangleoroverlapfilter as overlapfilter

class Overlap(DataClassWithPscale):
  """
  Overlap between two HPFs.

  n: id of the overlap, starting from 1
  p1, p2: ids of the two HPFs
  x1, y1, x2, y2: location of the two HPFs
  tag: gives the relative location of the HPFs:
    1 2 3
    4 5 6
    7 8 9
  nclip: how many pixels to cut off when doing alignment
  rectangles: a list of rectangles, which can be just p1 and p2
              but can also include others if that's easier
  """

  n: int
  p1: int
  p2: int
  x1: units.Distance = distancefield(pixelsormicrons="microns")
  y1: units.Distance = distancefield(pixelsormicrons="microns")
  x2: units.Distance = distancefield(pixelsormicrons="microns")
  y2: units.Distance = distancefield(pixelsormicrons="microns")
  tag: int

  def __post_init__(self, *, nclip, rectangles, **kwargs):
    super().__post_init__(**kwargs)

    self.nclip = nclip
    self.result = None

    self.updaterectangles(rectangles)

  def updaterectangles(self, rectangles):
    """
    update the rectangles in the overlap with a new list
    rectangles: a list of rectangles, which can be just p1 and p2
                but can also include others if that's easier
    """
    p1rect = None; p2rect=None
    for r in rectangles :
      if r.n==self.p1 :
        p1rect = r
      if r.n==self.p2 :
        p2rect = r
      if (p1rect is not None) and (p2rect is not None) :
        break
    if (p1rect is None) or (p2rect is None):
      raise ValueError(f"Searched for rectangles with n=p1={self.p1} and n=p2={self.p2} but p1rect={p1rect} and p2rect={p2rect}")
    self.rectangles = p1rect, p2rect

  @property
  def x1vec(self):
    """
    [x1, y1] as a numpy array
    """
    return np.array([self.x1, self.y1])
  @property
  def x2vec(self):
    """
    [x2, y2] as a numpy array
    """
    return np.array([self.x2, self.y2])

class OverlapCollection(units.ThingWithPscale):
  """
  Base class for a group of overlaps.
  """
  @property
  @abc.abstractmethod
  def overlaps(self): pass

  @methodtools.lru_cache()
  def overlapgraph(self, useexitstatus=False, skipoverlaps=()):
    """
    Get a networkx graph object.
    It has a node for each rectangle id and an edge for
    each overlap in the collection.

    If useexitstatus is true (which will only work if the overlaps
    have been aligned), it will only include overlaps with an exit
    status of 0.
    """
    if skipoverlaps is None: skipoverlaps = []
    g = nx.DiGraph()
    for o in self.overlaps:
      if useexitstatus and o.result.exit: continue
      if o in skipoverlaps: continue
      g.add_edge(o.p1, o.p2, overlap=o)

    return g

  def nislands(self, *args, **kwargs):
    """
    Number of islands in the overlap graph.

    If useexitstatus is true (which will only work if the overlaps
    have been aligned), it will only include overlaps with an exit
    status of 0.
    """
    return nx.number_strongly_connected_components(self.overlapgraph(*args, **kwargs))

  def islands(self, *args, **kwargs):
    """
    List of islands in the overlap graph.

    If useexitstatus is true (which will only work if the overlaps
    have been aligned), it will only include overlaps with an exit
    status of 0.
    """
    return list(nx.strongly_connected_components(self.overlapgraph(*args, **kwargs)))

  def overlapsforrectangle(self, rectangle_n, *args, **kwargs):
    """
    Return the overlaps for a given rectangle as graph edges

    rectangle_n: the identifier of the rectangle whose overlaps should be returned
    other arguments can be anything passed to overlapgraph or DiGraph.edges()
    """
    return list(self.overlapgraph(*args,**kwargs).edges(rectangle_n))

  def overlapsdictkey(self, overlap):
    """
    Key to be used for computing overlaps dict (can be overridden in subclasses)
    """
    return overlap.p1, overlap.p2

  @property
  def overlapsdict(self):
    """
    Gives a dict that can be used to access overlaps by the overlapsdictkey,
    by default the two rectangle ids.
    """
    result = {}
    for o in self.overlaps:
      key = self.overlapsdictkey(o)
      if key in result:
        raise KeyError(f"Multiple overlaps with key {key}")
      result[key] = o
    return result

  @property
  def overlaprectangleindices(self):
    """
    Gives the ids of all rectangles in overlaps.
    """
    return frozenset(o.p1 for o in self.overlaps) | frozenset(o.p2 for o in self.overlaps)

  @property
  def selectoverlaprectangles(self):
    """
    Gives a filter that returns whether or not the rectangle is in
    one of the overlaps.
    """
    return overlapfilter(self.overlaprectangleindices)

class OverlapList(list, OverlapCollection):
  """
  A list of overlaps with all the functionality of OverlapCollections.
  """
  @property
  def overlaps(self): return self

  @property
  def pscale(self):
    result, = {o.pscale for o in self.overlaps}
    return result

class RectangleOverlapCollection(RectangleCollection, OverlapCollection):
  """
  A collection of both rectangles and overlaps.  The overlapgraph
  has metadata for the nodes and also includes rectangles that
  aren't in an overlap.
  """
  def overlapgraph(self, *args, gridatol=None, skipoverlaps=None, **kwargs):
    if skipoverlaps is None: skipoverlaps = []
    skipoverlaps = list(skipoverlaps)
    if gridatol is not None:
      for overlap in self.overlaps:
        offset = overlap.x1vec - overlap.x2vec
        if not np.all(units.np.isclose(offset, 0, atol=gridatol) | units.np.isclose(abs(offset), self.hpfoffset, atol=gridatol)):
          skipoverlaps.append(overlap)
    g = super().overlapgraph(*args, skipoverlaps=tuple(skipoverlaps), **kwargs)
    for r in self.rectangles:
      g.add_node(r.n, rectangle=r)
    return g

  @property
  def tissue_edge_rects(self) :
    return [r for r in self.rectangles if len(self.overlapsforrectangle(r.n))<8]
  @property
  def tissue_bulk_rects(self) :
    return [r for r in self.rectangles if len(self.overlapsforrectangle(r.n))==8]

  def neighbors(self, rect, **kwargs):
    result = {}
    g = self.overlapgraph(**kwargs)
    for p1, p2, data in g.in_edges(rect.n, data=True):
      o = data["overlap"]
      assert p2 == o.p2 == rect.n
      assert p1 == o.p1
      result[o.tag] = self.rectangles[self.rectangledict[p1]]
    return result

class RectangleOverlapList(RectangleOverlapCollection):
  """
  Contains a list of rectangles and a list of overlaps.
  """
  def __init__(self, rectangles, overlaps):
    self.__rectangles = RectangleList(rectangles)
    self.__overlaps = OverlapList(overlaps)

  @property
  def rectangles(self): return self.__rectangles
  @property
  def overlaps(self): return self.__overlaps
  @methodtools.lru_cache()
  @property
  def pscale(self):
    result, = {self.rectangles.pscale, self.overlaps.pscale}
    return result

def rectangleoverlaplist_fromcsvs(dbloadfolder, *, layer, selectrectangles=None, selectoverlaps=None, onlyrectanglesinoverlaps=False, rectangletype=Rectangle, overlaptype=Overlap):
  """
  Standalone function to assemble lists of rectangles and overlaps from
  the dbload folder.
  """
  dbload = pathlib.Path(dbloadfolder)
  samp = dbload.parent.name
  dct = constantsdict(dbload/f"{samp}_constants.csv")
  nclip = dct["nclip"]
  pscale = dct["pscale"]

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
