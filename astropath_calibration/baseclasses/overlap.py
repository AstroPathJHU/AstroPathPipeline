import abc, networkx as nx, numpy as np, pathlib

from ..baseclasses.csvclasses import constantsdict
from ..utilities.tableio import readtable
from ..utilities.units.dataclasses import DataClassWithPscale, distancefield
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

  pixelsormicrons = "microns"

  n: int
  p1: int
  p2: int
  x1: distancefield(pixelsormicrons=pixelsormicrons)
  y1: distancefield(pixelsormicrons=pixelsormicrons)
  x2: distancefield(pixelsormicrons=pixelsormicrons)
  y2: distancefield(pixelsormicrons=pixelsormicrons)
  tag: int

  def __user_init__(self, *, nclip, rectangles, **kwargs):
    super().__user_init__(**kwargs)

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
  def layer(self):
    """
    Layer number of the rectangles in the overlap
    """
    try:
      layers = [r.layer for r in self.rectangles]
    except KeyError:
      raise TypeError("Trying to get layer for overlap whose rectangles don't have a layer assigned")
    if layers[0] != layers[1]:
      raise ValueError(f"Rectangles have inconsistent layers: {layers}")
    return layers[0]

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

class OverlapCollection(abc.ABC):
  """
  Base class for a group of overlaps.
  """
  @abc.abstractproperty
  def overlaps(self): pass

  def overlapgraph(self, useexitstatus=False):
    """
    Get a networkx graph object.
    It has a node for each rectangle id and an edge for
    each overlap in the collection.

    If useexitstatus is true (which will only work if the overlaps
    have been aligned), it will only include overlaps with an exit
    status of 0.
    """
    g = nx.DiGraph()
    for o in self.overlaps:
      if useexitstatus and o.result.exit: continue
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

class RectangleOverlapCollection(RectangleCollection, OverlapCollection):
  """
  A collection of both rectangles and overlaps.  The overlapgraph
  has metadata for the nodes and also includes rectangles that
  aren't in an overlap.
  """
  def overlapgraph(self, *args, **kwargs):
    g = super().overlapgraph(*args, **kwargs)
    for r in self.rectangles:
      g.add_node(r.n, rectangle=r)
    return g

class RectangleOverlapList(RectangleOverlapCollection):
  """
  Contains a list of rectangles and a list of overlaps.
  """
  def __init__(self, rectangles, overlaps):
    self.__rectangles = rectangles
    self.__overlaps = overlaps

  @property
  def rectangles(self): return self.__rectangles
  @property
  def overlaps(self): return self.__overlaps

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
