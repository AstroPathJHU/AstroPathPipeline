import itertools
from ..baseclasses.sample import ReadRectanglesBase
from .alignmentset import AlignmentSet, AlignmentSetBase
from .rectangle import AlignmentRectangleMultiLayer
from .overlap import AlignmentOverlap, LayerAlignmentResult
from .stitchlayers import stitchlayers

class OverlapForLayerAlignment(AlignmentOverlap):
  def __init__(self, *, p1, p2, x1, y1, x2, y2, layer1, layer2, tag, **kwargs):
    super().__init__(p1=p1, p2=p2, x1=x1, y1=y1, x2=x2, y2=y2, layer1=layer1, layer2=layer2, tag=tag, **kwargs)
    self.__layeroverlaps = (
      AlignmentOverlap(p1=p1, p2=p1, x1=x1, y1=y1, x2=x1, y2=y1, layer1=layer1, layer2=layer2, tag=5, **kwargs),
      AlignmentOverlap(p1=p2, p2=p2, x1=x2, y1=y2, x2=x2, y2=y2, layer1=layer1, layer2=layer2, tag=5, **kwargs),
    )
    self.__positionaloverlaps = (
      AlignmentOverlap(p1=p1, p2=p2, x1=x1, y1=y1, x2=x2, y2=y2, layer1=layer1, layer2=layer1, tag=tag, **kwargs),
      AlignmentOverlap(p1=p1, p2=p2, x1=x1, y1=y1, x2=x2, y2=y2, layer1=layer2, layer2=layer2, tag=tag, **kwargs),
    )

  def isinverseof(self, inverse):
    return (inverse.p1, inverse.p2) == (self.p1, self.p2) and inverse.layers == tuple(reversed(self.layers))

  @property
  def images(self):
    return self.__positionaloverlaps[0].images[0], self.__positionaloverlaps[1].images[0]
  @property
  def images2(self):
    return self.__positionaloverlaps[0].images[1], self.__positionaloverlaps[1].images[1]

  @property
  def cutimages(self):
    return self.__positionaloverlaps[0].cutimages[0], self.__positionaloverlaps[1].cutimages[0]
  @property
  def cutimages2(self):
    return self.__positionaloverlaps[0].cutimages[1], self.__positionaloverlaps[1].cutimages[1]

class AlignLayersBase(AlignmentSetBase, ReadRectanglesBase):
  multilayer = True
  rectangletype = AlignmentRectangleMultiLayer
  overlaptype = OverlapForLayerAlignment
  alignmentresulttype = LayerAlignmentResult

  def overlapsdictkey(self, overlap):
    return super().overlapsdictkey(overlap) + overlap.layers
  def inverseoverlapsdictkey(self, overlap):
    return super().overlapsdictkey(overlap) + tuple(reversed(overlap.layers))

  def readalloverlaps(self):
    return [
      self.overlaptype(
        n=i,
        p1=o.p1,
        p2=o.p2,
        x1=o.x1,
        y1=o.y1,
        x2=o.x2,
        y2=o.y2,
        tag=o.tag,
        layer1=l1,
        layer2=l2,
        nclip=self.nclip,
        rectangles=self.rectangles,
        pscale=self.pscale,
        readingfromfile=False,
      )
      for i, (o, l1, l2) in enumerate([
        (o, l1, l2)
        for o in super().readalloverlaps(layer1=1, layer2=1)
        for l1, l2 in itertools.permutations(self.layers, 2)
      ], start=1)
    ]

  def dostitching(self, **kwargs):
    return stitchlayers(overlaps=self.overlaps, logger=self.logger, **kwargs)
  def applystitchresult(self, result):
    result.applytooverlaps()
    self.stitchresult = result

class AlignLayers(AlignLayersBase, AlignmentSet):
  @property
  def alignmentsfilename(self): return self.csv("alignlayers")
  @property
  def stitchfilenames(self):
    return self.csv("layerpositions"), self.csv("layerpositioncovariances")
  def stitch(self, *, saveresult=True, **kwargs):
    result = super().stitch(saveresult=saveresult, **kwargs)
    if saveresult:
      self.writestitchresult(result)
    return result
