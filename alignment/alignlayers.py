import itertools
from ..baseclasses.sample import ReadRectanglesBase
from .alignmentset import AlignmentSet, AlignmentSetBase
from .rectangle import AlignmentRectangleMultiLayer
from .overlap import LayerAlignmentResult

class AlignLayersBase(AlignmentSetBase, ReadRectanglesBase):
  multilayer = True
  rectangletype = AlignmentRectangleMultiLayer
  alignmentresulttype = LayerAlignmentResult

  def overlapsdictkey(self, overlap):
    return super().overlapsdictkey(overlap) + overlap.layers
  def inverseoverlapsdictkey(self, overlap):
    return super().inverseoverlapsdictkey(overlap) + tuple(reversed(overlap.layers))

  def readalloverlaps(self):
    return [
      self.overlaptype(
        n=r.n,
        p1=r.n,
        p2=r.n,
        x1=r.x,
        y1=r.y,
        x2=r.x,
        y2=r.y,
        tag=5,
        layer1=l1,
        layer2=l2,
        nclip=self.nclip,
        rectangles=(r,),
        pscale=self.pscale,
        readingfromfile=False,
      )
      for r in self.rectangles
      for l1, l2 in itertools.permutations(self.layers, 2)
    ]

class AlignLayers(AlignLayersBase, AlignmentSet):
  @property
  def alignmentsfilename(self): return self.csv("alignlayers")
