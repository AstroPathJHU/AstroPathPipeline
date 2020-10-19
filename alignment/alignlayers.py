import cv2, itertools, methodtools
from ..baseclasses.sample import ReadRectangles, ReadRectanglesBase
from .alignmentset import AlignmentSet, AlignmentSetBase
from .rectangle import AlignmentRectangleMultiLayer
from .overlap import AlignmentOverlap, LayerAlignmentResult
from .stitchlayers import stitchlayers

class OverlapForLayerAlignment(AlignmentOverlap):
  def __init__(self, *, p1, p2, x1, y1, x2, y2, layer1, layer2, tag, positionaloverlaps, **kwargs):
    super().__init__(p1=p1, p2=p2, x1=x1, y1=y1, x2=x1, y2=y1, layer1=layer1, layer2=layer2, tag=tag, **kwargs)
    self.__layeroverlaps = (
      AlignmentOverlap(p1=p1, p2=p1, x1=x1, y1=y1, x2=x1, y2=y1, layer1=layer1, layer2=layer2, tag=5, **kwargs),
      AlignmentOverlap(p1=p2, p2=p2, x1=x2, y1=y2, x2=x2, y2=y2, layer1=layer1, layer2=layer2, tag=5, **kwargs),
    )
    positionaloverlap1, = {_ for _ in positionaloverlaps if _.p1 == p1 and _.p2 == p2 and _.layer1 == _.layer2 == layer1}
    positionaloverlap2, = {_ for _ in positionaloverlaps if _.p1 == p1 and _.p2 == p2 and _.layer1 == _.layer2 == layer2}
    self.__positionaloverlaps = positionaloverlap1, positionaloverlap2

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

  def align(self, *args, **kwargs):
    staterrorimages = []

    for _ in self.__positionaloverlaps:
      _.align(alreadyalignedstrategy="skip")
      error = abs(_.shifted[0] - _.shifted[1])

      hpad = self.cutimages[0].shape[0] - error.shape[0]
      if _.result.dx > 0:
        right = hpad
        left = 0
      else:
        left = hpad
        right = 0

      vpad = self.cutimages[1].shape[1] - error.shape[1]
      if _.result.dx > 0:
        bottom = vpad
        top = 0
      else:
        top = vpad
        bottom = 0

      shifterrorback = cv2.copyMakeBorder(
        error,
        top=top,
        bottom=bottom,
        left=left,
        right=right,
        borderType=cv2.BORDER_REPLICATE,
      )
      staterrorimages.append(shifterrorback)

    return super().align(*args, staterrorimages=staterrorimages, **kwargs)

class SampleWithLayerOverlaps(ReadRectanglesBase):
  multilayer = True
  rectangletype = AlignmentRectangleMultiLayer
  overlaptype = OverlapForLayerAlignment
  alignmentresulttype = LayerAlignmentResult

  def overlapsdictkey(self, overlap):
    return super().overlapsdictkey(overlap) + overlap.layers
  def inverseoverlapsdictkey(self, overlap):
    return super().overlapsdictkey(overlap) + tuple(reversed(overlap.layers))

  def readalloverlaps(self):
    positionaloverlaps = []
    for l in self.layers:
      positionaloverlaps += super().readalloverlaps(overlaptype=AlignmentOverlap, layer1=l, layer2=l)
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
        positionaloverlaps=positionaloverlaps,
      )
      for i, (o, l1, l2) in enumerate([
        (o, l1, l2)
        for o in positionaloverlaps
        for l1, l2 in itertools.permutations(self.layers, 2)
        if o.layer1 == self.layers[0]
      ], start=1)
    ]

class AlignLayersBase(SampleWithLayerOverlaps, AlignmentSetBase, ReadRectanglesBase):
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
  def getDAPI(self, *args, writeimstat=False, **kwargs):
    return super().getDAPI(*args, writeimstat=writeimstat, **kwargs)

class AlignLayersForBroadbandFilterBase(AlignLayersBase):
  def __init__(self, *args, inputrectangles, inputoverlaps, broadbandfilter, selectoverlaps=None, **kwargs):
    if selectoverlaps is None:
      def selectoverlaps(o): return True
    def newselectoverlaps(o):
      if not selectoverlaps(o): return False
      alllayers = o.rectangles[0].layers
      layerindices = alllayers.index(o.layer1), alllayers.index(o.layer2)
      return all(o.rectangles[0].broadbandfilters[_] == broadbandfilter for _ in layerindices)
    self.__broadbandfilter = broadbandfilter
    self.__inputrectangles = inputrectangles
    self.__inputoverlaps = inputoverlaps
    super().__init__(*args, selectoverlaps=newselectoverlaps, **kwargs)
    self.logger.info("%d", broadbandfilter)

  @property
  def broadbandfilter(self): return self.__broadbandfilter
  def readallrectangles(self): return self.__inputrectangles
  def readalloverlaps(self): return self.__inputoverlaps

class AlignLayersForBroadbandFilter(AlignLayersForBroadbandFilterBase, AlignLayers):
  @property
  def alignmentsfilename(self): return self.csv(f"alignlayers_{self.broadbandfilter}")
  @property
  def stitchfilenames(self):
    return self.csv(f"layerpositions_{self.broadbandfilter}"), self.csv(f"layerpositioncovariances_{self.broadbandfilter}")

class AlignLayersByBroadbandFilter(SampleWithLayerOverlaps, ReadRectangles):
  def __init__(self, *args, filetype="flatWarp", **kwargs):
    super().__init__(*args, filetype=filetype, **kwargs)
    self.__step1s = [AlignLayersForBroadbandFilter(*args, broadbandfilter=i, inputrectangles=self.rectangles, inputoverlaps=self.overlaps, **kwargs) for i in sorted(set(self.rectangles[0].broadbandfilters))]

  def getDAPI(self, *args, **kwargs):
    for _ in self.__step1s:
      _.getDAPI(*args, **kwargs)

  def alignstep1(self, *args, **kwargs):
    for _ in self.__step1s:
      _.align(*args, **kwargs)
  def stitchstep1(self, *args, **kwargs):
    for _ in self.__step1s:
      _.stitch(*args, **kwargs)

  @property
  def logmodule(self):
    return "alignlayers"
