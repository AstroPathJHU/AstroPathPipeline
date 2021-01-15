import cv2, itertools, methodtools, numpy as np
from ..baseclasses.sample import ReadRectanglesOverlapsBase, ReadRectanglesOverlapsIm3, ReadRectanglesOverlapsIm3Base
from .alignmentset import AlignmentSet, AlignmentSetBase
from .rectangle import AlignmentRectangleMultiLayer, RectanglePCAByBroadbandFilter
from .overlap import AlignmentOverlap, LayerAlignmentResult
from .stitchlayers import stitchlayers

class OverlapForLayerAlignment(AlignmentOverlap):
  @classmethod
  def transforminitargs(cls, *, p1, p2, x1, y1, x2, y2, layer1, layer2, tag, **kwargs):
    return (), {
      "p1": p1,
      "p2": p2,
      "x1": x1,
      "y1": y1,
      "x2": x1,
      "y2": y1,
      "_x2": x2,
      "_y2": y2,
      "layer1": layer1,
      "layer2": layer2,
      "tag": tag,
      **kwargs,
    }
  def __user_init__(self, *, _x2, _y2, layer1, layer2, positionaloverlaps, **kwargs):
    super().__user_init__(layer1=layer1, layer2=layer2, **kwargs)
    import pprint; pprint.pprint(kwargs)
    self.__layeroverlaps = (
      AlignmentOverlap(n=self.n, p1=self.p1, p2=self.p1, x1=self.x1, y1=self.y1, x2=self.x1, y2=self.y1, layer1=layer1, layer2=layer2, tag=5, **kwargs),
      AlignmentOverlap(n=self.n, p1=self.p2, p2=self.p2, x1=_x2,     y1=_y2,     x2=_x2,     y2=_y2,     layer1=layer1, layer2=layer2, tag=5, **kwargs),
    )
    positionaloverlap1, = {_ for _ in positionaloverlaps if _.p1 == self.p1 and _.p2 == self.p2 and _.layer1 == _.layer2 == layer1}
    positionaloverlap2, = {_ for _ in positionaloverlaps if _.p1 == self.p1 and _.p2 == self.p2 and _.layer1 == _.layer2 == layer2}
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

class SampleWithLayerOverlaps(ReadRectanglesOverlapsBase):
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

class AlignLayersBase(SampleWithLayerOverlaps, AlignmentSetBase, ReadRectanglesOverlapsIm3Base):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.logger.warningglobal("Layer alignment is not fully implemented.  Results are not necessarily going to be accurate.")
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

class AlignBroadbandFiltersBase(AlignLayersBase):
  def __init__(self, *args, step1s, inputrectangles, **kwargs):
    self.__step1s = step1s
    self.__inputrectangles = inputrectangles
    super().__init__(*args, **kwargs)

  def readallrectangles(self):
    stitchresultdict = {
      layer: step1.stitchresult
      for step1 in self.__step1s
      for layer in step1.stitchresult.layers
    }
    r = self.__inputrectangles[0]
    layershifts=np.array([
      stitchresultdict[layer].x(layer) for layer in r.layers
    ])
    return [RectanglePCAByBroadbandFilter(originalrectangle=_, layershifts=layershifts) for _ in self.__inputrectangles]

  @property
  def layers(self):
    return list(range(1, len(self.__step1s)+1))

class AlignBroadbandFilters(AlignBroadbandFiltersBase, AlignLayers):
  @property
  def alignmentsfilename(self): return self.csv("alignbroadbandfilters")
  @property
  def stitchfilenames(self):
    return self.csv("broadbandfilterpositions"), self.csv("broadbandpositioncovariances")

class AlignLayersByBroadbandFilter(SampleWithLayerOverlaps, ReadRectanglesOverlapsIm3):
  def __init__(self, *args, filetype="flatWarp", **kwargs):
    super().__init__(*args, filetype=filetype, **kwargs)
    self.__step1s = [AlignLayersForBroadbandFilter(*args, broadbandfilter=i, inputrectangles=self.rectangles, inputoverlaps=self.overlaps, **kwargs) for i in sorted(set(self.rectangles[0].broadbandfilters))]
    self.__initargs = args
    self.__initkwargs = kwargs

  def getDAPIstep1(self, *args, **kwargs):
    for _ in self.__step1s:
      _.getDAPI(*args, **kwargs)

  def alignstep1(self, *args, **kwargs):
    for _ in self.__step1s:
      _.align(*args, **kwargs)
  def stitchstep1(self, *args, **kwargs):
    for _ in self.__step1s:
      _.stitch(*args, **kwargs)
  def readstep1alignments(self, *args, **kwargs):
    for _ in self.__step1s:
      _.readalignments(*args, **kwargs)
  def readstep1stitchresults(self, *args, **kwargs):
    for _ in self.__step1s:
      _.readstitchresult(*args, **kwargs)

  @methodtools.lru_cache()
  @property
  def __step2(self):
    return AlignBroadbandFilters(*self.__initargs, step1s=self.__step1s, inputrectangles=self.rectangles, **self.__initkwargs)

  def getDAPIstep2(self, *args, **kwargs):
    self.__step2.getDAPI(*args, **kwargs)
  def alignstep2(self, *args, **kwargs):
    self.__step2.align(*args, **kwargs)
  def stitchstep2(self, *args, **kwargs):
    self.__step2.stitch(*args, **kwargs)
  def readstep2alignments(self, *args, **kwargs):
    self.__step2.readalignments(*args, **kwargs)
  def readstep2stitchresult(self, *args, **kwargs):
    self.__step2.readstitchresult(*args, **kwargs)

  @property
  def logmodule(self):
    return "alignlayers"
