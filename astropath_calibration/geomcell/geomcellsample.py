import cv2, itertools, matplotlib.pyplot as plt, more_itertools, numpy as np, scipy.ndimage, skimage.measure, skimage.morphology
from ..alignment.alignmentset import AlignmentSet
from ..alignment.field import Field, FieldReadComponentTiffMultiLayer
from ..baseclasses.csvclasses import constantsdict
from ..baseclasses.polygon import DataClassWithPolygon, polygonfield
from ..baseclasses.rectangle import GeomLoadRectangle, rectanglefilter
from ..baseclasses.sample import DbloadSample, GeomSampleBase, ReadRectanglesDbloadComponentTiff, WorkflowSample
from ..geom.contours import findcontoursaspolygons
from ..utilities import units
from ..utilities.misc import dict_product, dummylogger
from ..utilities.tableio import readtable, writetable
from ..utilities.units import ThingWithApscale, ThingWithPscale
from ..utilities.units.dataclasses import distancefield

class GeomLoadField(Field, GeomLoadRectangle):
  pass

class GeomLoadFieldReadComponentTiffMultiLayer(FieldReadComponentTiffMultiLayer, GeomLoadRectangle):
  pass

class GeomCellSample(GeomSampleBase, ReadRectanglesDbloadComponentTiff, DbloadSample, WorkflowSample):
  MEMBRANE_TUMOR = 13
  MEMBRANE_IMMUNE = 12
  NUCLEUS_TUMOR = 11
  NUCLEUS_IMMUNE = 10

  @classmethod
  def ismembrane(cls, layer):
    return {
      cls.MEMBRANE_TUMOR: True,
      cls.MEMBRANE_IMMUNE: True,
      cls.NUCLEUS_TUMOR: False,
      cls.NUCLEUS_IMMUNE: False,
    }[layer]

  def __init__(self, *args, **kwargs):
    super().__init__(
      *args,
      layers=[
        self.MEMBRANE_TUMOR,
        self.MEMBRANE_IMMUNE,
        self.NUCLEUS_TUMOR,
        self.NUCLEUS_IMMUNE,
      ],
      **kwargs
    )

  multilayer = True

  @property
  def rectanglecsv(self): return "fields"
  rectangletype = GeomLoadFieldReadComponentTiffMultiLayer
  @property
  def rectangleextrakwargs(self):
    return {
      **super().rectangleextrakwargs,
      "with_seg": True,
      "geomfolder": self.geomfolder,
    }

  @classmethod
  def logmodule(self):
    return "geomcell"

  def rungeomcell(self, *, _debugdraw=(), _debugdrawonerror=False, _onlydebug=False):
    self.geomfolder.mkdir(exist_ok=True, parents=True)
    if not _debugdraw: _onlydebug = False
    nfields = len(self.rectangles)
    for i, field in enumerate(self.rectangles, start=1):
      if _onlydebug and not any(fieldn == field.n for fieldn, celltype, celllabel in _debugdraw): continue
      self.logger.info(f"writing cells for field {field.n} ({i} / {nfields})")
      geomload = []
      with field.using_image() as im:
        im = im.astype(np.uint32)
        for celltype, (imlayernumber, imlayer) in enumerate(more_itertools.zip_equal(self.layers, im)):
          properties = skimage.measure.regionprops(imlayer)
          for cellproperties in properties:
            if not np.any(cellproperties.image):
              assert False
              continue
            celllabel = cellproperties.label
            if _onlydebug and (field.n, celltype, celllabel) not in _debugdraw: continue
            thiscell = imlayer==celllabel
            polygon = PolygonFinder(thiscell, ismembrane=self.ismembrane(imlayernumber), bbox=cellproperties.bbox, pxvec=units.nominal_values(field.pxvec), mxbox=field.mxbox, pscale=self.pscale, apscale=self.apscale, logger=self.logger, loginfo=f"{field.n} {celltype} {celllabel}", _debugdraw=(field.n, celltype, celllabel) in _debugdraw, _debugdrawonerror=_debugdrawonerror).findpolygon()

            box = np.array(cellproperties.bbox).reshape(2, 2) * self.onepixel * 1.0
            box += units.nominal_values(field.pxvec)
            box = box // self.onepixel * self.onepixel

            geomload.append(
              CellGeomLoad(
                field=field.n,
                ctype=celltype,
                n=celllabel,
                box=box,
                poly=polygon,
                pscale=self.pscale,
                apscale=self.apscale,
              )
            )

      writetable(field.geomloadcsv, geomload)

  @property
  def inputfiles(self):
    return [
      self.csv("constants"),
      self.csv("fields"),
      *(r.imagefile for r in self.rectangles),
    ]

  @property
  def workflowkwargs(self):
    return {"selectrectangles": rectanglefilter([r.n for r in self.rectangles]), **super().workflowkwargs}

  @classmethod
  def getoutputfiles(cls, SlideID, *, dbloadroot, geomroot, selectrectangles=lambda r: True, **otherworkflowkwargs):
    dbload = dbloadroot/SlideID/"dbload"
    fieldscsv = dbload/f"{SlideID}_fields.csv"
    constantscsv = dbload/f"{SlideID}_constants.csv"
    if not fieldscsv.exists(): return [fieldscsv]
    constants = constantsdict(constantscsv)
    rectangles = readtable(fieldscsv, GeomLoadField, extrakwargs={"pscale": constants["pscale"], "geomfolder": geomroot/SlideID/"geom"})
    return [
      *(r.geomloadcsv for r in rectangles if selectrectangles(r)),
    ]

  @classmethod
  def workflowdependencies(cls):
    return [AlignmentSet] + super().workflowdependencies()

class CellGeomLoad(DataClassWithPolygon):
  pixelsormicrons = "pixels"
  field: int
  ctype: int
  n: int
  x: distancefield(pixelsormicrons=pixelsormicrons)
  y: distancefield(pixelsormicrons=pixelsormicrons)
  w: distancefield(pixelsormicrons=pixelsormicrons)
  h: distancefield(pixelsormicrons=pixelsormicrons)
  poly: polygonfield()

  @classmethod
  def transforminitargs(cls, *args, box=None, **kwargs):
    boxkwargs = {}
    if box is not None:
      boxkwargs["x"], boxkwargs["y"] = box[0]
      boxkwargs["w"], boxkwargs["h"] = box[1] - box[0]
    return super().transforminitargs(
      *args,
      **kwargs,
      **boxkwargs,
    )



class PolygonFinder(ThingWithPscale, ThingWithApscale):
  def __init__(self, cellmask, *, ismembrane, bbox, pscale, apscale, pxvec, mxbox, _debugdraw=False, _debugdrawonerror=False, logger=dummylogger, loginfo=""):
    self.originalcellmask = self.cellmask = cellmask
    self.ismembrane = ismembrane
    self.__bbox = bbox
    self.logger = logger
    self.loginfo = loginfo
    self.__pscale = pscale
    self.__apscale = apscale
    self.pxvec = pxvec
    self.mxbox = mxbox
    self._debugdraw = _debugdraw
    self._debugdrawonerror = _debugdrawonerror
    if self._debugdraw:
      self.originalcellmask = self.cellmask.copy()

  @property
  def pscale(self): return self.__pscale
  @property
  def apscale(self): return self.__apscale

  def findpolygon(self):
    polygon = None
    try:
      if self.isprimary:
        if self.ismembrane:
          self.joinbrokenmembrane()
        self.connectdisjointregions()
      else:
        self.pickbiggestregion()
      polygon, = self.__findpolygons(cellmask=self.slicedmask.astype(np.uint8))

      if self.isprimary and self.ismembrane:
        if self.istoothin(polygon):
          self.logger.warningglobal(f"Long, thin polygon (perimeter = {polygon.perimeter / self.onepixel} pixels, area = {polygon.area / self.onepixel**2} pixels^2) - possibly a broken membrane that couldn't be fixed? {self.loginfo}")
    except:
      if self._debugdrawonerror: self._debugdraw = True
      raise
    finally:
      self.debugdraw(polygon)

    return polygon

  def __findpolygons(self, cellmask):
    top, left, bottom, right = self.adjustedbbox
    shiftby = self.pxvec + np.array([left, top]) * self.onepixel
    polygons = findcontoursaspolygons(cellmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE, pscale=self.pscale, apscale=self.apscale, shiftby=shiftby, fill=True)
    if len(polygons) > 1:
      polygons.sort(key=lambda x: x.area, reverse=True)
    return polygons

  def istoothin(self, polygon):
    area = polygon.area
    perimeter = polygon.perimeter
    return area / perimeter <= .8 * self.onepixel

  @property
  def adjustedbbox(self):
    top, left, bottom, right = self.__bbox
    if self.ismembrane:
      if top == 1: top = 0
      if left == 1: left = 0
      height, width = self.cellmask.shape
      if bottom == height - 1: bottom = height
      if right == width - 1: right = width
    return np.array([top, left, bottom, right])

  @property
  def isprimary(self):
    top, left, bottom, right = self.adjustedbbox * self.onepixel
    ptop, pleft, pbottom, pright = self.mxbox
    pleft -= self.pxvec[0]
    pright -= self.pxvec[0]
    ptop -= self.pxvec[1]
    pbottom -= self.pxvec[1]
    return bottom >= ptop and pbottom >= top and right >= pleft and pright >= left

  @property
  def isonedge(self):
    result = [0, 0]
    top, left, bottom, right = self.adjustedbbox
    height, width = self.cellmask.shape
    if top == 0: result[0] = -1
    if bottom == height: result[0] = 1
    if left == 0: result[1] = -1
    if right == width: result[1] = 1
    return result

  @property
  def bboxslice(self):
    top, left, bottom, right = self.adjustedbbox
    return slice(top, bottom+1), slice(left, right+1)

  @property
  def slicedmask(self):
    return self.cellmask[self.bboxslice]

  def joinbrokenmembrane(self):
    slicedmask = self.slicedmask.astype(np.uint8)

    polygons = self.__findpolygons(cellmask=slicedmask)
    if not self.istoothin(polygons[0]): return

    #find the endpoints: pixels of membrane that have exactly one membrane neighbor
    nneighbors = scipy.ndimage.convolve(slicedmask, [[1, 1, 1], [1, 0, 1], [1, 1, 1]], mode="constant")

    identifyneighborshorizontal = scipy.ndimage.convolve(slicedmask, [[8, 0, 2]], mode="constant")
    identifyneighborsvertical = scipy.ndimage.convolve(slicedmask, [[1], [0], [4]], mode="constant")
    identifyneighborssides = identifyneighborshorizontal | identifyneighborsvertical
    identifyneighborscorners = scipy.ndimage.convolve(slicedmask, [[9, 0, 3], [0, 0, 0], [12, 0, 6]], mode="constant")
    #endpoint that looks like this
    # x
    # xx
    #  xxxxx
    hastwoadjacentneighbors = (identifyneighborssides & identifyneighborscorners).astype(bool) & (nneighbors == 2)
    #endpoint that looks like this
    # xx
    # xxxxxxxxx
    #this actually happens in L1_4 23 1 194
    hasthreeadjacentneighborsbox = (identifyneighborshorizontal & identifyneighborscorners).astype(bool) & (identifyneighborsvertical & identifyneighborscorners).astype(bool) & (nneighbors==3)

    if not np.any(slicedmask & ((nneighbors <= 1) | hastwoadjacentneighbors | hasthreeadjacentneighborsbox)):
      return

    #find the separate pieces of membrane
    labeled, nlabels = scipy.ndimage.label(slicedmask, structure=np.ones(shape=(3, 3)))

    labels = range(1, nlabels+1)

    labelendpoints = {label: list(np.argwhere((labeled==label) & ((nneighbors == 1) | hastwoadjacentneighbors | hasthreeadjacentneighborsbox))) for label in labels}
    labelsinglepixels = {label: list(np.argwhere((labeled==label) & (nneighbors == 0))) for label in labels}

    for label in labels:
      if labelsinglepixels[label]:
        labelendpoints[label] += labelsinglepixels[label]*2
  
    possibleendpointorder = {label: itertools.permutations(labelendpoints[label], 2) for label in labels}

    possiblepointstoconnect = []
    for labelordering in itertools.permutations(labels):
      if labelordering[-1] > labelordering[0]: continue #[1, 2, 3] is the same as [3, 2, 1]
      for endpointorder in dict_product(possibleendpointorder):
        pointstoconnect = []
        possiblepointstoconnect.append(pointstoconnect)
        for label1, label2 in more_itertools.pairwise(labelordering+(labelordering[0],)):
          endpoint1 = endpointorder[label1][1]
          endpoint2 = endpointorder[label2][0]
          pointstoconnect.append((endpoint1, endpoint2))
  
    def totaldistance(pointstoconnect):
      return sum(
        np.sum((point1-point2)**2)**.5
        for point1, point2 in pointstoconnect
      )
  
    while possiblepointstoconnect:
      distances = np.array([totaldistance(pointstoconnect) for pointstoconnect in possiblepointstoconnect])
      bestidx = np.argmin(distances)
      bestpointstoconnect = possiblepointstoconnect[bestidx]
      lines = np.zeros_like(slicedmask)
      for point1, point2 in bestpointstoconnect:
        lines = cv2.line(lines, tuple(point1)[::-1], tuple(point2)[::-1], 1)
      intersectionsize = np.count_nonzero(lines & slicedmask)
      linepixels = np.count_nonzero(lines)
      nlines = len(bestpointstoconnect)
      if intersectionsize > nlines*3:
        self.logger.debug(f"{nlines} lines with {linepixels} pixels total, {intersectionsize} intersection with slicedmask: {self.loginfo}")
        del possiblepointstoconnect[bestidx]
        continue
      else:
        testmask = slicedmask | lines
        polygons = self.__findpolygons(cellmask=testmask)
        if self.istoothin(polygons[0]):
          self.logger.debug(f"tried connecting lines but polygon is still long and thin (area = {polygons[0].area}, perimeter = {polygons[0].perimeter}), will try other endpoints: {self.loginfo}")
          del possiblepointstoconnect[bestidx]
          continue
        else:
          self.logger.warning(f"Broken membrane: connecting {len(labels)} components, total length of broken line segments is {totaldistance(pointstoconnect)} pixels: {self.loginfo}")

          verticaledge, horizontaledge = self.isonedge
          if verticaledge == -1:
            testmask[0, :] |= lines[1, :] & ~(slicedmask[1, :])
          if verticaledge == 1:
            testmask[-1, :] |= lines[-2, :] & ~(slicedmask[-2, :])
          if horizontaledge == -1:
            testmask[:, 0] |= lines[:, 1] & ~(slicedmask[:, 1])
          if horizontaledge == 1:
            testmask[:, -1] |= lines[:, -2] & ~(slicedmask[:, -2])

          self.slicedmask[:] = testmask
          break

  def __connectdisjointregions(self, slicedmask):
    labeled, nlabels = scipy.ndimage.label(slicedmask, structure=np.ones(shape=(3, 3)))
    labels = range(1, nlabels+1)
    if nlabels == 1:
      return slicedmask, 0, 1

    best = None
    bestnfilled = float("inf")

    for label in labels:
      """
      following the algorithm here:
      https://blogs.mathworks.com/steve/2011/11/01/exploring-shortest-paths-part-1/
      """
      thisregion = labeled == label
      otherregions = (labeled != 0) & (labeled != label)
      distance1 = scipy.ndimage.distance_transform_edt(~thisregion)
      distance2 = scipy.ndimage.distance_transform_edt(~otherregions)

      maxmultiply = 4
      for multiplytoround in range(1, maxmultiply+1):
        totaldistance = np.round((distance1+distance2)*multiplytoround, 0)

        path = np.where(totaldistance == np.min(totaldistance[~slicedmask]), True, False)
        thinnedpath = skimage.morphology.thin(path)

        partiallyconnected = slicedmask | thinnedpath

        labeled2, nlabels2 = scipy.ndimage.label(partiallyconnected, structure=np.ones(shape=(3, 3)))
        if nlabels2 < nlabels:
          break
        else:
          if self._debugdrawonerror:
            plt.imshow(slicedmask)
            plt.show()
            print(nlabels)
            plt.imshow(thisregion)
            plt.show()
            plt.imshow(distance1)
            plt.show()
            plt.imshow(otherregions)
            plt.show()
            plt.imshow(distance2)
            plt.show()
            plt.imshow(totaldistance)
            plt.show()
            plt.imshow(path)
            plt.show()
            plt.imshow(thinnedpath)
            plt.show()
            plt.imshow(partiallyconnected)
            plt.show()
            print(nlabels2)
          if multiplytoround == maxmultiply:
            raise ValueError(f"Connecting regions didn't reduce the number of regions (?) {self.loginfo}")

      fullyconnected, nfilled, _ = self.__connectdisjointregions(partiallyconnected)
      nfilled += np.count_nonzero(thinnedpath)

      if nfilled < bestnfilled:
        bestnfilled = nfilled
        best = fullyconnected

    return best, bestnfilled, nlabels

  def connectdisjointregions(self):
    slicedmask = self.slicedmask
    connected, nfilled, nlabels = self.__connectdisjointregions(slicedmask)
    if nfilled:
      self.logger.warningglobal(f"Broken cell: connecting {nlabels} disjoint regions by filling {nfilled} pixels: {self.loginfo}")
      slicedmask[:] = connected

  def pickbiggestregion(self):
    slicedmask = self.slicedmask
    labeled, nlabels = scipy.ndimage.label(slicedmask, structure=np.ones(shape=(3, 3)))
    labels = range(1, nlabels+1)
    if nlabels == 1:
      return slicedmask
    nlabeled = {label: np.sum(labeled == label) for label in labels}
    biggest = max(nlabeled.items(), key=lambda kv: kv[1])[0]
    slicedmask[labeled != biggest] = 0

  def debugdraw(self, polygon):
    if not self._debugdraw: return
    plt.imshow(self.originalcellmask.astype(np.uint8) + self.cellmask)
    ax = plt.gca()
    if polygon is not None: ax.add_patch(polygon.matplotlibpolygon(alpha=0.7, shiftby=-self.pxvec))
    top, left, bottom, right = self.adjustedbbox
    plt.xlim(left=left-1, right=right)
    plt.ylim(top=top-1, bottom=bottom)
    plt.show()
    self.logger.debug(f"{polygon}: {self.loginfo}")
