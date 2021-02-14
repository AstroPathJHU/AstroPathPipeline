import cv2, itertools, matplotlib.pyplot as plt, more_itertools, numpy as np, scipy.ndimage, skimage.measure
from ..alignment.field import Field
from ..baseclasses.polygon import DataClassWithPolygon, polygonfield
from ..baseclasses.rectangle import RectangleReadComponentTiffMultiLayer, GeomLoadRectangle
from ..baseclasses.sample import DbloadSample, GeomSampleBase, ReadRectanglesComponentTiff
from ..geom.contours import findcontoursaspolygons
from ..utilities import units
from ..utilities.misc import dict_product, dummylogger
from ..utilities.tableio import writetable
from ..utilities.units import ThingWithApscale, ThingWithPscale
from ..utilities.units.dataclasses import distancefield

class FieldReadComponentTiffMultiLayer(Field, RectangleReadComponentTiffMultiLayer, GeomLoadRectangle):
  pass

class GeomCellSample(GeomSampleBase, ReadRectanglesComponentTiff, DbloadSample):
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

  @property
  def rectanglecsv(self): return "fields"
  rectangletype = FieldReadComponentTiffMultiLayer
  @property
  def rectangleextrakwargs(self):
    return {
      **super().rectangleextrakwargs,
      "with_seg": True,
      "geomfolder": self.geomfolder,
    }

  @property
  def logmodule(self):
    return "geomcell"

  def rungeomcell(self, *, _debugdraw=()):
    self.geomfolder.mkdir(exist_ok=True, parents=True)
    nfields = len(self.rectangles)
    for i, field in enumerate(self.rectangles, start=1):
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
            thiscell = (imlayer==celllabel).astype(np.uint8)
            polygons = PolygonFinder(thiscell, ismembrane=self.ismembrane(imlayernumber), bbox=cellproperties.bbox, pxvec=units.nominal_values(field.pxvec), pscale=self.pscale, apscale=self.apscale, logger=self.logger, loginfo=f"{field.n} {celltype} {celllabel}", _debugdraw=(field.n, celltype, celllabel) in _debugdraw).findpolygons()

            polygon = polygons[0]

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
  def __init__(self, cellmask, *, ismembrane, bbox, pscale, apscale, pxvec, _debugdraw=False, logger=dummylogger, loginfo=""):
    self.cellmask = self.originalcellmask = cellmask
    self.ismembrane = ismembrane
    self.bbox = bbox
    self.logger = logger
    self.loginfo = loginfo
    self.__pscale = pscale
    self.__apscale = apscale
    self.pxvec = pxvec
    self._debugdraw = _debugdraw

  @property
  def pscale(self): return self.__pscale
  @property
  def apscale(self): return self.__apscale

  def findpolygons(self):
    polygons = []
    try:
      if self.ismembrane:
        self.joinbrokenmembrane()
      polygons = self.__findpolygons(cellmask=self.slicedmask)

      if self.ismembrane:
        if self.istoothin(polygons[0]):
          self.logger.warningglobal(f"Long, thin polygon (perimeter = {polygons[0].perimeter / self.onepixel} pixels, area = {polygons[0].area / self.onepixel**2} pixels^2) - possibly a broken membrane that couldn't be fixed? {self.loginfo}")
      for polygon in polygons[1:]:
        area = polygon.area
        perimeter = polygon.perimeter
        message = f"Extra disjoint polygon with an area of {area/self.onepixel**2} pixels^2 and a perimeter of {perimeter / polygon.onepixel} pixels: {self.loginfo}"
        if area <= 10*self.onepixel**2:
          self.logger.warning(message)
        else:
          raise ValueError(message)
    finally:
      self.debugdraw(polygons)

    return polygons

  def __findpolygons(self, cellmask):
    top, left, bottom, right = self.bbox
    shiftby = self.pxvec + np.array([left, top]) * self.onepixel
    polygons = findcontoursaspolygons(cellmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE, pscale=self.pscale, apscale=self.apscale, shiftby=shiftby, fill=True)
    if len(polygons) > 1:
      polygons.sort(key=lambda x: x.area, reverse=True)
    return polygons

  def istoothin(self, polygon):
    area = polygon.area
    perimeter = polygon.perimeter
    return area / perimeter <= 1 * self.onepixel

  @property
  def bboxslice(self):
    top, left, bottom, right = self.bbox
    return slice(top, bottom+1), slice(left, right+1)

  @property
  def slicedmask(self):
    return self.cellmask[self.bboxslice]

  def joinbrokenmembrane(self):
    slicedmask = self.slicedmask

    polygons = self.__findpolygons(cellmask=self.slicedmask)
    if not self.istoothin(polygons[0]): return

    #find the endpoints: pixels of membrane that have exactly one membrane neighbor
    nneighbors = scipy.ndimage.convolve(slicedmask, [[1, 1, 1], [1, 0, 1], [1, 1, 1]], mode="constant")
    if not np.any(slicedmask & (nneighbors <= 1)):
      return

    identifyneighborssides = scipy.ndimage.convolve(slicedmask, [[0, 1, 0], [8, 0, 2], [0, 4, 0]], mode="constant")
    identifyneighborscorners = scipy.ndimage.convolve(slicedmask, [[9, 0, 3], [0, 0, 0], [12, 0, 6]], mode="constant")
    hastwoadjacentneighbors = (identifyneighborssides & identifyneighborscorners).astype(bool) & (nneighbors == 2)

    #find the separate pieces of membrane
    labeled, nlabels = scipy.ndimage.label(slicedmask, structure=np.ones(shape=(3, 3)))

    labels = range(1, nlabels+1)

    labelendpoints = {label: list(np.argwhere((labeled==label) & (nneighbors == 1))) for label in labels}
    labelsinglepixels = {label: list(np.argwhere((labeled==label) & (nneighbors == 0))) for label in labels}

    for label in labels:
      if labelsinglepixels[label]:
        labelendpoints[label] += labelsinglepixels[label]*2
      if len(labelendpoints[label]) == 1:
        #try to find a place like this
        # x
        # xx
        #   xxxxxxx
        #where it's an endpoint but has 2 neighbors
        endpointcandidates = np.argwhere((labeled==label) & hastwoadjacentneighbors)
        labelendpoints[label] += list(endpointcandidates)
  
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
      touchingsize = np.count_nonzero(lines & (~slicedmask) & nneighbors.astype(bool))
      linepixels = np.count_nonzero(lines)
      nlines = len(bestpointstoconnect)
      if intersectionsize > nlines*2 or touchingsize > nlines*3:
        self.logger.debug(f"{nlines} lines with {linepixels} pixels total, {intersectionsize} intersection with slicedmask and {touchingsize} touching slicedmask: {self.loginfo}")
        del possiblepointstoconnect[bestidx]
        continue
      else:
        self.logger.warning(f"Broken membrane: connecting {len(labels)} components, total length of broken line segments is {totaldistance(pointstoconnect)} pixels: {self.loginfo}")
        testmask = slicedmask | lines
        polygons = self.__findpolygons(cellmask=testmask)
        if self.istoothin(polygons[0]):
          self.logger.debug(f"tried connecting lines but polygon is still long and thin, will try other endpoints: {self.loginfo}")
          continue
        else:
          slicedmask[:] = testmask
          break

  def debugdraw(self, polygons):
    if not self._debugdraw: return
    plt.imshow(self.cellmask)
    ax = plt.gca()
    for i, polygon in enumerate(polygons):
      ax.add_patch(polygon.matplotlibpolygon(color=f"C{i}", alpha=0.7, shiftby=-self.pxvec))
    top, left, bottom, right = self.bbox
    plt.xlim(left=left-1, right=right)
    plt.ylim(top=top-1, bottom=bottom)
    plt.show()
    self.logger.debug(f"{polygons}: {self.loginfo}")
