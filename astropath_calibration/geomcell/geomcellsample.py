import cv2, itertools, matplotlib.pyplot as plt, more_itertools, numpy as np, scipy.ndimage, skimage.measure
from ..alignment.field import Field
from ..baseclasses.polygon import DataClassWithPolygon, polygonfield
from ..baseclasses.rectangle import RectangleReadComponentTiffMultiLayer, GeomLoadRectangle
from ..baseclasses.sample import DbloadSample, GeomSampleBase, ReadRectanglesComponentTiff
from ..geom.contours import findcontoursaspolygons
from ..utilities import units
from ..utilities.misc import dict_product, dummylogger
from ..utilities.tableio import writetable
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
            polygons = []
            try:
              if self.ismembrane(imlayernumber):
                thiscell = joinbrokenmembrane(thiscell, logger=self.logger, loginfo=f"{field.n} {celltype} {celllabel}")
              polygons = findcontoursaspolygons(thiscell, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE, pscale=self.pscale, apscale=self.apscale, shiftby=units.nominal_values(field.pxvec), fill=True)

              if len(polygons) > 1:
                polygons.sort(key=lambda x: x.area, reverse=True)

            finally:
              if (field.n, celltype, celllabel) in _debugdraw:
                debugdraw(img=thiscell, polygons=polygons, field=field, logger=self.logger, bbox=cellproperties.bbox)

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

            if len(polygons) == 1:
              if self.ismembrane(imlayernumber):
                polygon = polygons[0]
                area = polygon.area
                perimeter = polygon.perimeter
                if area / perimeter <= 1 * self.onepixel:
                  self.logger.warningglobal(f"Long, thin polygon (perimeter = {perimeter / self.onepixel} pixels, area = {area / self.onepixel**2} pixels^2) - possibly a broken membrane that couldn't be fixed? {field.n} {celltype} {celllabel}")
            else:
              for polygon in polygons[1:]:
                area = polygon.area
                perimeter = polygon.perimeter
                message = f"Extra disjoint polygon with an area of {area/self.onepixel**2} pixels^2 and a perimeter of {perimeter / polygon.onepixel} pixels: {field.n} {celltype} {celllabel}"
                if area <= 10*self.onepixel**2:
                  self.logger.warning(message)
                else:
                  raise ValueError(message)

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

def joinbrokenmembrane(membranemask, *, logger=dummylogger, loginfo=""):
  #find the endpoints: pixels of membrane that have exactly one membrane neighbor
  nneighbors = scipy.ndimage.convolve(membranemask, [[1, 1, 1], [1, 0, 1], [1, 1, 1]], mode="constant")
  if not np.any(membranemask & (nneighbors <= 1)):
    return membranemask

  nneighborsnodiag = scipy.ndimage.convolve(membranemask, [[0, 1, 0], [1, 0, 1], [0, 1, 0]], mode="constant")

  #find the separate pieces of membrane
  labeled, nlabels = scipy.ndimage.label(membranemask, structure=np.ones(shape=(3, 3)))

  labels = range(1, nlabels+1)

  labelendpoints = {label: list(np.argwhere((labeled==label) & (nneighbors == 1))) for label in labels}
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
    bestpointstoconnect = min(
      possiblepointstoconnect,
      key=totaldistance,
    )
    lines = np.zeros_like(membranemask)
    for point1, point2 in bestpointstoconnect:
      lines = cv2.line(lines, tuple(point1)[::-1], tuple(point2)[::-1], 1)
    intersectionsize = np.count_nonzero(lines & membranemask)
    linepixels = np.count_nonzero(lines)
    nlines = len(bestpointstoconnect)
    if intersectionsize > nlines*3:
      logger.debug(f"{nlines} lines with {linepixels} pixels total, {intersectionsize} intersection with membranemask")
      possiblepointstoconnect.remove(bestpointstoconnect)
      continue
    else:
      logger.warning(f"Broken membrane: connecting {len(labels)} components, total length of broken line segments is {totaldistance(pointstoconnect)} pixels: {loginfo}")
      membranemask = membranemask | lines
      break

  return membranemask

def debugdraw(img, polygons, field, bbox, logger=dummylogger):
  plt.imshow(img)
  ax = plt.gca()
  for i, polygon in enumerate(polygons):
    ax.add_patch(polygon.matplotlibpolygon(color=f"C{i}", alpha=0.7, shiftby=-units.nominal_values(field.pxvec)))
  top, left, bottom, right = bbox
  plt.xlim(left=left-1, right=right)
  plt.ylim(top=top-1, bottom=bottom)
  plt.show()
  logger.debug(f"{polygons}")
