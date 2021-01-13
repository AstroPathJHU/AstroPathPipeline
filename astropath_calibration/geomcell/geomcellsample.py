import cv2, dataclasses, matplotlib.pyplot as plt, numpy as np, skimage.measure
from ..alignment.field import Field
from ..baseclasses.csvclasses import Polygon
from ..baseclasses.rectangle import RectangleReadComponentTiffMultiLayer, GeomLoadRectangle
from ..baseclasses.sample import DbloadSample, GeomSampleBase, ReadRectanglesComponentTiff
from ..geom.contours import findcontoursaspolygons
from ..utilities import units
from ..utilities.tableio import writetable
from ..utilities.units.dataclasses import DataClassWithApscale, DataClassWithPscale, distancefield

class FieldReadComponentTiffMultiLayer(Field, RectangleReadComponentTiffMultiLayer, GeomLoadRectangle):
  pass

class GeomCellSample(GeomSampleBase, ReadRectanglesComponentTiff, DbloadSample):
  def __init__(self, *args, **kwargs):
    super().__init__(
      *args,
      layers=[
        13,  #membrane tumor
        12,  #membrane immune
        11,  #nucleus tumor
        10,  #nucleus immune
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

  def rungeomcell(self, *, _debugdraw={}):
    self.geomfolder.mkdir(exist_ok=True, parents=True)
    nfields = len(self.rectangles)
    for i, field in enumerate(self.rectangles, start=1):
      self.logger.info(f"writing cells for field {field.n} ({i} / {nfields})")
      geomload = []
      with field.using_image() as im:
        im = im.astype(np.uint32)
        for celltype, imlayer in enumerate(im):
          properties = skimage.measure.regionprops(imlayer)
          for cellproperties in properties:
            if not np.any(cellproperties.image):
              assert False
              continue
            celllabel = cellproperties.label
            thiscell = imlayer==celllabel
            polygons = findcontoursaspolygons(thiscell.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE, pscale=self.pscale, apscale=self.apscale, shiftby=units.nominal_values(field.pxvec), fill=True)
            if (field.n, celltype, celllabel) in _debugdraw:
              kwargs = _debugdraw[field.n, celltype, celllabel]
              plt.imshow(thiscell)
              plt.xlim(**kwargs.pop("xlim", {}))
              plt.ylim(**kwargs.pop("ylim", {}))
              plt.show()
              print(polygons)
              assert not kwargs, kwargs
            if len(polygons) > 1:
              self.logger.warn(f"Multiple polygons: {field.n} {celltype} {celllabel}")
              polygons.sort(key=lambda x: len(x.vertices), reverse=True)
            polygon = sum(polygons)

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

@Polygon.dataclasswithpolygon(dc_init=True)
class CellGeomLoad(DataClassWithPscale, DataClassWithApscale):
  field: int
  ctype: int
  n: int
  x: units.Distance = distancefield(pixelsormicrons="pixels")
  y: units.Distance = distancefield(pixelsormicrons="pixels")
  w: units.Distance = distancefield(pixelsormicrons="pixels")
  h: units.Distance = distancefield(pixelsormicrons="pixels")
  poly: Polygon = Polygon.field()
  readingfromfile: dataclasses.InitVar[bool] = False

  def __init__(self, *args, box=None, **kwargs):
    boxkwargs = {}
    if box is not None:
      boxkwargs["x"], boxkwargs["y"] = box[0]
      boxkwargs["w"], boxkwargs["h"] = box[1] - box[0]
    self.__dc_init__(
      *args,
      **kwargs,
      **boxkwargs,
    )
