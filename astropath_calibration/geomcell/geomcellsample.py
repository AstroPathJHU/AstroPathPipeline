import cv2, matplotlib.pyplot as plt, numpy as np, skimage.measure
from ..alignment.field import Field
from ..baseclasses.polygon import DataClassWithPolygon, polygonfield
from ..baseclasses.rectangle import RectangleReadComponentTiffMultiLayer, GeomLoadRectangle
from ..baseclasses.sample import DbloadSample, GeomSampleBase, ReadRectanglesComponentTiff
from ..geom.contours import findcontoursaspolygons
from ..utilities import units
from ..utilities.tableio import writetable
from ..utilities.units.dataclasses import distancefield

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

            for i, poly in reversed(list(enumerate(polygons[:]))):
              if poly.area == 0 or poly.perimeter / poly.area > 2 / self.onepixel:
                try:
                  hull = poly.convexhull
                except RuntimeError as e:
                  self.logger.warning(f"can't get convex hull for small polygon {poly}: {e}")
                  continue
                if hull.area != 0 and hull.perimeter / hull.area <= 1 / self.onepixel:
                  polygons[i] = hull

            if len(polygons) > 1:
              polygons.sort(key=lambda x: x.area, reverse=True)

            if (field.n, celltype, celllabel) in _debugdraw:
              kwargs = _debugdraw[field.n, celltype, celllabel]
              plt.imshow(thiscell)
              ax = plt.gca()
              for i, polygon in enumerate(polygons):
                ax.add_patch(polygon.matplotlibpolygon(color=f"C{i}", alpha=0.7, shiftby=-units.nominal_values(field.pxvec)))
              plt.xlim(**kwargs.pop("xlim", {}))
              plt.ylim(**kwargs.pop("ylim", {}))
              plt.show()
              print(polygons)
              assert not kwargs, kwargs

            box = np.array(cellproperties.bbox).reshape(2, 2) * self.onepixel * 1.0
            box += units.nominal_values(field.pxvec)
            box = box // self.onepixel * self.onepixel

            geomload.append(
              CellGeomLoad(
                field=field.n,
                ctype=celltype,
                n=celllabel,
                box=box,
                poly=polygons[0],
                pscale=self.pscale,
                apscale=self.apscale,
              )
            )

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
