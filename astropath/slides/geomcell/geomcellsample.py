import cv2, datetime, itertools, job_lock, matplotlib.pyplot as plt, methodtools, more_itertools, numpy as np, scipy.ndimage, skimage.measure, skimage.morphology
from ...utilities.config import CONST as UNIV_CONST
from ...shared.argumentparser import CleanupArgumentParser
from ...shared.contours import findcontoursaspolygons
from ...shared.csvclasses import constantsdict
from ...shared.logging import dummylogger
from ...shared.polygon import DataClassWithPolygon, InvalidPolygonError, Polygon, polygonfield
from ...shared.rectangle import GeomLoadRectangle, rectanglefilter
from ...shared.sample import DbloadSample, GeomSampleBase, ParallelSample, ReadRectanglesDbloadComponentTiff, WorkflowSample
from ...utilities import units
from ...utilities.misc import dict_product
from ...utilities.tableio import readtable, writetable
from ...utilities.units import ThingWithApscale, ThingWithPscale
from ...utilities.units.dataclasses import distancefield
from ..align.alignsample import AlignSample
from ..align.field import Field, FieldReadComponentTiffMultiLayer

class GeomLoadField(Field, GeomLoadRectangle):
  pass

class GeomLoadFieldReadComponentTiffMultiLayer(FieldReadComponentTiffMultiLayer, GeomLoadRectangle):
  pass

class GeomCellSample(GeomSampleBase, ReadRectanglesDbloadComponentTiff, DbloadSample, ParallelSample, WorkflowSample, CleanupArgumentParser):
  @property
  def segmentationorder(self):
    return sorted(
      self.segmentationids,
      key=lambda x: -2*(x=="Tumor")-(x=="Immune")
    )

  def celltype(self, layer):
    segid = self.segmentationidfromlayer(layer)
    membrane = self.ismembranelayer(layer)
    nucleus = self.isnucleuslayer(layer)
    assert membrane ^ nucleus
    if membrane and segid == "Tumor": return 0
    if membrane and segid == "Immune": return 1
    if nucleus and segid == "Tumor": return 2
    if nucleus and segid == "Immune": return 3
    if isinstance(segid, int) and segid >= 3:
      if membrane: nucleusmembranebit = 0
      if nucleus: nucleusmembranebit = 2
      #shift all bits besides the first to the left, and put in the nucleus/membrane bit
      return (((segid-1) & ~0b1) << 1) | ((segid-1) & 0b1) | nucleusmembranebit
    assert False, (membrane, nucleus, segid)

  def __init__(self, *args, **kwargs):
    super().__init__(
      *args,
      with_seg=True,
      layers="setlater",
      **kwargs
    )
    self.setlayers(
      layers=[
        self.segmentationmembranelayer(seg) for seg in self.segmentationorder
      ] + [
        self.segmentationnucleuslayer(seg) for seg in self.segmentationorder
      ],
    )

  multilayer = True

  @property
  def rectanglecsv(self): return "fields"
  rectangletype = GeomLoadFieldReadComponentTiffMultiLayer
  @property
  def rectangleextrakwargs(self):
    return {
      **super().rectangleextrakwargs,
      "geomfolder": self.geomfolder,
    }

  @classmethod
  def logmodule(self):
    return "geomcell"

  @classmethod
  def defaultunits(cls):
    return "fast_microns"

  def rungeomcell(self, *, minarea=None, **kwargs):
    self.geomfolder.mkdir(exist_ok=True, parents=True)
    if minarea is None: minarea = (3 * self.onemicron)**2
    kwargs.update({
      "nfields": len(self.rectangles),
      "minarea": minarea,
      "logger": self.logger,
      "layers": self.layers,
      "celltypes": [self.celltype(imlayernumber) for imlayernumber in self.layers],
      "arelayersmembrane": [self.ismembranelayer(imlayernumber) for imlayernumber in self.layers],
      "pscale": self.pscale,
      "apscale": self.apscale,
      "unitsargs": units.currentargs(),
    })
    if self.njobs is None or self.njobs > 1:
      with self.pool() as pool:
        results = [
          pool.apply_async(self.rungeomcellfield, args=(i, field), kwds=kwargs)
          for i, field in enumerate(self.rectangles, start=1)
        ]
        for r in results:
          r.get()
    else:
      for i, field in enumerate(self.rectangles, start=1):
        self.rungeomcellfield(i, field, **kwargs)

  def run(self, *, cleanup=False, **kwargs):
    if cleanup: self.cleanup()
    self.rungeomcell(**kwargs)

  @staticmethod
  def rungeomcellfield(i, field, *, _debugdraw=(), _debugdrawonerror=False, _onlydebug=False, repair=True, rerun=False, minarea, nfields, logger, layers, celltypes, arelayersmembrane, pscale, apscale, unitsargs):
    with units.setup_context(*unitsargs), job_lock.JobLock(field.geomloadcsv.with_suffix(".lock"), corruptfiletimeout=datetime.timedelta(minutes=10), outputfiles=[field.geomloadcsv], checkoutputfiles=not rerun) as lock:
      if not lock: return
      if _onlydebug and not any(fieldn == field.n for fieldn, celltype, celllabel in _debugdraw): return
      onepixel = units.onepixel(pscale)
      if not _debugdraw: _onlydebug = False
      logger.info(f"writing cells for field {field.n} ({i} / {nfields})")
      geomload = []
      pxvec = units.nominal_values(field.pxvec)
      with field.using_image() as im:
        im = im.astype(np.uint32)
        for imlayernumber, imlayer, celltype, ismembranelayer in more_itertools.zip_equal(layers, im.transpose(2, 0, 1), celltypes, arelayersmembrane):
          properties = skimage.measure.regionprops(imlayer)
          for cellproperties in properties:
            if not np.any(cellproperties.image):
              assert False
              continue
            celllabel = cellproperties.label
            if _onlydebug and (field.n, celltype, celllabel) not in _debugdraw: continue
            polygon = PolygonFinder(imlayer, celllabel, ismembrane=ismembranelayer, bbox=cellproperties.bbox, pxvec=pxvec, mxbox=field.mxbox, pscale=pscale, apscale=apscale, logger=logger, loginfo=f"{field.n} {celltype} {celllabel}", _debugdraw=(field.n, celltype, celllabel) in _debugdraw, _debugdrawonerror=_debugdrawonerror, repair=repair).findpolygon()
            if polygon is None: continue
            if polygon.area < minarea: continue

            box = np.array(cellproperties.bbox).reshape(2, 2)[:,::-1] * onepixel * 1.0
            box += pxvec
            box = box // onepixel * onepixel

            geomload.append(
              CellGeomLoad(
                field=field.n,
                ctype=celltype,
                n=celllabel,
                box=box,
                poly=polygon,
                pscale=pscale,
                apscale=apscale,
              )
            )

      writetable(field.geomloadcsv, geomload, rowclass=CellGeomLoad)

  def inputfiles(self, **kwargs):
    return super().inputfiles(**kwargs) + [
      self.csv("constants"),
      self.csv("fields"),
      *(r.imagefile for r in self.rectangles),
    ]

  @classmethod
  def getworkinprogressfiles(cls, SlideID, *, geomroot, **otherworkflowkwargs):
    geomfolder = geomroot/SlideID/"geom"
    return [
      *geomfolder.glob("*.csv"),
    ]

  @property
  def workflowkwargs(self):
    return {"selectrectangles": rectanglefilter(lambda r: r.n in {r.n for r in self.rectangles}), **super().workflowkwargs}

  @classmethod
  def getoutputfiles(cls, SlideID, *, dbloadroot, geomroot, selectrectangles=lambda r: True, **otherworkflowkwargs):
    dbload = dbloadroot/SlideID/UNIV_CONST.DBLOAD_DIR_NAME
    fieldscsv = dbload/f"{SlideID}_fields.csv"
    constantscsv = dbload/f"{SlideID}_constants.csv"
    if not fieldscsv.exists(): return [fieldscsv]
    constants = constantsdict(constantscsv)
    rectangles = readtable(fieldscsv, GeomLoadField, extrakwargs={"pscale": constants["pscale"], "geomfolder": geomroot/SlideID/"geom"})
    return [
      *(r.geomloadcsv for r in rectangles if selectrectangles(r)),
    ]

  @classmethod
  def workflowdependencyclasses(cls):
    return [AlignSample] + super().workflowdependencyclasses()

class CellGeomLoad(DataClassWithPolygon):
  field: int
  ctype: int
  n: int
  x: units.Distance = distancefield(pixelsormicrons="pixels", dtype=int)
  y: units.Distance = distancefield(pixelsormicrons="pixels", dtype=int)
  w: units.Distance = distancefield(pixelsormicrons="pixels", dtype=int)
  h: units.Distance = distancefield(pixelsormicrons="pixels", dtype=int)
  poly: Polygon = polygonfield()

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
  def __init__(self, image, celllabel, *, ismembrane, bbox, pscale, apscale, pxvec, mxbox, _debugdraw=False, _debugdrawonerror=False, repair=True, logger=dummylogger, loginfo=""):
    self.image = image
    self.celllabel = celllabel
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
    self.repair = repair and self.slicedimage.size <= 200*200

  @property
  def pscale(self): return self.__pscale
  @property
  def apscale(self): return self.__apscale

  def findpolygon(self):
    polygon = None
    try:
      if self.isprimary and self.repair:
        if self.ismembrane:
          self.joinbrokenmembrane()
        #self.connectdisjointregions()
        self.pickbiggestregion()
        self.cleanup()
      else:
        self.pickbiggestregion()
        self.cleanup()
      if not np.any(self.slicedmask):
        if self.isprimary:
          self.logger.warningglobal(f"Cleaned up cell is empty {self.loginfo}")
        return None
      polygon, = self.__findpolygons(cellmask=self.slicedmask.astype(np.uint8))

      try:
        polygons = polygon.makevalid(round=True)
      except InvalidPolygonError as e:
        if self.isprimary:
          estring = str(e).replace("\n", " ")
          self.logger.warningglobal(f"{estring} {self.loginfo}")
        return None
      else:
        if not polygons:
          return None
        if len(polygons) > 1:
          if self.isprimary:
            biggestarea = polygons[0].area
            discardedarea = sum(p.area for p in polygons[1:])
            self.logger.warningglobal(f"Multiple polygons connected by 1-dimensional lines - keeping the biggest ({biggestarea} pixels) and discarding the rest (total {discardedarea} pixels)")
        polygon = polygons[0]
    except:
      if self._debugdrawonerror: self._debugdraw = True
      raise
    finally:
      self.debugdraw(polygon)

    return polygon

  def __findpolygons(self, cellmask):
    if not np.any(cellmask): return []
    top, left, bottom, right = self.adjustedbbox
    shiftby = self.pxvec + np.array([left, top]) * self.onepixel
    polygons = findcontoursaspolygons(cellmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE, pscale=self.pscale, apscale=self.apscale, shiftby=shiftby, fill=True)
    if len(polygons) > 1:
      polygons.sort(key=lambda x: x.area, reverse=True)
    return polygons

  def istoothin(self, polygon):
    area = polygon.area
    perimeter = polygon.perimeter
    if area == perimeter == 0: return True
    return area / perimeter <= .8 * self.onepixel

  @property
  def adjustedbbox(self):
    top, left, bottom, right = self.__bbox
    if self.ismembrane:
      if top == 1: top = 0
      if left == 1: left = 0
      height, width = self.image.shape
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
    height, width = self.image.shape
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
  def slicedimage(self):
    return self.image[self.bboxslice]
  @methodtools.lru_cache()
  @property
  def slicedmask(self):
    return self.slicedimage == self.celllabel
  @property
  def originalslicedmask(self):
    return self.slicedimage == self.celllabel

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

  def __connectdisjointregions(self, slicedmask, *, shortcut=False):
    labeled, nlabels = scipy.ndimage.label(slicedmask, structure=np.ones(shape=(3, 3)))
    if nlabels == 1:
      return slicedmask, 0, 1, 1
    labels = range(1, nlabels+1)
    shortcut = shortcut or nlabels > 6

    best = slicedmask
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

        labeled2, nnewlabels = scipy.ndimage.label(partiallyconnected, structure=np.ones(shape=(3, 3)))
        if nnewlabels < nlabels:
          fullyconnected, nfilled, _, nnewlabels = self.__connectdisjointregions(partiallyconnected, shortcut=shortcut)
          nfilled += np.count_nonzero(thinnedpath)
          break
      else:
        nfilled = float("inf")
        if self._debugdraw:
          plt.imshow(slicedmask)
          plt.show()
          self.logger.info(nlabels)
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
          self.logger.info(nnewlabels)

      if nfilled < bestnfilled:
        bestnfilled = nfilled
        best = fullyconnected
        if shortcut: break

    return best, bestnfilled, nlabels, nnewlabels

  def connectdisjointregions(self):
    slicedmask = self.slicedmask
    connected, nfilled, nlabels, nnewlabels = self.__connectdisjointregions(slicedmask)
    if nfilled:
      slicedmask[:] = connected
      if nnewlabels == 1:
        self.logger.warning(f"Broken cell: connecting {nlabels} disjoint regions by filling {nfilled} pixels: {self.loginfo}")
    if nnewlabels > 1:
      self.logger.warningglobal(f"Couldn't connect all the disjoint regions with the same label, picking the biggest {self.loginfo}")
      return self.pickbiggestregion()

  def pickbiggestregion(self):
    slicedmask = self.slicedmask
    labeled, nlabels = scipy.ndimage.label(slicedmask, structure=np.ones(shape=(3, 3)))
    labels = range(1, nlabels+1)
    if nlabels == 1:
      return slicedmask
    nlabeled = {label: np.sum(labeled == label) for label in labels}
    biggest = max(nlabeled.items(), key=lambda kv: kv[1])[0]
    self.logger.warning(f"Broken cell: picking the biggest region ({nlabeled[biggest]} pixels) and discarding the others (total {sum(nlabeled.values()) - nlabeled[biggest]} pixels)")
    slicedmask[labeled != biggest] = 0

  def cleanup(self):
    slicedmask = self.slicedmask.astype(np.uint8)
    size = np.sum(slicedmask)
    lastsize = float("inf")
    while lastsize != size:
      lastsize = size
      nneighbors = scipy.ndimage.convolve(slicedmask, [[1, 1, 1], [1, 0, 1], [1, 1, 1]], mode="constant")
      oneneighbor = nneighbors == 1
      slicedmask[oneneighbor] = 0
      size = np.sum(slicedmask)
    self.slicedmask = slicedmask.astype(bool)

  def debugdraw(self, polygon):
    if not self._debugdraw: return
    im = np.zeros_like(self.image, dtype=np.uint8)
    im[self.bboxslice] += self.originalslicedmask
    im[self.bboxslice] += self.slicedmask
    plt.imshow(im)
    ax = plt.gca()
    if polygon is not None: ax.add_patch(polygon.matplotlibpolygon(alpha=0.7, shiftby=-self.pxvec))
    top, left, bottom, right = self.adjustedbbox
    plt.xlim(left=left-1, right=right)
    plt.ylim(top=top-1, bottom=bottom)
    plt.show()
    self.logger.debug(f"{polygon}: {self.loginfo}")

def main(args=None):
  GeomCellSample.runfromargumentparser(args)

if __name__ == "__main__":
  main()
