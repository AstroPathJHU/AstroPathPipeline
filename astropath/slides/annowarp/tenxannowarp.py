import abc, cv2, hashlib, itertools, matplotlib.patches, matplotlib.pyplot as plt, methodtools, numpy as np, skimage.morphology, uncertainties as unc
from ...shared.csvclasses import Annotation, AnnotationInfo, Region
from ...shared.polygon import SimplePolygon
from ...shared.tenx import Spot, TenXSampleBase
from ...utilities import units
from ...utilities.miscmath import covariance_matrix
from ...utilities.tableio import writetable
from ...utilities.units.dataclasses import DataClassWithPscale, distancefield
from .annowarpsample import WarpedVertex
from .stitch import AnnoWarpStitchResultDefaultModel

class TenXAnnoWarp(TenXSampleBase):
  def __init__(self, *args, breaksx, breaksy, **kwargs):
    super().__init__(*args, **kwargs)
    self.breaksx = np.array(breaksx)
    self.breaksy = np.array(breaksy)
    self.logger.warningonenter("This is a work in progress, doesn't actually work yet")

  @property
  def logmodule(self):
    return "tenxannowarp"

  def circle_subplot(self, spot):
    xc = float(spot.imageX / self.onepixel)
    yc = float(spot.imageY / self.onepixel)
    dia = float(spot.dia / self.onepixel)
    x1 = int(xc - dia)
    x2 = int(xc + dia)
    y1 = int(yc - dia)
    y2 = int(yc + dia)
    with self.using_wsi() as wsi:
      return wsi[y1:y2, x1:x2], (x1, y1, x2, y2)

  def findcircle(self, spot, draw=False, returnall=False):
    wsi, (x1, y1, x2, y2) = self.circle_subplot(spot)
    pcacomponent = np.array([0.43822899, -0.0274227, -0.89844496])
    projected = np.tensordot(wsi, pcacomponent, axes=(2, 0))
    thresh = -80
    thresholded = np.where(projected>thresh, 0, 255).astype(np.uint8)
    thresholded = skimage.morphology.closing(thresholded, footprint=np.ones((5, 5)))
    thresholded = skimage.morphology.opening(thresholded, footprint=np.ones((10, 10)))
    projected -= projected.min()
    projected *= 255/projected.max()
    projected = projected.astype(np.uint8)
    blur = cv2.GaussianBlur(projected, (31,31), 1)
    xc = float(spot.imageX / self.onepixel)
    yc = float(spot.imageY / self.onepixel)
    dia = float(spot.dia / spot.onepixel)
    houghcircles = cv2.HoughCircles(
      blur,cv2.HOUGH_GRADIENT_ALT,dp=1,minDist=50,
      param1=300,param2=0.9,minRadius=int(dia*.8/2),maxRadius=int(dia*1.1/2)
    )

    if houghcircles is None:
      houghcircles = np.zeros(shape=(0, 3))
    else:
      houghcircles, = houghcircles

    circles = []
    allcircles = []

    eroded = skimage.morphology.binary_erosion(thresholded, np.ones((10, 10)))
    for x, y, r in houghcircles:
      if not np.all(np.abs([x, y] - np.array(thresholded.shape)/2) < 100):
        continue
      allangles = np.linspace(-np.pi, np.pi, 1001)
      goodindices = np.zeros_like(allangles, dtype=int)
      for i, angle in enumerate(allangles):
        coordinate = (np.array([y, x]) + r*np.array([np.sin(angle), np.cos(angle)])).astype(int)
        try:
          goodindices[i] = not eroded[tuple(coordinate)]
        except IndexError:
          goodindices[i] = False

      circle = FittedCircle(x=(x+x1)*self.onepixel, y=(y+y1)*self.onepixel, r=r*self.onepixel, angles=allangles, goodindices=goodindices, pscale=self.pscale)
      allcircles.append(circle)
      if circle.isgood:
        circles.append(circle)

    if any(c.xcut is None and c.ycut is None for c in circles):
      try:
        circle, = [c for c in circles if c.xcut is None and c.ycut is None]
      except ValueError:
        circles = []
      else:
        circles = [circle]
    else:
      pairs = []
      for c1, c2 in itertools.combinations(circles, 2):
        if c1.xcut is None and c2.ycut is None: continue
        if c1.ycut is None and c2.xcut is None: continue

        cut1, direction1 = c1.xcut if c1.xcut is not None else c1.ycut
        cut2, direction2 = c2.xcut if c2.xcut is not None else c2.ycut

        if direction1 == direction2: continue
        if not np.isclose(cut1, cut2, atol=75*self.onepixel): continue

        pairs.append([c1, c2])

      if len(pairs) == 1:
        circles, = pairs
      elif not pairs:
        circles = [c for c in circles if c.fractionalcoverage > 0.8]
        if len(circles) > 1: circles = []
      else:
        raise ValueError(pairs)

    if draw:
      fig, ax = plt.subplots()
      plt.imshow(
        #wsi,
        #thresholded,
        #eroded,
        blur,
        extent=(x1, x2, y2, y1),
      )
      patchkwargs = {
        #"alpha": 0.3,
        "fill": False,
        "linewidth": 2,
      }
      for circle in allcircles:
        ax.add_patch(circle.patch(color='b' if circle in circles else 'g', **patchkwargs))
      ax.add_patch(matplotlib.patches.Circle((xc, yc), dia/2, color='r', **patchkwargs))
      plt.show()

    if returnall:
      return allcircles

    return circles

  @methodtools.lru_cache()
  @property
  def __bigtilesizeoffset(self):
    xdiffs = self.breaksx[1:] - self.breaksx[:-1]
    ydiffs = self.breaksy[1:] - self.breaksy[:-1]

    xmean = np.mean(xdiffs)
    xstd = np.std(xdiffs)
    ymean = np.mean(ydiffs)
    ystd = np.std(ydiffs)

    if xstd / xmean > 0.02: raise ValueError(f"xdiffs {xdiffs} are inconsistent (mean {xmean} std {xstd})")
    if ystd / ymean > 0.02: raise ValueError(f"ydiffs {ydiffs} are inconsistent (mean {ymean} std {ystd})")

    x0 = np.mean([xi - i*xmean for i, xi in enumerate(self.breaksx)])
    y0 = np.mean([yi - i*ymean for i, yi in enumerate(self.breaksy)])

    return np.array([xmean, ymean]), np.array([x0, y0])

  @property
  def bigtilesize(self):
    return self.__bigtilesizeoffset[0]
  @property
  def bigtileoffset(self):
    return self.__bigtilesizeoffset[1]

  def alignspots(self, *, write_result=True, draw=False):
    commonalignmentresultkwargs = dict(
      pscale=self.pscale,
      bigtilesize=self.bigtilesize,
      bigtileoffset=self.bigtileoffset,
    )
    spots = self.spots["fiducial"]
    nspots = len(spots)
    results = []
    for i, spot in enumerate(spots, start=1):
      self.logger.debug("aligning fiducial spot %d / %d", i, nspots)
      nominal = np.array([spot.imageX, spot.imageY])
      circles = self.findcircle(spot, draw=draw)
      alignmentresultkwargs = dict(
        **commonalignmentresultkwargs,
        n=spot.n,
        x=nominal[0],
        y=nominal[1],
        row=spot.row,
        col=spot.col,
        r=spot.r,
      )
      if len(circles) != 1:
        results.append(
          TenXAnnoWarpAlignmentResult(
            **alignmentresultkwargs,
            dxvec=[unc.ufloat(0, 9999.)*self.onepixel]*2,
            exit=1 if not circles else 2,
          )
        )
      else:
        circle, = circles
        center = circle.center
        cov = np.identity(2)
        fitted = units.correlated_distances(distances=center, covariance=cov)
        dxvec = fitted - nominal

        results.append(
          TenXAnnoWarpAlignmentResult(
            **alignmentresultkwargs,
            dxvec=dxvec,
            exit=0,
          )
        )

    self.__alignmentresults = [result for result in results if result]
    if write_result:
      self.writealignments()
    return results

  @property
  def alignmentcsv(self):
    self.dbloadfolder.mkdir(exist_ok=True, parents=True)
    return self.dbloadfolder/"align_fiducal_spots.csv"
  @property
  def annotationinfocsv(self):
    self.dbloadfolder.mkdir(exist_ok=True, parents=True)
    return self.dbloadfolder/"annotationinfo.csv"
  @property
  def annotationscsv(self):
    self.dbloadfolder.mkdir(exist_ok=True, parents=True)
    return self.dbloadfolder/"annotations.csv"
  @property
  def verticescsv(self):
    self.dbloadfolder.mkdir(exist_ok=True, parents=True)
    return self.dbloadfolder/"vertices.csv"
  @property
  def regionscsv(self):
    self.dbloadfolder.mkdir(exist_ok=True, parents=True)
    return self.dbloadfolder/"regions.csv"

  def writealignments(self, *, filename=None):
    """
    write the alignments to a csv file
    """
    if filename is None: filename = self.alignmentcsv
    alignmentresults = [result for result in self.__alignmentresults if result]
    writetable(filename, alignmentresults, logger=self.logger)

  def readalignments(self, *, filename=None):
    """
    read the alignments from a csv file
    """
    if filename is None: filename = self.alignmentcsv
    results = self.__alignmentresults = self.readtable(filename, TenXAnnoWarpAlignmentResult, extrakwargs={"bigtilesize": self.bigtilesize, "bigtileoffset": self.bigtileoffset})
    return results

  @property
  def alignmentresults(self):
    return self.__alignmentresults
  @property
  def goodresults(self):
    return [_ for _ in self.alignmentresults if _]

  def stitch(self):
    A, b, c = AnnoWarpStitchResultDefaultModel.Abc(self.goodresults, None, None, self.logger)
    result = units.np.linalg.solve(2*A, -b)
    delta2nllfor1sigma = 1
    covariancematrix = units.np.linalg.inv(A) * delta2nllfor1sigma
    result = np.array(units.correlated_distances(distances=result, covariance=covariancematrix))
    self.__stitchresult = stitchresult = AnnoWarpStitchResultDefaultModel(result, A=A, b=b, c=c, constraintmus=None, constraintsigmas=None, pscale=1, apscale=1)
    return stitchresult

  def fittedspots(self, spottype):
    return [TenXFittedSpot(spot=spot, stitchresult=self.__stitchresult, bigtilesize=self.bigtilesize, bigtileoffset=self.bigtileoffset) for spot in self.spots[spottype]]

  def writeannotations(self, *, nangles=128):
    annotations = []
    allvertices = []
    regions = []
    annotationinfos = []
    with open(self.spotsfile, "rb") as f:
      hash = hashlib.sha256()
      hash.update(f.read())
      xmlsha = hash.hexdigest()

    for layer, (name, color, visible) in enumerate((
      ("FiducialSpotNominal", "0x307846", False),
      ("FiducialSpotAligned", "0x307178", False),
      ("FiducialSpotFitted", "0x303578", True),
      ("OligoSpotNominal", "0x733078", False),
      ("OligoSpotFitted", "0xA62FAF", True),
    ), start=1):
      annotationinfo = AnnotationInfo(
        sampleid=0,
        originalname=name,
        dbname=name,
        annotationsource="wsi",
        position=(0, 0),
        pscale=self.pscale,
        apscale=self.pscale,
        annoscale=self.pscale,
        xmlfile=self.spotsfile,
        xmlsha=xmlsha,
        scanfolder=None,
      )
      annotationinfos.append(annotationinfo)
      annotation = Annotation(
        sampleid=0,
        layer=layer,
        name=name,
        color=color,
        visible=visible,
        poly=None,
        apscale=self.pscale,
        pscale=self.pscale,
        annoscale=self.pscale,
        annotationinfo=annotationinfo,
      )
      annotations.append(annotation)

      spottype, aligntype = name.lower().split("spot")
      if aligntype == "nominal":
        spots = self.spots[spottype]
        xc = lambda spot: spot.imageX
        yc = lambda spot: spot.imageY
      elif aligntype == "aligned":
        assert spottype == "fiducial"
        spots = self.alignmentresults
        xc = lambda spot: spot.x + spot.dx
        yc = lambda spot: spot.y + spot.dy
      elif aligntype == "fitted":
        spots = self.fittedspots(spottype)
        xc = lambda spot: spot.px
        yc = lambda spot: spot.py
      else:
        assert False

      for n, spot in enumerate(spots, start=1):
        r = spot.r
        rid = spot.n
        regionid = layer*10000 + rid
        vertices = [
          WarpedVertex(
            regionid=regionid,
            x=xc(spot) + r*np.cos(phi),
            y=yc(spot) + r*np.sin(phi),
            vid=k,
            pscale=self.pscale,
            annotation=annotation,
            wx=xc(spot) + r*np.cos(phi),
            wy=yc(spot) + r*np.sin(phi),
          ) for k, phi in enumerate(np.linspace(-np.pi, np.pi, nangles+1), start=1)
        ]
        poly = SimplePolygon(vertices=vertices, pscale=spot.pscale, annoscale=spot.pscale)
        poly = poly.round()
        poly = poly.smooth_rdp(epsilon=2*self.onepixel)
        allvertices += poly.vertices
        region = Region(
          regionid=regionid,
          sampleid=0,
          layer=layer,
          rid=rid,
          isNeg=0,
          type="Polygon",
          nvert=len(vertices),
          poly=poly,
        )
        regions.append(region)
    writetable(self.annotationinfocsv, annotationinfos, logger=self.logger)
    writetable(self.annotationscsv, annotations, logger=self.logger)
    writetable(self.verticescsv, allvertices, logger=self.logger)
    writetable(self.regionscsv, regions, logger=self.logger)
    return annotationinfos, annotations, allvertices, regions

  def findjumps(self, goodresults, ii, jj):
    import uncertainties.unumpy as unp, matplotlib.pyplot as plt, scipy.interpolate, peakutils

    window = 1000
    resample_density = 20 #pixels, spot separation ~ 200 pixels
    threshold = 18
    min_dist = 50
    min_n_between = 5

    x = [r.xvec[ii] for r in goodresults]
    dx = unp.nominal_values([r.dxvec - self.__stitchresult.dxvec(r, apscale=1) for r in goodresults])[:,jj]
    unique_x = np.unique(x)
    between = (unique_x[:-1] + unique_x[1:]) / 2

    x_valid = []
    diff = []
    stds = []

    for u in between:
      left_idx = (x < u) & (x > u - window)
      right_idx = (x > u) & (x < u + window)
      nleft = np.count_nonzero(left_idx)
      nright = np.count_nonzero(right_idx)
      if nleft < 5 or nright < 5: continue

      frac = .9
      left_average = frac*np.median(dx[left_idx]) + (1-frac)*np.mean(dx[left_idx])
      right_average = frac*np.median(dx[right_idx]) + (1-frac)*np.mean(dx[right_idx])
      left_std = np.std(dx[left_idx])
      right_std = np.std(dx[right_idx])
      x_valid.append(u)
      diff.append(right_average - left_average)
      stds.append((left_std**2+right_std**2)**.5)

    x_valid = np.array(x_valid)
    diff = np.array(diff)
    stds = np.array(stds)
    interpolator = scipy.interpolate.interp1d(x_valid, diff)
    std_interpolator = scipy.interpolate.interp1d(x_valid, stds)
    minx = np.min(x_valid)
    maxx = np.max(x_valid)
    newx = np.linspace(minx, maxx, int((maxx-minx) // resample_density))
    newy = interpolator(newx)
    newstd = std_interpolator(newx)

    plt.scatter(x_valid, diff)
    plt.scatter(newx, newy)
    plt.scatter(newx, newstd)


    #peaks, _ = scipy.signal.find_peaks(newy, width=10, height=10)
    #peaks2, _ = scipy.signal.find_peaks(-newy, width=10, height=10)

    maxima = peakutils.indexes(newy, min_dist=min_dist)
    minima = peakutils.indexes(-newy, min_dist=min_dist)
    maxima = maxima[newy[maxima]>0]
    minima = minima[newy[minima]<0]
    assert not (frozenset(minima) & frozenset(maxima)), (minima, maxima)

    extrema = np.concatenate([maxima, minima])
    extrema.sort()

    extrema = extrema[newstd[extrema] < abs(newy[extrema])]

    last = None
    toremove = set()
    for e in extrema:
      if last is None:
        last = e
      elif (e in minima and last in maxima) or (e in maxima and last in minima):
        last = e
      elif (e in minima and last in minima):
        if newy[e] < newy[last]:
          toremove.add(last)
          last = e
        else:
          toremove.add(e)
      elif (e in maxima and last in maxima):
        if newy[e] > newy[last]:
          toremove.add(last)
          last = e
        else:
          toremove.add(e)

    extrema = np.array([e for e in extrema if e not in toremove])
    toremove = True

    plt.scatter(newx[extrema], newy[extrema])

    while toremove:
      toremove = set()
      for i, e1 in enumerate(extrema):
        try:
          e2 = extrema[i+1]
          e3 = extrema[i+2]
        except IndexError:
          continue

        y1, y2, y3 = newy[[e1, e2, e3]]
        if abs(y2-y1) < threshold and abs(y2-y1) <= abs(y2-y3):
          toremove |= {e1, e2}
          break
        elif abs(y2-y3) < threshold:
          toremove |= {e2, e3}
          break

        try:
          e4 = extrema[i+3]
        except IndexError:
          pass
        else:
          n_between_12 = np.count_nonzero((newx[e1] < x_valid) & (x_valid < newx[e2]))
          n_between_23 = np.count_nonzero((newx[e2] < x_valid) & (x_valid < newx[e3]))
          n_between_34 = np.count_nonzero((newx[e3] < x_valid) & (x_valid < newx[e4]))
          if n_between_23 < min_n_between and n_between_23 < n_between_12 and n_between_23 < n_between_34:
            toremove |= {e2, e3}

      if not toremove:
        if abs(newy[extrema[0]]) < threshold/2:
          toremove.add(extrema[0])
        if abs(newy[extrema[-1]]) < threshold/2:
          toremove.add(extrema[-1])
      extrema = np.array([e for e in extrema if e not in toremove])

    plt.scatter(newx[extrema], newy[extrema])
    plt.show()
    plt.scatter(x, dx)
    for e in extrema:
      plt.axvline(newx[e])
    plt.show()

class BigTileCoordinateBase(abc.ABC):
  @property
  @abc.abstractmethod
  def bigtilesize(self): pass
  @property
  @abc.abstractmethod
  def bigtileoffset(self): pass
  @property
  @abc.abstractmethod
  def xvec(self): pass
  @property
  def bigtileindex(self):
    return (self.xvec - self.bigtileoffset) // self.bigtilesize
  @property
  def bigtilecorner(self):
    return self.bigtileindex * self.bigtilesize + self.bigtileoffset
  @property
  def coordinaterelativetobigtile(self):
    return self.xvec - self.bigtilecorner

class BigTileCoordinate(BigTileCoordinateBase):
  def __init__(self, x, y, bigtilesize, bigtileoffset):
    self.__bigtilesize = bigtilesize
    self.__bigtileoffset = bigtileoffset
    self.__x = x
    self.__y = y

  @property
  def xvec(self): return np.array([self.__x, self.__y])
  @property
  def bigtileoffset(self): return self.__bigtileoffset
  @property
  def bigtilesize(self): return self.__bigtilesize

class TenXAnnoWarpAlignmentResult(DataClassWithPscale, BigTileCoordinateBase):
  """
  A result from the alignment of one tile of the annowarp

  n: the numerical id of the tile, starting from 1
  x, y: the x and y positions of the tile
  dx, dy: the shift in x and y
  covxx, covxy, covyy: the covariance matrix for dx and dy
  exit: the exit code of the alignment (0=success, nonzero=failure, 255=exception)
  """
  n: int
  x: units.Distance = distancefield(pixelsormicrons="pixels")
  y: units.Distance = distancefield(pixelsormicrons="pixels")
  r: units.Distance = distancefield(pixelsormicrons="pixels")
  row: int
  col: int
  dx: units.Distance = distancefield(pixelsormicrons="pixels", secondfunction="{:.6g}".format)
  dy: units.Distance = distancefield(pixelsormicrons="pixels", secondfunction="{:.6g}".format)
  covxx: units.Distance = distancefield(pixelsormicrons="pixels", power=2, secondfunction="{:.6g}".format)
  covxy: units.Distance = distancefield(pixelsormicrons="pixels", power=2, secondfunction="{:.6g}".format)
  covyy: units.Distance = distancefield(pixelsormicrons="pixels", power=2, secondfunction="{:.6g}".format)
  exit: int

  def __post_init__(self, *args, bigtilesize, bigtileoffset, **kwargs):
    self.__bigtilesize = bigtilesize
    self.__bigtileoffset = bigtileoffset
    super().__post_init__(*args, **kwargs)

  @classmethod
  def transforminitargs(cls, *args, **kwargs):
    dxvec = kwargs.pop("dxvec", None)
    morekwargs = {}

    if dxvec is not None:
      morekwargs["dx"] = dxvec[0].n
      morekwargs["dy"] = dxvec[1].n
      covariancematrix = covariance_matrix(dxvec)
    else:
      covariancematrix = kwargs.pop("covariance", None)

    if covariancematrix is not None:
      units.np.testing.assert_allclose(covariancematrix[0, 1], covariancematrix[1, 0])
      (morekwargs["covxx"], morekwargs["covxy"]), (morekwargs["covxy"], morekwargs["covyy"]) = covariancematrix

    return super().transforminitargs(*args, **kwargs, **morekwargs)

  def __bool__(self):
    return self.exit == 0

  @property
  def xvec(self):
    return np.array([self.x, self.y])
  @property
  def dxvec(self):
    return np.array([self.dx, self.dy])
  @property
  def covariance(self):
    return np.array([[self.covxx, self.covxy], [self.covxy, self.covyy]])

  @property
  def bigtileoffset(self): return self.__bigtileoffset
  @property
  def bigtilesize(self): return self.__bigtilesize

class TenXFittedSpot(Spot):
  px: units.Distance = distancefield(pixelsormicrons="pixels")
  py: units.Distance = distancefield(pixelsormicrons="pixels")
  covxx: units.Distance = distancefield(pixelsormicrons="pixels", power=2, secondfunction="{:.6g}".format)
  covxy: units.Distance = distancefield(pixelsormicrons="pixels", power=2, secondfunction="{:.6g}".format)
  covyy: units.Distance = distancefield(pixelsormicrons="pixels", power=2, secondfunction="{:.6g}".format)

  @classmethod
  def transforminitargs(cls, *args, spot=None, stitchresult=None, bigtilesize=None, bigtileoffset=None, **kwargs):
    spotkwargs = {}
    if stitchresult is not None and bigtilesize is not None and bigtileoffset is not None:
      if spot is None: raise TypeError("Have to provide a spot if calculating pxvec from stitchresult")
      coordinate = BigTileCoordinate(
        x=spot.imageX,
        y=spot.imageY,
        bigtilesize=bigtilesize,
        bigtileoffset=bigtileoffset,
      )
      pxvec = coordinate.xvec + stitchresult.dxvec(coordinate, apscale=spot.pscale)
      px, py = units.nominal_values(pxvec)
      (covxx, covxy), (covxy, covyy) = units.covariance_matrix(pxvec)
      spotkwargs = {
        "px": px,
        "py": py,
        "covxx": covxx,
        "covxy": covxy,
        "covyy": covyy,
      }
    elif stitchresult is not None or bigtilesize is not None or bigtileoffset is not None:
      raise TypeError("Have to provide all of stitchresult, bigtilesize, bigtileoffset or none of them")
    return super().transforminitargs(*args, spot=spot, **spotkwargs, **kwargs)

class FittedCircle(DataClassWithPscale):
  x: units.Distance = distancefield(pixelsormicrons="pixels")
  y: units.Distance = distancefield(pixelsormicrons="pixels")
  r: units.Distance = distancefield(pixelsormicrons="pixels")
  angles: np.ndarray
  goodindices: np.ndarray

  @property
  def center(self):
    return np.array([self.x, self.y])

  @methodtools.lru_cache()
  @property
  def __isgood_xcut_ycut(self):
    #close up little holes and get rid of little segments
    padded = np.concatenate([self.goodindices]*3)
    padded = skimage.morphology.closing(
      padded,
      footprint=np.ones(shape=25)
    )
    padded = skimage.morphology.opening(
      padded,
      footprint=np.ones(shape=25)
    )

    if np.all(padded):
      return True, None, None
    if not np.any(padded):
      return False, None, None

    derivative = padded[len(self.goodindices):2*len(self.goodindices)] - padded[len(self.goodindices)-1:2*len(self.goodindices)-1]
    assert np.sum(derivative) == 0

    regionstarts, = np.where(derivative==1)
    regionends, = np.where(derivative==-1)
    if regionstarts[0] > regionends[0]:
      regionstarts = np.roll(regionstarts, 1)

    regionlengths = regionends - regionstarts
    if regionlengths[0] < 0: regionlengths[0] += len(self.goodindices)
    np.testing.assert_array_less(0, regionlengths)

    biggestidx = np.argmax(regionlengths)
    start, end = regionstarts[biggestidx], regionends[biggestidx]

    startangle = self.angles[start]
    endangle = self.angles[end]

    startsin = np.sin(startangle)
    startcos = np.cos(startangle)
    endsin = np.sin(endangle)
    endcos = np.cos(endangle)

    yclose = np.isclose(startsin, endsin, atol=0.3) and np.isclose(startcos, -endcos, atol=0.3)
    xclose = np.isclose(startcos, endcos, atol=0.3) and np.isclose(startsin, -endsin, atol=0.3)

    if xclose and not yclose:
      return True, (self.x + self.r*(startcos+endcos)/2, 1 if startsin > 0 else -1), None
    elif yclose and not xclose:
      return True, None, (self.y + self.r*(startsin+endsin)/2, 1 if startcos > 0 else -1)
    elif sum(regionlengths) > 0.8 * len(self.goodindices):
      return True, None, None
    else:
      return False, None, None

  @property
  def isgood(self):
    return self.__isgood_xcut_ycut[0]
  @property
  def xcut(self):
    return self.__isgood_xcut_ycut[1]
  @property
  def ycut(self):
    return self.__isgood_xcut_ycut[2]

  def patch(self, **patchkwargs):
    return matplotlib.patches.Circle((self.x/self.onepixel, self.y/self.onepixel), self.r/self.onepixel, **patchkwargs)

  @property
  def fractionalcoverage(self):
    if not self.isgood: return 0
    if self.xcut is self.ycut is None: return 1
    if self.xcut is not None is not self.ycut: assert False

    if self.xcut is not None:
      xcutangle = np.arccos((self.xcut[0] - self.x) / self.r)
      if self.xcut[1] == -1:
        return xcutangle / np.pi
      elif self.xcut[1] == 1:
        return 1 - xcutangle / np.pi
      else:
        assert False

    if self.ycut is not None:
      ycutangle = np.arcsin((self.ycut[0] - self.y) / self.r)
      if self.ycut[1] == 1:
        return 0.5 - ycutangle / np.pi
      elif self.ycut[1] == -1:
        return 0.5 + ycutangle / np.pi
      else:
        assert False

    assert False
