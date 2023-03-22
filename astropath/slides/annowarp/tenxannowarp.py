import cv2, itertools, matplotlib.patches, matplotlib.pyplot as plt, methodtools, numpy as np, skimage.morphology, uncertainties as unc
from ...shared.tenx import TenXSampleBase
from ...utilities import units
from ...utilities.miscmath import covariance_matrix
from ...utilities.tableio import writetable
from ...utilities.units.dataclasses import DataClassWithPscale, distancefield

class TenXAnnoWarp(TenXSampleBase):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
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

  def alignspots(self, *, write_result=True, draw=False):
    commonalignmentresultkwargs = dict(
      pscale=self.pscale,
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
        n=i,
        x=nominal[0],
        y=nominal[1],
        row=spot.row,
        col=spot.col,
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
    results = self.__alignmentresults = self.readtable(filename, TenXAnnoWarpAlignmentResult)
    return results

class TenXAnnoWarpAlignmentResult(DataClassWithPscale):
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
  row: int
  col: int
  dx: units.Distance = distancefield(pixelsormicrons="pixels", secondfunction="{:.6g}".format)
  dy: units.Distance = distancefield(pixelsormicrons="pixels", secondfunction="{:.6g}".format)
  covxx: units.Distance = distancefield(pixelsormicrons="pixels", power=2, secondfunction="{:.6g}".format)
  covxy: units.Distance = distancefield(pixelsormicrons="pixels", power=2, secondfunction="{:.6g}".format)
  covyy: units.Distance = distancefield(pixelsormicrons="pixels", power=2, secondfunction="{:.6g}".format)
  exit: int

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
