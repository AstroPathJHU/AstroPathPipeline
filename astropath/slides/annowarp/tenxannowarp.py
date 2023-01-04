import cv2, matplotlib.patches, matplotlib.pyplot as plt, methodtools, numpy as np, skimage.morphology, uncertainties as unc
from ...shared.tenx import TenXSampleBase
from ...utilities import units
from ...utilities.miscmath import covariance_matrix
from ...utilities.tableio import writetable
from ...utilities.units.dataclasses import DataClassWithPscale, distancefield

class TenXAnnoWarp(TenXSampleBase):
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

  def findcircle(self, spot, draw=False):
    wsi, (x1, y1, x2, y2) = self.circle_subplot(spot)
    gray = cv2.cvtColor(wsi, cv2.COLOR_BGR2GRAY)
    thresh = 130
    gray[gray>thresh] = 255
    gray[gray<=thresh] = 0
    xc = float(spot.imageX / self.onepixel)
    yc = float(spot.imageY / self.onepixel)
    dia = float(spot.dia / spot.onepixel)
    houghcircles = cv2.HoughCircles(
      gray,cv2.HOUGH_GRADIENT,dp=1,minDist=50,
      param1=100,param2=10,minRadius=int(dia*.9//2),maxRadius=int(dia*1.1//2)
    )

    if houghcircles is None:
      houghcircles = np.zeros(shape=(0, 3))
    else:
      houghcircles, = houghcircles

    circles = []

    eroded = skimage.morphology.binary_erosion(gray, np.ones((10, 10)))
    for x, y, r in houghcircles:
      if not np.all(np.abs([x, y] - np.array(gray.shape)/2) < 100):
        continue
      allangles = np.linspace(-np.pi, np.pi, 1001)
      goodindices = np.zeros_like(allangles, dtype=int)
      goodangles = []
      for i, angle in enumerate(allangles):
        coordinate = (np.array([y, x]) + r*np.array([np.sin(angle), np.cos(angle)])).astype(int)
        try:
          goodindices[i] = not eroded[tuple(coordinate)]
        except IndexError:
          goodindices[i] = False

      circle = FittedCircle(x=(x+x1)*self.onepixel, y=(y+y1)*self.onepixel, r=r*self.onepixel, angles=allangles, goodindices=goodindices, pscale=self.pscale)
      if circle.isgood:
        circles.append(circle)

    if draw:
      fig, ax = plt.subplots()
      plt.imshow(
        #wsi,
        #gray,
        eroded,
        extent=(x1, x2, y2, y1),
      )
      patchkwargs = {
        #"alpha": 0.3,
        "fill": False,
        "linewidth": 2,
      }
      for circle in circles:
        ax.add_patch(circle.patch(color='b', **patchkwargs))
      ax.add_patch(matplotlib.patches.Circle((xc, yc), dia/2, color='r', **patchkwargs))
      plt.show()

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
      )
      if circles.shape[0] <= 1:
        results.append(
          TenXAnnoWarpAlignmentResult(
            **alignmentresultkwargs,
            dxvec=[unc.ufloat(0, 9999.)*self.onepixel]*2,
            exit=2-circles.shape[0],
          )
        )
      else:
        centers = circles[:, 0:2]
        mean = np.mean(centers, axis=0)
        cov = np.cov(centers.T)
        fitted = units.correlated_distances(distances=mean, covariance=cov)
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

    if xclose and yclose:
      return False, None, None
    elif xclose:
      return True, (self.x + self.r*(startcos+endcos)/2, 1 if startsin > 0 else -1), None
    elif yclose:
      return True, None, (self.y + self.r*(startsin+endsin)/2, 1 if startcos > 0 else -1)
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
