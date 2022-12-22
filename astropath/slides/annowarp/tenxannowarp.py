import cv2, matplotlib.patches, matplotlib.pyplot as plt, numpy as np, uncertainties as unc
from ...shared.tenx import TenXSampleBase
from ...utilities import units
from ...utilities.miscmath import covariance_matrix
from ...utilities.tableio import writetable
from ...utilities.units.dataclasses import DataClassWithPscale, distancefield
from ..align.overlap import AlignmentComparison

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
    xc = float(spot.imageX / self.onepixel)
    yc = float(spot.imageY / self.onepixel)
    dia = float(spot.dia / spot.onepixel)
    circles = cv2.HoughCircles(
      gray,cv2.HOUGH_GRADIENT,1,5,
      param1=50,param2=50,minRadius=int(dia*.9//2),maxRadius=int(dia*1.1//2)
    )
    if circles is None:
      circles = np.zeros(shape=(0, 3))
    else:
      circles, = circles
    circles[:, 0:2] += [x1, y1]

    if draw:
      fig, ax = plt.subplots()
      plt.imshow(
        wsi,
        #smooth,
        extent=(x1, x2, y2, y1),
      )
      patchkwargs = {
        #"alpha": 0.3,
        "fill": False,
        "linewidth": 2,
      }
      for circle in circles:
        ax.add_patch(matplotlib.patches.Circle((circle[0], circle[1]), circle[2], color='b', **patchkwargs))
      ax.add_patch(matplotlib.patches.Circle((xc, yc), dia/2, color='r', **patchkwargs))
      plt.show()

    return circles * self.onepixel

  def alignspots(self, *, write_result=True):
    commonalignmentresultkwargs = dict(
      pscale=self.pscale,
    )
    spots = self.spots["fiducial"]
    nspots = len(spots)
    results = []
    for i, spot in enumerate(spots, start=1):
      if i % 50: continue
      self.logger.debug("aligning fiducial spot %d / %d", i, nspots)
      nominal = np.array([spot.imageX, spot.imageY])
      circles = self.findcircle(spot, draw=True)
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
