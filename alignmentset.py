#!/usr/bin/env python3

import collections, cv2, dataclasses, logging, methodtools, numpy as np, os, scipy, typing, uncertainties, uncertainties.unumpy as unp

from .flatfield import meanimage
from .overlap import Overlap
from .tableio import readtable, writetable

logger = logging.getLogger("align")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(message)s, %(funcName)s, %(asctime)s"))
logger.addHandler(handler)

class AlignmentSet:
  """
  Main class for aligning a set of images
  """
  def __init__(self, root1, root2, samp, *, interactive=False, selectrectangles=None, selectoverlaps=None):
    """
    Directory structure should be
    root1/
      samp/
        dbload/
          samp_*.csv
          samp_qptiff.jpg
    root2/
      samp/
        samp_*.(fw01/camWarp_layer01) (if using DAPI, could also be 02 etc. to align with other markers)

    interactive: if this is true, then the script might try to prompt
                 you for input if things go wrong
    """
    logger.info(samp)
    self.root1 = root1
    self.root2 = root2
    self.samp = samp
    self.interactive = interactive

    if selectrectangles is None:
      self.rectanglefilter = lambda r: True
    elif isinstance(selectrectangles, collections.abc.Container):
      self.rectanglefilter = lambda r: r.n in selectrectangles
    else:
      self.rectanglefilter = selectrectangles

    if selectoverlaps is None:
      overlapfilter = lambda o: True
    elif isinstance(selectoverlaps, collections.abc.Container):
      overlapfilter = lambda o: o.n in selectoverlaps
    else:
      overlapfilter = selectoverlaps
    self.overlapfilter = lambda o: overlapfilter(o) and o.p1 in self.rectangleindices() and o.p2 in self.rectangleindices()

    if not os.path.exists(os.path.join(self.root1, self.samp)):
      raise IOError(f"{os.path.join(self.root1, self.samp)} does not exist")

    self.readmetadata()
    self.rawimages=None

  @methodtools.lru_cache()
  def rectangleindices(self):
    return {r.n for r in self.rectangles}

  @property
  def dbload(self):
    return os.path.join(self.root1, self.samp, "dbload")

  def readmetadata(self):
    """
    Read metadata from csv files
    """
    def intorfloat(string):
      assert isinstance(string, str)
      try: return int(string)
      except ValueError: return float(string)
    self.annotations = readtable(os.path.join(self.dbload, self.samp+"_annotations.csv"), "Annotation", sampleid=int, layer=int, visible=int)
    self.regions     = readtable(os.path.join(self.dbload, self.samp+"_regions.csv"), "Region", regionid=int, sampleid=int, layer=int, rid=int, isNeg=int, nvert=int)
    self.vertices    = readtable(os.path.join(self.dbload, self.samp+"_vertices.csv"), "Vertex", regionid=int, vid=int, x=int, y=int)
    self.batch       = readtable(os.path.join(self.dbload, self.samp+"_batch.csv"), "Batch", SampleID=int, Scan=int, Batch=int)
    self.overlaps    = readtable(os.path.join(self.dbload, self.samp+"_overlap.csv"), self.overlaptype)
    self.imagetable  = readtable(os.path.join(self.dbload, self.samp+"_qptiff.csv"), "ImageInfo", SampleID=int, XPosition=float, YPosition=float, XResolution=float, YResolution=float, qpscale=float, img=int)
    self.__image     = None
    self.constants   = readtable(os.path.join(self.dbload, self.samp+"_constants.csv"), "Constant", value=intorfloat)
    self.rectangles  = readtable(os.path.join(self.dbload, self.samp+"_rect.csv"), Rectangle)

    self.constantsdict = {constant.name: constant.value for constant in self.constants}

    self.scan = f"Scan{self.batch[0].Scan:d}"

    self.fwidth    = self.constantsdict["fwidth"]
    self.fheight   = self.constantsdict["fheight"]
    self.pscale    = self.constantsdict["pscale"]
    self.qpscale   = self.constantsdict["qpscale"]
    self.xposition = self.constantsdict["xposition"]
    self.yposition = self.constantsdict["yposition"]
    self.nclip     = self.constantsdict["nclip"]
    self.layer     = self.constantsdict["layer"]

    self.rectangles = [r for r in self.rectangles if self.rectanglefilter(r)]
    self.overlaps = [o for o in self.overlaps if self.overlapfilter(o)]

    self.overlapsdict = {(o.p1, o.p2): o for o in self.overlaps}

  @property
  @methodtools.lru_cache()
  def image(self):
    return cv2.imread(os.path.join(self.dbload, self.samp+"_qptiff.jpg"))

  def align(self, *, compute_R_error_stat=True, compute_R_error_syst=True, compute_F_error=False, maxpairs=float("inf"), chooseoverlaps=None):
    #if the raw images haven't already been loaded, load them with the default argument
    if self.rawimages is None :
      self.getDAPI()

    aligncsv = os.path.join(self.dbload, self.samp+"_align.csv")

    logger.info("starting align loop for "+self.samp)

    alignments = []
    sum_mse = 0.
    done = set()

    for i, overlap in enumerate(self.overlaps, start=1):
      if chooseoverlaps is not None and i not in chooseoverlaps: continue
      if i > maxpairs: break
      logger.info(f"aligning overlap {i}/{len(self.overlaps)}")
      p1image = [r.image for r in self.rectangles if r.n==overlap.p1]
      p2image = [r.image for r in self.rectangles if r.n==overlap.p2]
      try :
        overlap_images = p1image[0], p2image[0]
      except IndexError :
        errormsg=f"error indexing images from rectangle.n for overlap #{i} (p1={overlap.p1}, p2={overlap.p2})"
        errormsg+=f" [len(p1image)={len(p1image)}, len(p2image)={len(p2image)}]"
        raise IndexError(errormsg)
      overlap.setalignmentinfo(layer=self.layer, pscale=self.pscale, nclip=self.nclip, images=overlap_images)
      if (overlap.p2, overlap.p1) in done:
        result = overlap.getinversealignment(self.overlapsdict[overlap.p2, overlap.p1])
      else:
        result = overlap.align(compute_R_error_stat=compute_R_error_stat, compute_R_error_syst=compute_R_error_syst, compute_F_error=compute_F_error)
      done.add((overlap.p1, overlap.p2))
      if result is not None: 
        alignments.append(result)
        sum_mse+=result.mse[2]

    writetable(aligncsv, alignments, retry=self.interactive)

    logger.info("finished align loop for "+self.samp)
    return sum_mse

  @methodtools.lru_cache(maxsize=1)
  def getDAPI(self, filetype="flatWarpDAPI", keeprawimages=False):
    logger.info(self.samp)
    rawimages = self.__getrawlayers(filetype, keep=keeprawimages)

    # apply the extra flattening

    self.meanimage = meanimage(rawimages, self.samp)

    for image in rawimages:
        image[:] = np.rint(image / self.meanimage.flatfield)
    self.images = rawimages

    for rectangle, image in zip(self.rectangles, self.images):
      rectangle.image = image

    self.imagestats = [
      ImageStats(
        n=rectangle.n,
        mean=np.mean(rectangle.image),
        min=np.min(rectangle.image),
        max=np.max(rectangle.image),
        std=np.std(rectangle.image),
        cx=rectangle.cx,
        cy=rectangle.cy,
      ) for rectangle in self.rectangles
    ]
    writetable(os.path.join(self.dbload, self.samp+"_imstat.csv"), self.imagestats, retry=self.interactive)

  def updateRectangleImages(self,imgdict,ext) :
    """
    Updates the "image" variable in each rectangle based on a dictionary of image layers
    imgdict = dictionary indexed first by filename then by layer number; values are image layers 
    ext     = string of file extension identifying image layers in imgdict (will be replaced by ".im3" to match "file" variable in rectangles)
    """
    warped_image_filenames = imgdict.keys()
    for r in self.rectangles :
      imgdictfn = r.file.replace('.im3',ext)
      if imgdictfn in warped_image_filenames :
        r.image = imgdict[imgdictfn][self.layer] / self.meanimage.flatfield

  def __getrawlayers(self, filetype, keep=False):
    logger.info(self.samp)
    if filetype=="flatWarpDAPI" :
      ext = f".fw{self.layer:02d}"
    elif filetype=="camWarpDAPI" :
      ext = f".camWarp_layer{self.layer:02d}"
    else :
      raise ValueError(f"requested file type {filetype} not recognized by getrawlayers")
    path = os.path.join(self.root2, self.samp)

    rawimages = np.ndarray(shape=(len(self.rectangles), self.fheight, self.fwidth), dtype=np.uint16)

    if not self.rectangles:
      raise IOError("didn't find any rows in the rectangles table for "+self.samp, 1)

    for i, rectangle in enumerate(self.rectangles):
      #logger.info(f"loading rectangle {i+1}/{len(self.rectangles)}")
      with open(os.path.join(path, rectangle.file.replace(".im3", ext)), "rb") as f:
        #use fortran order, like matlab!
        rawimages[i] = np.memmap(
          f,
          dtype=np.uint16,
          shape=(self.fheight, self.fwidth),
          order="F",
          mode="r"
        )

    if keep:
        self.rawimages = rawimages.copy()
        for rectangle, rawimage in zip(rectangles, self.rawimages):
            rectangle.rawimage = rawimage

    return rawimages

  overlaptype = Overlap #can be overridden in subclasses

  @property
  def overlapgraph(self):
    try:
      import networkx as nx
    except ImportError:
      raise ImportError("To get the overlap graph you have to install networkx")

    g = nx.DiGraph()
    for o in self.overlaps:
      g.add_edge(o.p1, o.p2, overlap=o)

    return g

  def stitch(self, *, usecvxpy=False, saveresult=True, **kwargs):
    result = (self.__stitch_cvxpy if usecvxpy else self.__stitch)(**kwargs)

    if saveresult:
      rectangledict = {rectangle.n: i for i, rectangle in enumerate(self.rectangles)}
      for o in self.overlaps:
        o.stitchresult = result.dx(o)

    return result

  def __stitch(self, *, scaleby=1, getcovariance=True, geterror=True, scalejittererror=1, scaleoverlaperror=1):
    """
    \begin{align}
    -2 \ln L =&
      1/2 \sum_\text{overlaps}
      (\vec{x}_{p1} - \vec{x}_{p2} - d\vec{x} - \vec{x}_{p1}^n + \vec{x}_{p2}^n)^T \\
      &\mathbf{cov}^{-1}
      (\vec{x}_{p1} - \vec{x}_{p2} - d\vec{x} - \vec{x}_{p1}^n + \vec{x}_{p2}^n) \\
      +&
      \sum_p \left(
        \frac{(\vec{x}_p-\vec{x}_0) - \mathbf{T}(\vec{x}_p^n-\vec{x}_0)}
        {\sigma}
      \right)^2
    \end{align}
    \begin{equation}
    \mathbf{T} = \begin{pmatrix}
    T_{xx} & T_{xy} \\ T_{yx} & T_{yy}
    \end{pmatrix}
    \end{equation}
    """
    #nll = x^T A x + bx + c

    size = 2*len(self.rectangles) + 4 #2* because each rectangle has an x and a y, + 4 for the components of T
    A = np.zeros(shape=(size, size))
    b = np.zeros(shape=(size,))
    c = 0

    Txx = -4
    Txy = -3
    Tyx = -2
    Tyy = -1

    rectangledict = {rectangle.n: i for i, rectangle in enumerate(self.rectangles)}
    for o in self.overlaps:
      ix = 2*rectangledict[o.p1]
      iy = 2*rectangledict[o.p1]+1
      jx = 2*rectangledict[o.p2]
      jy = 2*rectangledict[o.p2]+1
      assert ix >= 0, ix
      assert iy < 2*len(self.rectangles), iy
      assert jx >= 0, jx
      assert jy < 2*len(self.rectangles), jy

      ii = np.ix_((ix,iy), (ix,iy))
      ij = np.ix_((ix,iy), (jx,jy))
      ji = np.ix_((jx,jy), (ix,iy))
      jj = np.ix_((jx,jy), (jx,jy))
      inversecovariance = np.linalg.inv(o.result.covariance) * scaleby**2 / scaleoverlaperror**2

      A[ii] += inversecovariance / 2
      A[ij] -= inversecovariance / 2
      A[ji] -= inversecovariance / 2
      A[jj] += inversecovariance / 2

      i = np.ix_((ix, iy))
      j = np.ix_((jx, jy))

      constpiece = (-o.result.dxvec - o.x1vec + o.x2vec) / scaleby

      b[i] += inversecovariance @ constpiece
      b[j] -= inversecovariance @ constpiece

      c += constpiece @ inversecovariance @ constpiece / 2

    dxs, dys = zip(*(o.result.dxdy for o in self.overlaps))

    weightedvariancedx = np.average(
      unp.nominal_values(dxs)**2,
      weights=1/unp.std_devs(dxs)**2,
    )
    sigmax = np.sqrt(weightedvariancedx) / scaleby * scalejittererror

    weightedvariancedy = np.average(
      unp.nominal_values(dys)**2,
      weights=1/unp.std_devs(dys)**2,
    )
    sigmay = np.sqrt(weightedvariancedy) / scaleby * scalejittererror

    x0vec = np.mean([r.xvec for r in self.rectangles], axis=0)
    x0, y0 = x0vec

    for r in self.rectangles:
      ix = 2*rectangledict[r.n]
      iy = 2*rectangledict[r.n]+1

      x = r.x / scaleby
      y = r.y / scaleby

      A[ix,ix] += 1 / sigmax**2
      A[iy,iy] += 1 / sigmay**2
      A[(ix, Txx), (Txx, ix)] -= (x - x0) / sigmax**2
      A[(ix, Txy), (Txy, ix)] -= (y - y0) / sigmax**2
      A[(iy, Tyx), (Tyx, iy)] -= (x - x0) / sigmay**2
      A[(iy, Tyy), (Tyy, iy)] -= (y - y0) / sigmay**2

      A[Txx, Txx]               += (x - x0)**2       / sigmax**2
      A[(Txx, Txy), (Txy, Txx)] += (x - x0)*(y - y0) / sigmax**2
      A[Txy, Txy]               += (y - y0)**2       / sigmax**2

      A[Tyx, Tyx]               += (x - x0)**2       / sigmay**2
      A[(Tyx, Tyy), (Tyy, Tyx)] += (x - x0)*(y - y0) / sigmay**2
      A[Tyy, Tyy]               += (y - y0)**2       / sigmay**2

      b[ix] -= 2 * x0 / sigmax**2
      b[iy] -= 2 * y0 / sigmay**2

      b[Txx] += 2 * x0 * (x - x0) / sigmax**2
      b[Txy] += 2 * x0 * (y - y0) / sigmax**2
      b[Tyx] += 2 * y0 * (x - x0) / sigmay**2
      b[Tyy] += 2 * y0 * (y - y0) / sigmay**2

      c += x0**2 / sigmax**2
      c += y0**2 / sigmay**2

    result = np.linalg.solve(2*A, -b)

    if getcovariance or geterror:
      onesigmaCL = scipy.stats.chi2.cdf(1, df=1)
      solveresult = scipy.optimize.root_scalar(
        f=lambda x: scipy.stats.chi2.cdf(x, df=size) - onesigmaCL,
        fprime=lambda x: scipy.stats.chi2.pdf(x, df=size),
        x0=size,
        method="newton",
      )
      if not solveresult.converged:
        raise ValueError(f"finding -2 delta ln L for 1sigma failed with flag {solveresult.flag}")
      delta2nllfor1sigma = solveresult.root

    if getcovariance:
      covariancematrix = np.linalg.inv(A) * delta2nllfor1sigma
      result = np.array(uncertainties.correlated_values(result, covariancematrix))

    elif geterror:  #less computationally intensive, don't need to invert big matrix
      errorvector = (np.diag(A) * delta2nllfor1sigma) ** -0.5
      result = unp.uarray(result, errorvector)

    x = result[:-4].reshape(len(self.rectangles), 2) * scaleby
    T = result[-4:].reshape(2, 2)

    return StitchResult(x=x, T=T, A=A, b=b, c=c, rectangledict=rectangledict)

  def __stitch_cvxpy(self):
    """
    \begin{align}
    -2 \ln L =&
      1/2 \sum_\text{overlaps}
      (\vec{x}_{p1} - \vec{x}_{p2} - d\vec{x} - \vec{x}_{p1}^n + \vec{x}_{p2}^n)^T \\
      &\mathbf{cov}^{-1}
      (\vec{x}_{p1} - \vec{x}_{p2} - d\vec{x} - \vec{x}_{p1}^n + \vec{x}_{p2}^n) \\
      +&
      \sum_p \left(
        \frac{(\vec{x}_p-\vec{x}_0) - \mathbf{T}(\vec{x}_p^n-\vec{x}_0)}
        {\sigma}
      \right)^2
    \end{align}
    \begin{equation}
    \mathbf{T} = \begin{pmatrix}
    T_{xx} & T_{xy} \\ T_{yx} & T_{yy}
    \end{pmatrix}
    \end{equation}
    """

    try:
      import cvxpy as cp
    except ImportError:
      raise ImportError("To stitch with cvxpy, you have to install cvxpy")

    x = cp.Variable(shape=(len(self.rectangles), 2))
    T = cp.Variable(shape=(2, 2))

    twonll = 0
    rectanglex = {r.n: xx for r, xx in zip(self.rectangles, x)}

    for o in self.overlaps:
      x1 = rectanglex[o.p1]
      x2 = rectanglex[o.p2]
      twonll += 0.5 * cp.quad_form(
        x1 - x2 - o.result.dxvec - o.x1vec + o.x2vec,
        np.linalg.inv(o.result.covariance)
      )

    dxs, dys = zip(*(o.result.dxdy for o in self.overlaps))

    weightedvariancedx = np.average(
      unp.nominal_values(dxs)**2,
      weights=1/unp.std_devs(dxs)**2,
    )
    sigmax = np.sqrt(weightedvariancedx)

    weightedvariancedy = np.average(
      unp.nominal_values(dys)**2,
      weights=1/unp.std_devs(dys)**2,
    )
    sigmay = np.sqrt(weightedvariancedy)

    sigma = np.array(sigmax, sigmay)

    x0vec = np.mean([r.xvec for r in self.rectangles], axis=0)

    for r in self.rectangles:
      twonll += cp.norm(((rectanglex[r.n] - x0vec) - T @ (r.xvec - x0vec)) / sigma)

    minimize = cp.Minimize(twonll)
    prob = cp.Problem(minimize)
    prob.solve()

    rectangledict = {rectangle.n: i for i, rectangle in enumerate(self.rectangles)}
    return StitchResultCvxpy(x=x, T=T, problem=prob, rectangledict=rectangledict)

@dataclasses.dataclass
class Rectangle:
  n: int
  x: float
  y: float
  w: int
  h: int
  cx: int
  cy: int
  t: int
  file: str

  @property
  def xvec(self):
    return np.array([self.x, self.y])

@dataclasses.dataclass(frozen=True)
class ImageStats:
  n: int
  mean: float
  min: float
  max: float
  std: float
  cx: int
  cy: int

class StitchResultBase:
  def __init__(self, x, T, rectangledict):
    self.__x = x
    self.T = T
    self.__rectangledict = rectangledict

  def x(self, rectangle_or_id=None):
    if rectangle_or_id is None: return self.__x
    if isinstance(rectangle_or_id, Rectangle): rectangle_or_id = rectangle_or_id.n
    return self.__x[self.__rectangledict[rectangle_or_id]]

  def dx(self, overlap):
    return self.x(overlap.p1) - self.x(overlap.p2) - (overlap.x1vec - overlap.x2vec)

class StitchResult(StitchResultBase):
  def __init__(self, x, T, rectangledict, A, b, c):
    super().__init__(x=x, T=T, rectangledict=rectangledict)
    self.A = A
    self.b = b
    self.c = c

class StitchResultCvxpy(StitchResultBase):
  def __init__(self, x, T, rectangledict, problem):
    super().__init__(x=x.value, T=T.value, rectangledict=rectangledict)
    self.problem = problem
    self.xvar = x
    self.Tvar = T

if __name__ == "__main__":
  print(Aligner(r"G:\heshy", r"G:\heshy\flatw", "M21_1", 0))
