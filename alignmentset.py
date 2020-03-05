#!/usr/bin/env python3

import collections, cv2, dataclasses, itertools, logging, methodtools, numpy as np, os, scipy, typing, uncertainties as unc, uncertainties.unumpy as unp

from .flatfield import meanimage
from .overlap import AlignmentResult, Overlap
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

    self.rectanglefilter = rectangleoroverlapfilter(selectrectangles)
    overlapfilter = rectangleoroverlapfilter(selectoverlaps)
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

    for overlap in self.overlaps:
      p1rect = [r for r in self.rectangles if r.n==overlap.p1]
      p2rect = [r for r in self.rectangles if r.n==overlap.p2]
      if not len(p1rect) == len(p2rect) == 1:
        raise ValueError(f"Expected exactly one rectangle each with n={overlap.p1} and {overlap.p2}, found {len(p1rect)} and {len(p2rect)}")
      overlap_rectangles = p1rect[0], p2rect[0]
      overlap.setalignmentinfo(layer=self.layer, pscale=self.pscale, nclip=self.nclip, rectangles=overlap_rectangles)

  @property
  def overlapsdict(self):
    return {(o.p1, o.p2): o for o in self.overlaps}

  @property
  @methodtools.lru_cache()
  def image(self):
    return cv2.imread(os.path.join(self.dbload, self.samp+"_qptiff.jpg"))

  @property
  def aligncsv(self):
    return os.path.join(self.dbload, self.samp+"_align.csv")

  def align(self,*,write_result=True,return_on_invalid_result=False,**kwargs):
    #if the raw images haven't already been loaded, load them with the default argument
    #if self.rawimages is None :
    #  self.getDAPI()

    logger.info("starting align loop for "+self.samp)

    alignments = []
    sum_mse = 0.
    done = set()

    for i, overlap in enumerate(self.overlaps, start=1):
      logger.info(f"aligning overlap {i}/{len(self.overlaps)}")
      if (overlap.p2, overlap.p1) in done:
        result = overlap.getinversealignment(self.overlapsdict[overlap.p2, overlap.p1])
      else:
        result = overlap.align(**kwargs)
      done.add((overlap.p1, overlap.p2))
      if result is not None: 
        alignments.append(result)
        if result.exit==0 :
          sum_mse+=result.mse[2]
        else :
          if return_on_invalid_result :
            logger.warning(f'WARNING: Overlap number {i} alignment result is invalid, returning 1e10!!')
            return 1e10
          else :
            logger.warning(f'WARNING: Overlap number {i} alignment result is invalid, adding 1e10 to sum_mse!!')
            sum_mse+=1e10
      else :
        if return_on_invalid_result :
            logger.warning(f'WARNING: Overlap number {i} alignment result is "None"; returning 1e10!!')
            return 1e10
        else :
          logger.warning(f'WARNING: Overlap number {i} alignment result is "None"!')
          sum_mse+=1e10

    if write_result :
      self.writealignments(alignments)

    logger.info("finished align loop for "+self.samp)
    return sum_mse

  def writealignments(self, alignments):
    writetable(self.aligncsv, alignments, retry=self.interactive)

  def readalignments(self):
    alignmentresults = {o.n: o for o in readtable(self.aligncsv, AlignmentResult)}
    for o in self.overlaps:
      try:
        o.result = alignmentresults[o.n]
      except KeyError:
        pass
    return alignmentresults

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
        #np.copyto(r.image,imgdict[imgdictfn][self.layer] / self.meanimage.flatfield,casting='no')
        np.copyto(r.image,imgdict[imgdictfn][self.layer],casting='no') #question for Alex: applying meanimage?

  def writeOverlapComparisonImages(self) :
    """
    Write out a figure for each overlap showing comparisons between the original and shifted images
    """
    for o in self.overlaps :
      o.writeShiftComparisonImages()

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

      result.writetable(
        os.path.join(self.dbload, self.samp+"_stitch.csv"),
        os.path.join(self.dbload, self.samp+"_stitch_covariance.csv"),
        retry=self.interactive,
      )

    return result

  def __stitch(self, *, scaleby=1, scalejittererror=1, scaleoverlaperror=1, fixpoint="origin"):
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
    overlaps = [o for o in self.overlaps if not o.result.exit]

    for o in overlaps:
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

    dxs, dys = zip(*(o.result.dxdy for o in overlaps))

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

    if fixpoint == "origin":
      x0vec = np.array([0, 0]) #fix the origin, linear scaling is with respect to that
    elif fixpoint == "center":
      x0vec = np.mean([r.xvec for r in self.rectangles], axis=0)
    else:
      x0vec = fixpoint

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

    delta2nllfor1sigma = 1

    covariancematrix = np.linalg.inv(A) * delta2nllfor1sigma
    result = np.array(unc.correlated_values(result, covariancematrix))

    x = result[:-4].reshape(len(self.rectangles), 2) * scaleby
    T = result[-4:].reshape(2, 2)

    return StitchResult(x=x, T=T, A=A, b=b, c=c, rectangledict=rectangledict)

  def __stitch_cvxpy(self, fixpoint="origin"):
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
    overlaps = [o for o in self.overlaps if not o.result.exit]

    for o in overlaps:
      x1 = rectanglex[o.p1]
      x2 = rectanglex[o.p2]
      twonll += 0.5 * cp.quad_form(
        x1 - x2 - o.result.dxvec - o.x1vec + o.x2vec,
        np.linalg.inv(o.result.covariance)
      )

    dxs, dys = zip(*(o.result.dxdy for o in overlaps))

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

    if fixpoint == "origin":
      x0vec = np.array([0, 0]) #fix the origin, linear scaling is with respect to that
    elif fixpoint == "center":
      x0vec = np.mean([r.xvec for r in self.rectangles], axis=0)
    else:
      x0vec = fixpoint

    for r in self.rectangles:
      twonll += cp.norm(((rectanglex[r.n] - x0vec) - T @ (r.xvec - x0vec)) / sigma)

    minimize = cp.Minimize(twonll)
    prob = cp.Problem(minimize)
    prob.solve()

    rectangledict = {rectangle.n: i for i, rectangle in enumerate(self.rectangles)}
    return StitchResultCvxpy(x=x, T=T, problem=prob, rectangledict=rectangledict)

  def subset(self, *, selectrectangles=None, selectoverlaps=None):
    rectanglefilter = rectangleoroverlapfilter(selectrectangles)
    overlapfilter = rectangleoroverlapfilter(selectoverlaps)

    result = AlignmentSet(
      self.root1, self.root2, self.samp,
      interactive=self.interactive,
      selectrectangles=lambda r: self.rectanglefilter(r) and rectanglefilter(r),
      selectoverlaps=lambda o: self.overlapfilter(o) and overlapfilter(o),
    )
    for i, rectangle in enumerate(result.rectangles):
      result.rectangles[i] = [r for r in self.rectangles if r.n == rectangle.n][0]
    result.meanimage = self.meanimage
    result.images = self.images
    for i, overlap in enumerate(result.overlaps):
      result.overlaps[i] = [o for o in self.overlaps if o.n == overlap.n][0]
    return result

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

def rectangleoroverlapfilter(selection):
  if selection is None:
    return lambda r: True
  elif isinstance(selection, collections.abc.Container):
    return lambda r: r.n in selection
  else:
    return selection

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

  def writetable(self, filename, covariancefilename, **kwargs):
    n = 0
    rows = []
    for rectangleid in self.__rectangledict:
      for coordinate, position in enumerate(self.x(rectangleid)):
        n += 1
        rows.append(
          StitchCoordinate(
            n=n,
            rectangle=rectangleid,
            coordinate=coordinate,
            position=position,
          )
        )
    i = 0
    for i, Tii in enumerate(np.ravel(self.T)):
      n+=1
      rows.append(
        StitchCoordinate(
          n=n,
          rectangle=-99,
          coordinate=i,
          position=Tii,
        )
      )
    writetable(filename, rows, **kwargs)

    n = 0
    covrows = []
    for row1, row2 in itertools.combinations_with_replacement(rows, 2):
      n += 1
      covrows.append(
        StitchCovarianceEntry(
          n=n,
          coordinate1=row1.n,
          coordinate2=row2.n,
          covariance=unc.covariance_matrix([row1.positionwithuncertainty, row2.positionwithuncertainty])[0][1],
        )
      )
    writetable(covariancefilename, covrows, **kwargs)

@dataclasses.dataclass
class StitchCoordinate:
  n: int
  rectangle: int  #rectangle = -99: T matrix
  coordinate: int #for a rectangle: 0=x, 1=y
                  #for T matrix: 0=xx, 1=xy, 2=yx, 3=yy
  position: float

  def __init__(self, n, rectangle, coordinate, position):
    self.n = n
    self.rectangle = rectangle
    self.coordinate = coordinate
    self.positionwithuncertainty = position
    self.position = unc.nominal_value(position)

@dataclasses.dataclass
class StitchCovarianceEntry:
  n: int
  coordinate1: int
  coordinate2: int
  covariance: float

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
