#!/usr/bin/env python3

import cv2, dataclasses, functools, logging, numpy as np, os, typing, uncertainties.unumpy as unp

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
  def __init__(self, root1, root2, samp, *, interactive=False, selectrectangles=None):
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
    self.selectrectangles = selectrectangles

    if not os.path.exists(os.path.join(self.root1, self.samp)):
      raise IOError(f"{os.path.join(self.root1, self.samp)} does not exist")

    self.readmetadata()
    self.rawimages=None

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
    self.image       = cv2.imread(os.path.join(self.dbload, self.samp+"_qptiff.jpg"))
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

    if self.selectrectangles is not None:
      self.rectangles = [r for r in self.rectangles if r.n in self.selectrectangles]
      self.overlaps = [o for o in self.overlaps if o.p1 in self.selectrectangles and o.p2 in self.selectrectangles]

    self.overlapsdict = {(o.p1, o.p2): o for o in self.overlaps}


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
        sum_mse+=result.mse[0]+result.mse[1]

    writetable(aligncsv, alignments, retry=self.interactive)

    logger.info("finished align loop for "+self.samp)
    return sum_mse

  @functools.lru_cache(maxsize=1)
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

  def stitch(self):
    """
    \begin{align}
    -2 \ln L =&
      1/2 \sum_\text{overlaps}
      (\vec{x}_{p1} - \vec{x}_{p2} - d\vec{x} - \vec{x}_{p1}^n + \vec{x}_{p2}^n)^T \\
      &\mathbf{cov}^{-1}
      (\vec{x}_{p1} - \vec{x}_{p2} - d\vec{x} - \vec{x}_{p1}^n + \vec{x}_{p2}^n) \\
      +&
      \sum_p \left(\frac{\vec{x}_p - \mathbf{T}\vec{x}_p^n}{\sigma}\right)^2
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
      inversecovariance = np.linalg.inv(o.result.covariance)

      A[ii] += inversecovariance / 2
      A[ij] -= inversecovariance / 2
      A[ji] -= inversecovariance / 2
      A[jj] += inversecovariance / 2

      i = np.ix_((ix, iy))
      j = np.ix_((jx, jy))

      constpiece = -o.result.dxvec - o.x1vec + o.x2vec

      b[i] += 2 * inversecovariance @ constpiece
      b[j] -= 2 * inversecovariance @ constpiece

      c += constpiece @ inversecovariance @ constpiece

      print(A); print(b); print(c)

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

    for r in self.rectangles:
      ix = 2*rectangledict[r.n]
      iy = 2*rectangledict[r.n]+1

      A[ix] += 1 / sigmax**2
      A[iy] += 1 / sigmay**2
      A[(ix, Txx), (Txx, ix)] -= r.cx / sigmax**2
      A[(ix, Txy), (Txy, ix)] -= r.cy / sigmax**2
      A[(iy, Tyx), (Tyx, iy)] -= r.cx / sigmay**2
      A[(iy, Tyy), (Tyy, iy)] -= r.cy / sigmay**2

      A[Txx, Txx]               += r.cx**2   / sigmax**2
      A[(Txx, Txy), (Txy, Txx)] += r.cx*r.cy / sigmax**2
      A[Txy, Txy]               += r.cy**2   / sigmax**2

      A[Tyx, Tyx]               += r.cx**2   / sigmay**2
      A[(Tyx, Tyy), (Tyy, Tyx)] += r.cx*r.cy / sigmay**2
      A[Tyy, Tyy]               += r.cy**2   / sigmay**2

      print(A); print(b); print(c)

    result = np.linalg.solve(A/2, -b)
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
  rawimage: typing.Optional[np.ndarray] = None
  image: typing.Optional[np.ndarray] = None

  @property
  def cxvec(self):
    return np.array([cx, cy])

@dataclasses.dataclass(frozen=True)
class ImageStats:
  n: int
  mean: float
  min: float
  max: float
  std: float
  cx: int
  cy: int

if __name__ == "__main__":
  print(Aligner(r"G:\heshy", r"G:\heshy\flatw", "M21_1", 0))
