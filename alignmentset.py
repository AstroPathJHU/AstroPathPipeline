#!/usr/bin/env python3

import cv2, dataclasses, functools, logging, numpy as np, os, typing

from .flatfield import meanimage
from .overlap import Overlap
from .readtable import readtable, writetable

logger = logging.getLogger("align")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(message)s, %(funcName)s, %(asctime)s"))
logger.addHandler(handler)

class AlignmentSet:
  """
  Main class for aligning a set of images
  """
  def __init__(self, root1, root2, samp, interactive=False):
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
        overlap_images = np.array([p1image[0],p2image[0]])
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

  @functools.lru_cache(maxsize=1)
  def getDAPI(self,filetype="flatWarpDAPI"):
    logger.info(self.samp)
    self.rawimages = self.__getrawlayers(filetype)

    # apply the extra flattening

    self.meanimage = meanimage(self.rawimages, self.samp)

    self.images = self.rawimages / self.meanimage.flatfield
    self.images = np.rint(self.images).astype(np.uint16)

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

  @functools.lru_cache(maxsize=1)
  def __getrawlayers(self,filetype):
    logger.info(self.samp)
    if filetype=="flatWarpDAPI" :
      ext = f".fw{self.layer:02d}"
    elif filetype=="camWarpDAPI" :
      ext = f".camWarp_layer{self.layer:02d}"
    else :
      raise ValueError(f"requested file type {filetype} not recognized by getrawlayers", 1)
    path = os.path.join(self.root2, self.samp)

    images = []

    if not self.rectangles:
      raise IOError("didn't find any rows in the rectangles table for "+self.samp, 1)

    for rectangle in self.rectangles:
      with open(os.path.join(path, rectangle.file.replace(".im3", ext)), "rb") as f:
        img = np.fromfile(f, np.uint16)
        #use fortran order, like matlab!
        images.append(img.reshape((self.fheight, self.fwidth), order="F"))

    rawimages = np.array(images)

    for rectangle, rawimage in zip(self.rectangles, rawimages):
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
