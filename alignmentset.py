#!/usr/bin/env python3

import cv2, dataclasses, logging, numpy as np, os, typing

from .flatfield import meanimage
from .overlap import Overlap
from .readtable import readtable, writetable

logger = logging.getLogger("align")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(message)s, %(funcName)s, %(asctime)s"))
logger.addHandler(handler)

class AlignmentError(Exception):
  """
  Class for errors that come up during alignment.
  It has an error id, which is passed to the output of the alignment
  and stored in the csv file.
  """
  def __init__(self, errormessage, errorid):
    self.errorid = errorid
    super().__init__(errormessage)

class AlignmentSet:
  """
  Main class for aligning a set of images
  """
  def __init__(self, root1, root2, samp, opt=0):
    """
    Directory structure should be
    root1/
      samp/
        dbload/
          samp_*.csv
          samp_qptiff.jpg
    root2/
      samp/
        samp_*.fw01 (if using DAPI, could also be fw02 etc. to align with other markers)

    #todo: explain opt here
    """
    logger.info(samp)
    self.root1 = root1
    self.root2 = root2
    self.samp = samp
    self.opt = opt

    if not os.path.exists(os.path.join(self.root1, self.samp)):
      raise AlignmentError(f"{self.root1}/{self.samp} does not exist", 1)

    self.readmetadata()

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
    try:
      self.annotations = readtable(os.path.join(self.dbload, self.samp+"_annotations.csv"), "Annotation", sampleid=int, layer=int, visible=int)
      self.regions     = readtable(os.path.join(self.dbload, self.samp+"_regions.csv"), "Region", regionid=int, sampleid=int, layer=int, rid=int, isNeg=int, nvert=int)
      self.vertices    = readtable(os.path.join(self.dbload, self.samp+"_vertices.csv"), "Vertex", regionid=int, vid=int, x=int, y=int)
      self.batch       = readtable(os.path.join(self.dbload, self.samp+"_batch.csv"), "Batch", SampleID=int, Scan=int, Batch=int)
      self.overlaps    = readtable(os.path.join(self.dbload, self.samp+"_overlap.csv"), Overlap)
      self.imagetable  = readtable(os.path.join(self.dbload, self.samp+"_qptiff.csv"), "ImageInfo", SampleID=int, XPosition=float, YPosition=float, XResolution=float, YResolution=float, qpscale=float, img=int)
      self.image       = cv2.imread(os.path.join(self.dbload, self.samp+"_qptiff.jpg"))
      self.constants   = readtable(os.path.join(self.dbload, self.samp+"_constants.csv"), "Constant", value=intorfloat)
      self.rectangles  = readtable(os.path.join(self.dbload, self.samp+"_rect.csv"), Rectangle)
    except:
      raise AlignmentError(f"ERROR in reading metadata files in {self.dbload}", 1)

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


  def align(self):
    self.getDAPI()
    if self.opt != 0: return

    aligncsv = os.path.join(self.dbpath, self.samp+"_align.csv")

    logger.log("starting align loop for "+self.samp)

    alignments = []
    for i, overlap in enumerate(self.overlaps):
      logger.log(f"aligning overlap {i}/{len(self.overlaps)}")
      overlap.align()

    logger.log("finished align loop for "+self.samp)

  def getDAPI(self):
    logger.info(self.samp)
    self.getrawlayers()

    if self.opt == -1:
      return

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
    if self.opt==0:
      writetable(os.path.join(self.dbload, self.samp+"_imstat.csv"), self.imagestats)

  def getrawlayers(self):
    logger.info(self.samp)
    ext = f".fw{self.layer:02d}"
    path = os.path.join(self.root2, self.samp)

    images = []

    if not self.rectangles:
      raise AlignmentError("didn't find any rows in the rectangles table for "+self.samp, 1)

    for rectangle in self.rectangles:
      with open(os.path.join(path, rectangle.file.replace(".im3", ext)), "rb") as f:
        img = np.fromfile(f, np.uint16)
        #use fortran order, like matlab!
        images.append(img.reshape((self.fheight, self.fwidth), order="F"))

    self.rawimages = np.array(images)

    for rectangle, rawimage in zip(self.rectangles, self.rawimages):
      rectangle.rawimage = rawimage

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
