#!/usr/bin/env python3

import cv2, logging, numpy as np, os

from .flatfield import meanimage
from .readtable import readtable

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

class Aligner:
  """
  Main class for running alignment
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
      self.overlap     = readtable(os.path.join(self.dbload, self.samp+"_overlap.csv"), "Overlap", n=int, p1=int, p2=int, x1=float, y1=float, x2=float, y2=float, tag=int)
      self.imagetable  = readtable(os.path.join(self.dbload, self.samp+"_qptiff.csv"), "ImageInfo", SampleID=int, XPosition=float, YPosition=float, XResolution=float, YResolution=float, qpscale=float, img=int)
      self.image       = cv2.imread(os.path.join(self.dbload, self.samp+"_qptiff.jpg"))
      self.constants   = readtable(os.path.join(self.dbload, self.samp+"_constants.csv"), "Constant", value=intorfloat)
      self.rectangles  = readtable(os.path.join(self.dbload, self.samp+"_rect.csv"), "Rectangle", n=int, x=float, y=float, w=int, h=int, cx=int, cy=int, t=int)
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

  def getDAPI(self):
    logger.info(self.samp)
    self.getrawlayers()

    if self.opt == -1:
      return

    # apply the extra flattening

    self.meanimage = meanimage(self.imagesraw, self.samp)

    self.images = self.imagesraw / self.meanimage.flatfield
    self.images = np.rint(self.images).astype(np.uint16)

    #todo: finish this function
    """
    %
    % get the image statistics
    %
    C.I = getImstat(C)
    %
    if (C.opt==0)
      writeImstat(C)
    end
    """

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

    self.imagesraw = np.array(images)

if __name__ == "__main__":
  print(Aligner(r"G:\heshy", r"G:\heshy\flatw", "M21_1", 0))
