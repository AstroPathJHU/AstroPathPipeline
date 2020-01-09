#!/usr/bin/env python3

import cv2, logging, os

from readtable import readtable

logger = logging.getLogger("align")

class AlignmentError(Exception):
    def __init__(self, errormessage, errorid):
        self.errorid = errorid
        super().__init__(errormessage)

class Aligner(object):
    def __init__(self, root1, root2, samp, opt):
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
        try:
            self.annotations = readtable(os.path.join(self.dbload, self.samp+"_annotations.csv"))
            self.regions = readtable(os.path.join(self.dbload, self.samp+"_regions.csv"))
            self.vertices = readtable(os.path.join(self.dbload, self.samp+"_vertices.csv"))
            self.batch = readtable(os.path.join(self.dbload, self.samp+"_batch.csv"))
            self.overlap = readtable(os.path.join(self.dbload, self.samp+"_overlap.csv"))
            self.imagetable = readtable(os.path.join(self.dbload, self.samp+"_qptiff.csv"))
            self.image  = cv2.imread(os.path.join(self.dbload, self.samp+"_qptiff.jpg"))
            self.constants = readtable(os.path.join(self.dbload, self.samp+"_constants.csv"))
            self.rectangles = readtable(os.path.join(self.dbload, self.samp+"_rect.csv"))
        except:
            raise AlignmentError(f"ERROR in reading metadata files in {self.dbload}", 1)
            
        self.constantsdict = {constant.name: constant.value for name, value in self.constants}
        
        self.scan = f"Scan{self.batch[0].scan:d}"
        
        self.fwidth    = self.constantsdict["fwidth"]
        self.fheight   = self.constantsdict["fheight"]
        self.pscale    = self.constantsdict["pscale"]
        self.qpscale   = self.constantsdict["qpscale"]
        self.xposition = self.constantsdict["xposition"]
        self.yposition = self.constantsdict["yposition"]
        self.nclip     = self.constantsdict["nclip"]
        self.layer     = self.constantsdict["layer"]
        
if __name__ == "__main__":
    print(Aligner(r"G:\heshy", r"G:\heshy\flatw", "M21_1", 0))
