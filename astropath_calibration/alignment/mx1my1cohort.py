import numpy as np, pathlib

from ..baseclasses.sample import ReadRectanglesDbloadIm3
from ..baseclasses.cohort import DbloadCohort, Im3Cohort
from ..utilities import units
from .alignmentset import AlignmentSet
from .field import FieldReadIm3

class Mx1My1Sample(ReadRectanglesDbloadIm3):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, filetype="flatw", **kwargs)
  @property
  def rectanglecsv(self): return "fields"
  rectangletype = FieldReadIm3
  @property
  def logmodule(self): return "readmx1my1"
  def checkprint(self):
    minmx1 = min(f.mx1 for f in self.rectangles)
    minmy1 = min(f.my1 for f in self.rectangles)
    if minmx1 < 0 or minmy1 < 0:
      print(f"{self.SlideID:8}", f"{minmx1:5.0f} {minmy1:5.0f}")

class Mx1My1Cohort(DbloadCohort, Im3Cohort):
  def __init__(self, *args, uselogfiles=False, **kwargs):
    super().__init__(*args, uselogfiles=False, **kwargs)

  sampleclass = Mx1My1Sample
  @property
  def logmodule(self): return "readmx1my1"

  def runsample(self, sample):
    sample.checkprint()

def main(args=None):
  Mx1My1Cohort.runfromargumentparser(args)

if __name__ == "__main__":
  main()
