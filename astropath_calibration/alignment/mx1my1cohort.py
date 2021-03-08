import numpy as np
from ..baseclasses.sample import ReadRectanglesDbloadIm3
from ..baseclasses.cohort import DbloadCohort, Im3Cohort
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
    rminmx1 = min(self.rectangles, key=lambda f: f.mx1)
    rminmy1 = min(self.rectangles, key=lambda f: f.my1)
    if rminmx1.mx1 < 0 or rminmy1.my1 < 0 or np.any(self.position <= 0):
      print(f"{self.SlideID:8}", f"{rminmx1.mx1:5.0f} {rminmy1.my1:5.0f} {self.position[0]:5.0f} {self.position[1]:5.0f} {rminmx1.x:5.0f} {rminmy1.y:5.0f}")

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
