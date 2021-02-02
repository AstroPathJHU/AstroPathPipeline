import collections, methodtools, numpy as np
from ..baseclasses.cohort import DbloadCohort, ZoomCohort
from ..utilities.dataclasses import MyDataClass
from ..utilities.misc import dict_zip_equal
from ..utilities.tableio import readtable
from .annowarpsample import AnnoWarpSample
from .stitch import AnnoWarpStitchResultEntry

class Stats(MyDataClass):
  description: str
  average: object
  std: object

class GatherStatsSample(AnnoWarpSample):
  @property
  def logmodule(self): return "gatherannowarpstats"

class GatherStatsCohort(DbloadCohort, ZoomCohort):
  def __init__(self, *args, uselogfiles=False, filter=lambda samp: True, **kwargs):
    super().__init__(
      *args,
      **kwargs,
      uselogfiles=False,
      filter=lambda samp: filter(samp) and (self.dbloadroot/samp.SlideID/"dbload"/f"{samp.SlideID}_warp-100-stitch.csv").exists(),
    )
    self.__parametervalues = collections.defaultdict(list)

  sampleclass = GatherStatsSample

  def runsample(self, sample):
    stitchresults = readtable(sample.stitchcsv, AnnoWarpStitchResultEntry, extrakwargs={"pscale": sample.pscale})
    for sr in stitchresults:
      self.__parametervalues[sr.description].append(sr.value)

  @methodtools.lru_cache()
  def run(self): return super().run()

  @property
  def logmodule(self): return "gatherannowarpstats"

  @methodtools.lru_cache()
  @property
  def stats(self):
    self.run()
    stats = []
    for name, values in self.__parametervalues.items():
      if "covariance(" in name: continue
      nominals = np.array(values)
      errors = np.array(self.__parametervalues[f"covariance({name}, {name})"]) ** .5
      average = np.sum(nominals/errors**2) / np.sum(1/errors**2)
      std = np.sqrt(np.sum((nominals-average)**2/errors**2) / np.sum(1/errors**2))
      stats.append(Stats(description=name, average=average, std=std))
    return stats

class StitchFailedCohort(DbloadCohort, ZoomCohort):
  def __init__(self, *args, filter=lambda samp: True, multiplystd=np.array([0.0001]*8+[1]*2), **kwargs):
    super().__init__(
      *args,
      **kwargs,
      filter=lambda samp:
        filter(samp)
        and (self.dbloadroot/samp.SlideID/"dbload"/f"{samp.SlideID}_warp-100.csv").exists()
        and not (self.dbloadroot/samp.SlideID/"dbload"/f"{samp.SlideID}_warp-100-stitch.csv").exists(),
    )
    self.__gatherstatscohort = GatherStatsCohort(*args, **kwargs)
    self.__multiplystd = multiplystd

  def runsample(self, sample):
    sample.runannowarp(readalignments=True, constraintmus=self.mus, constraintsigmas=self.sigmas)

  @property
  def mus(self):
    return [stats.average for stats in self.stats]
  @property
  def sigmas(self):
    return np.array([stats.std for stats in self.stats]) * self.__multiplystd

  @property
  def stats(self):
    return self.__gatherstatscohort.stats

  sampleclass = AnnoWarpSample
  @property
  def logmodule(self): return "annowarp"

def main(args=None):
  StitchFailedCohort.runfromargumentparser(args)

if __name__ == "__main__":
  main()
