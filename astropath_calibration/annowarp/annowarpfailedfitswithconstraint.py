import collections, methodtools, numpy as np
from ..utilities.dataclasses import MyDataClass
from ..utilities.tableio import readtable
from .annowarpcohort import AnnoWarpCohortBase
from .stitch import AnnoWarpStitchResultEntry

"""
If any sample in a cohort has too few tiles for the annowarp stitching to converge,
this script will rerun that sample using constraints obtained from the average of
the other samples in the cohort.

The constant pieces will be allowed to float, with a Gaussian constraint obtained
from the weighted average and standard deviation of the other samples in the cohort.
The other parameters are fixed to the weighted average of the other samples.
"""

class Stats(MyDataClass):
  """
  Container class for storing the average and standard deviation
  for a parameter.
  """
  description: str
  average: object
  std: object

class GatherStatsCohort(AnnoWarpCohortBase):
  """
  This cohort loops over the samples that finished successfully
  and saves the average and standard deviation for each parameter.
  """
  def __init__(self, *args, uselogfiles=False, filters=[], **kwargs):
    super().__init__(
      *args,
      **kwargs,
      uselogfiles=False,
      filters=filters+[lambda samp: (self.dbloadroot/samp.SlideID/"dbload"/f"{samp.SlideID}_warp-100-stitch.csv").exists()],
    )
    self.__parametervalues = collections.defaultdict(list)

  def runsample(self, sample):
    #read the stitch results from the sample and save the fitted parameter values and covariances
    stitchresults = readtable(sample.stitchcsv, AnnoWarpStitchResultEntry, extrakwargs={"pscale": sample.pscale})
    for sr in stitchresults:
      self.__parametervalues[sr.description].append(sr.value)

  #this is cached, because we only need to calculate the stats once
  @methodtools.lru_cache()
  def run(self): return super().run()

  @methodtools.lru_cache()
  @property
  def stats(self):
    self.run()
    stats = []
    for name, values in self.__parametervalues.items():
      if "covariance(" in name: continue
      #extract the fitted value and error for each parameter
      nominals = np.array(values)
      errors = np.array(self.__parametervalues[f"covariance({name}, {name})"]) ** .5
      #calculate the weighted average and standard deviation
      average = np.sum(nominals/errors**2) / np.sum(1/errors**2)
      std = np.sqrt(np.sum((nominals-average)**2/errors**2) / np.sum(1/errors**2))
      #save it in stats
      stats.append(Stats(description=name, average=average, std=std))
    return stats

class StitchFailedCohort(AnnoWarpCohortBase):
  """
  Rerun any samples that aligned successfully but didn't stitch,
  using a constraint obtained from the other samples in the cohort.

  The constant pieces will be allowed to float, with a Gaussian constraint obtained
  from the weighted average and standard deviation of the other samples in the cohort.
  The other parameters are fixed to the weighted average of the other samples.

  multiplystd: multiply the standard deviation of the parameters by this number
               to obtain the gaussian width
  """
  def __init__(self, *args, multiplystd=np.array([1]*8+[1]*2), filters=[], **kwargs):
    super().__init__(
      *args,
      **kwargs,
      filters=filters+[
        lambda samp:
          (self.dbloadroot/samp.SlideID/"dbload"/f"{samp.SlideID}_warp-100.csv").exists()
          and not (self.dbloadroot/samp.SlideID/"dbload"/f"{samp.SlideID}_warp-100-stitch.csv").exists()
      ],
    )
    self.__gatherstatscohort = GatherStatsCohort(*args, **kwargs)
    self.__multiplystd = multiplystd

  def runsample(self, sample):
    sample.runannowarp(
      readalignments=True,
      constraintmus=self.mus,
      constraintsigmas=self.sigmas,
      floatedparams="constants",
    )

  @property
  def mus(self):
    return [stats.average for stats in self.stats]
  @property
  def sigmas(self):
    return np.array([stats.std for stats in self.stats]) * self.__multiplystd

  @property
  def stats(self):
    return self.__gatherstatscohort.stats

def main(args=None):
  StitchFailedCohort.runfromargumentparser(args)

if __name__ == "__main__":
  main()
