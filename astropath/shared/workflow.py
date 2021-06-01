from .argumentparser import RunFromArgumentParserBase
from ..slides.align.aligncohort import AlignCohort
from ..slides.annowarp.annowarpcohort import AnnoWarpCohortSelectMask
from ..slides.csvscan.csvscancohort import CsvScanCohort
from ..slides.deepzoom.deepzoomcohort import DeepZoomCohort
from ..slides.geom.geomcohort import GeomCohort
from ..slides.geomcell.geomcellcohort import GeomCellCohort
from ..slides.prepdb.prepdbcohort import PrepDbCohort
from ..slides.zoom.stitchmaskcohort import StitchAstroPathTissueMaskCohort
from ..slides.zoom.zoomcohort import ZoomCohort

class Workflow(RunFromArgumentParserBase):
  cohorts = PrepDbCohort, AlignCohort, ZoomCohort, DeepZoomCohort, StitchAstroPathTissueMaskCohort, AnnoWarpCohortSelectMask, GeomCohort, GeomCellCohort, CsvScanCohort

  _istmpclass = False
  @classmethod
  def makeargumentparser(cls):
    if cls._istmpclass:
      return super().makeargumentparser()

    class tmpclass(cls, *cls.cohorts):
      _istmpclass = True
    p = tmpclass.makeargumentparser()
    return p

  @classmethod
  def runfromparsedargs(cls, parsed_args):
    for cohort in cls.cohorts:
      p = cohort.makeargumentparser()
      cohort.runfromparsedargs(
        argparse.Namespace(**{
          k: v for k, v in parsed_args.__dict__
          if any(action.dest == k for action in p._actions)
        })
      )

def main(args=None):
  Workflow.runfromargumentparser(args=args)

if __name__ == "__main__":
  main()
