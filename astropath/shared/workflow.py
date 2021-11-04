import argparse
from .argumentparser import RunFromArgumentParserBase
from ..slides.align.aligncohort import AlignCohort
from ..slides.annowarp.annowarpcohort import AnnoWarpCohortAstroPathTissueMask
from ..slides.csvscan.csvscancohort import CsvScanCohort
from ..slides.deepzoom.deepzoomcohort import DeepZoomCohort
from ..slides.geom.geomcohort import GeomCohort
from ..slides.geomcell.geomcellcohort import GeomCellCohort
from ..slides.prepdb.prepdbcohort import PrepDbCohort
from ..slides.stitchmask.stitchmaskcohort import StitchAstroPathTissueMaskCohort
from ..slides.zoom.zoomcohort import ZoomCohort

class Workflow(RunFromArgumentParserBase):
  """
  Run the full AstroPath slide processing workflow.
  """

  cohorts = PrepDbCohort, AlignCohort, StitchAstroPathTissueMaskCohort, ZoomCohort, DeepZoomCohort, AnnoWarpCohortAstroPathTissueMask, GeomCohort, GeomCellCohort, CsvScanCohort

  _istmpclass = False
  @classmethod
  def makeargumentparser(cls, **kwargs):
    if cls._istmpclass:
      return super().makeargumentparser(**kwargs)

    class tmpclass(cls, *cls.cohorts):
      _istmpclass = True
      @classmethod
      def argumentparserhelpmessage(cls):
        return Workflow.__doc__

    p = tmpclass.makeargumentparser(_forworkflow=True, **kwargs)
    return p

  @classmethod
  def runfromparsedargs(cls, parsed_args):
    for cohort in cls.cohorts:
      p = cohort.makeargumentparser(_forworkflow=True)
      print(cohort)
      cohort.runfromparsedargs(
        argparse.Namespace(**{
          k: v for k, v in parsed_args.__dict__.items()
          if any(action.dest == k for action in p._actions)
        })
      )

def main(args=None):
  Workflow.runfromargumentparser(args=args)

if __name__ == "__main__":
  main()
