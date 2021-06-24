#imports
from .meanimagesample import MeanImageSample
from ...shared.cohort import Im3Cohort, WorkflowCohort

class MeanImageCohort(Im3Cohort,WorkflowCohort) :
    sampleclass = MeanImageSample
    __doc__ = sampleclass.__doc__

def main(args=None):
    MeanImageCohort.runfromargumentparser(args)

if __name__ == "__main__":
    main()
