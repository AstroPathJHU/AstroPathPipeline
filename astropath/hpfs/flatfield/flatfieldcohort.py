#imports
from ...shared.cohort import Im3Cohort, SelectRectanglesCohort, WorkflowCohort
from .flatfieldsample import FlatfieldSample

class FlatfieldCohort(Im3Cohort,SelectRectanglesCohort,WorkflowCohort) :
    #set some attributes corresponding to the specific sample type
    __doc__ = FlatfieldSample.__doc__
    sampleclass = FlatfieldSample

def main(args=None):
    FlatfieldCohort.runfromargumentparser(args)

if __name__ == "__main__":
    main()
