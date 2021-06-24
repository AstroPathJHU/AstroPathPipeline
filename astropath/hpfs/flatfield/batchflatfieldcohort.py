#imports
from .meanimage import MeanImage
from ...shared.sample import WorkflowSample
from ...shared.cohort import WorkflowCohort

class BatchFlatfieldSample(WorkflowSample) :
    """
    Small utility class to hold sample-dependent information for the batch flatfield run
    Just requires as input files the output of the meanimage mode
    """
    @property
    def inputfiles(self,**kwargs) :
        pass

class BatchFlatfieldCohort(WorkflowCohort) :
    """
    Class to handle combining several samples' meanimages into a single flatfield model for a batch
    """

    #################### PUBLIC FUNCTIONS ####################

    def __init__(self,*args,**kwargs) :
        super().__init__(*args,**kwargs)

    def run(self) :
        pass

    def runsample(self, sample, **kwargs) :
        """
        The "samples" run by this cohort should be run using a meanimagecohort instead
        """
        self.logger.error(f'ERROR: meanimage for sample {sample.SlideID} cannot be run using BatchFlatfieldCohort!')

    #################### CLASS VARIABLES + PROPERTIES ####################

    sampleclass = BatchFlatfieldSample

    @classmethod
    def defaultunits(cls) :
        return "fast"
    @classmethod
    def logmodule(cls) : 
        return "batchflatfield"

    #################### CLASS METHODS ####################

    @classmethod
    def makeargumentparser(cls) :
        pass