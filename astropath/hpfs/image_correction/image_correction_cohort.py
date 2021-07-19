#imports 
from .image_correction_sample import ImageCorrectionSample
from ...shared.cohort import Im3Cohort, SelectRectanglesCohort, ParallelCohort, WorkflowCohort

class ImageCorrectionCohort(Im3Cohort, SelectRectanglesCohort, ParallelCohort, WorkflowCohort) :
    sampleclass = ImageCorrectionSample
    __doc__ = sampleclass.__doc__

    def __init__(self,*args,**kwargs) :
        super().__init__(*args,**kwargs)
