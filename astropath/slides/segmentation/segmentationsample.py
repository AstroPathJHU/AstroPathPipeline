#imports
from ...shared.argumentparser import WorkingDirArgumentParser
from ...shared.sample import ReadRectanglesComponentTiffFromXML, WorkflowSample, ParallelSample

class SegmentationSample(ReadRectanglesComponentTiffFromXML,WorkflowSample,ParallelSample,WorkingDirArgumentParser) :
    """
    Write out nuclear segmentation maps based on the DAPI layers of component tiffs for a single sample
    Algorithms available include pre-trained nnU-Net and DeepCell/mesmer models 
    """

    #################### PUBLIC FUNCTIONS ####################

    def __init__(self,*args,workingdir=None,**kwargs) :
        super().__init__(*args,**kwargs)
        self.__workingdir = workingdir

    def inputfiles(self,**kwargs) :
        return [*super().inputfiles(**kwargs),
                *(r.imagefile for r in self.rectangles),
               ]

    def run(self) :
        pass

    #################### PROPERTIES ####################

    @property
    def workflowkwargs(self) :
        return {
            **super().workflowkwargs,
            'workingdir':self.__workingdir
        }

    #################### CLASS METHODS ####################

    @classmethod
    def getoutputfiles(cls,SlideID,root,workingdir,**otherworkflowkwargs) :
        pass

    @classmethod
    def logmodule(cls) : 
        return "segmentation"
    @classmethod
    def workflowdependencyclasses(cls, **kwargs):
        return super().workflowdependencyclasses(**kwargs)

#################### FILE-SCOPE FUNCTIONS ####################

def main(args=None) :
    SegmentationSample.runfromargumentparser(args)

if __name__=='__main__' :
    main()
