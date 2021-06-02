#imports
from ...shared.sample import ReadRectanglesIm3FromXML, WorkflowSample

class MeanImageSample(ReadRectanglesIm3FromXML,WorkflowSample) :
    """
    Main class to handle creating the meanimage for a slide
    """

    #################### PUBLIC FUNCTIONS ####################

    def __init__(self,*args,**kwargs) :
        super().__init__(*args,**kwargs)

    #################### CLASS METHODS ####################

    @classmethod
    def defaultunits(cls):
        return "fast"
    @classmethod
    def logmodule(self): 
        return "slide_mean_image"

#################### FILE-SCOPE FUNCTIONS ####################

def main(args=None) :
    MeanImageSample.runfromargumentparser(args)

if __name__=='__main__' :
    main()