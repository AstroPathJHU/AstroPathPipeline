#imports
from ...shared.argumentparser import WorkingDirArgumentParser, FileTypeArgumentParser
from ...shared.sample import ReadCorrectedRectanglesIm3MultiLayerFromXML, WorkflowSample, ParallelSample

class ImageCorrectionSample(ReadCorrectedRectanglesIm3MultiLayerFromXML, WorkflowSample, ParallelSample, WorkingDirArgumentParser, FileTypeArgumentParser) :
    """
    Read raw image files, correct them for flatfielding and/or warping effects 
    (not exposure time), and write them out as .fw files or .fw[layer] files
    """

    def __init__(self,*args,workingdir=None,**kwargs) :
        super().__init__(*args,**kwargs)
        self.__workingdir = workingdir
        if self.__workingdir is None :
            flatw_dir_name = 'flatw'
            if len(self.root.name.split('_'))>2 :
                flatw_dir_name+=f'_{"_".join([s for s in self.root.split("_")[2:]])}'
            self.__workingdir = self.root.parent / flatw_dir_name / self.SlideID

    def inputfiles(self,**kwargs) :
        pass

    def run(self) :
        pass

    @classmethod
    def getoutputfiles(cls,SlideID,root,**otherworkflowkwargs) :
        pass
    @classmethod
    def logmodule(cls) : 
        return "imagecorrection"
    @classmethod
    def workflowdependencyclasses(cls):
        return super().workflowdependencyclasses()

def main(args=None) :
    ImageCorrectionSample.runfromargumentparser(args)

if __name__=='__main__' :
    main()
