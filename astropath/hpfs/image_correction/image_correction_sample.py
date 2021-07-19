#imports
from ...shared.argumentparser import SelectLayersArgumentParser, WorkingDirArgumentParser
from ...shared.sample import ReadCorrectedRectanglesIm3MultiLayerFromXML, WorkflowSample, ParallelSample
from ...utilities.img_file_io import write_image_to_file
from ...utilities.config import CONST as UNIV_CONST

class ImageCorrectionSample(ReadCorrectedRectanglesIm3MultiLayerFromXML, WorkflowSample, ParallelSample, WorkingDirArgumentParser, SelectLayersArgumentParser) :
    """
    Read raw image files, correct them for flatfielding and/or warping effects 
    (not exposure time), and write them out as .fw files or .fw[layer] files
    """

    #################### PUBLIC FUNCTIONS ####################

    def __init__(self,*args,workingdir=None,**kwargs) :
        super().__init__(*args,**kwargs)
        self.__workingdir = workingdir
        if self.__workingdir is None :
            self.__workingdir = self.automatic_output_dir

    def inputfiles(self,**kwargs) :
        return [*super().inputfiles(**kwargs),
                *(r.imagefile for r in self.rectangles),
               ]

    def run(self) :
        layers = self.layers
        if layers==range(1,self.nlayers+1) :
            layers = [-1]
        msg_append = f'{self.applied_corrections_string} and'
        if -1 in layers :
            msg_append+=' multilayer'
        if layers!=[-1] :
            if -1 in layers :
                msg_append+=' and'
            msg_append+=f' layer {",".join([str(ln) for ln in layers if ln!=-1])}'
        msg_append+=' files written out'
        for ir,r in enumerate(self.rectangles,start=1) :
            with r.using_image() as im :
                with cd(self.__workingdir) :
                    for ln in layers :
                        if ln==-1 :
                            write_image_to_file(im,r.imagefile.replace(UNIV_CONST.IM3_EXT,UNIV_CONST.FLATW_EXT))
                        else :
                            write_image_to_file(im[:,:,ln-1],r.imagefile.replace(UNIV_CONST.IM3_EXT,f'{UNIV_CONST.FLATW_EXT}{ln:02d}'))
            self.logger.debug(f'{r.imagefile.rstrip(UNIV_CONST.IM3_EXT)} {msg_append} ({ri}/{len(self.rectangles)})')

    #################### PROPERTIES ####################

    @property
    def automatic_output_dir(self) :
        flatw_dir_name = 'flatw'
        if len(self.root.name.split('_'))>2 :
            flatw_dir_name+=f'_{"_".join([s for s in self.root.split("_")[2:]])}'
        return self.root.parent / flatw_dir_name / self.SlideID
    @property
    def workflowkwargs(self) :
        return {
            **super().workflowkwargs,
            'nlayers':self.nlayers,
            'layers':self.layers,
            'output_dir':self.automatic_output_dir,
        }

    #################### CLASS METHODS ####################

    @classmethod
    def getoutputfiles(cls,SlideID,root,nlayers,layers,output_dir,**otherworkflowkwargs) :
        #if "layers" wasn't given in the command line arguments then reset the variable 
        #to show that just the multilayer files should be saved
        if layers==range(1,nlayers+1) :
            layers = [-1]
        #outputfiles are the corrected image files
        outputfiles = []
        for r in self.rectangles :
            for ln in layers :
                if ln==-1 : #then the multilayer files should be saved
                    outputfilename = r.imagefile.replace(UNIV_CONST.IM3_EXT,UNIV_CONST.FLATW_EXT)
                else :
                    outputfilename = r.imagefile.replace(UNIV_CONST.IM3_EXT,f'{UNIV_CONST.FLATW_EXT}{ln:02d}')
                outputfiles.append(output_dir / outputfilename)

    @classmethod
    def logmodule(cls) : 
        return "imagecorrection"
    @classmethod
    def workflowdependencyclasses(cls):
        return super().workflowdependencyclasses()
    @classmethod
    def initkwargsfromargumentparser(cls, parsed_args_dict):
        return {
            **super().initkwargsfromargumentparser(parsed_args_dict),
            "filetype": 'raw', # only ever run image correction on raw files
        }

def main(args=None) :
    ImageCorrectionSample.runfromargumentparser(args)

if __name__=='__main__' :
    main()
