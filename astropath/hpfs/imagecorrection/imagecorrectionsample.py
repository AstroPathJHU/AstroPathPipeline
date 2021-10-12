#imports
from ...shared.argumentparser import SelectLayersArgumentParser, WorkingDirArgumentParser
from ...shared.sample import ReadCorrectedRectanglesIm3MultiLayerFromXML, WorkflowSample, ParallelSample
from ...utilities.img_file_io import write_image_to_file
from ...utilities.misc import cd
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
            self.__workingdir = self.__class__.automatic_output_dir(self.SlideID,self.root)
        self.__workingdir.mkdir(parents=True,exist_ok=True)

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
        with cd(self.__workingdir) :
            if self.njobs>1 :
                proc_results = {}
                with self.pool() as pool :
                    for ri,r in enumerate(self.rectangles,start=1) :
                        self.logger.debug(f'{r.file.rstrip(UNIV_CONST.IM3_EXT)} {msg_append} ({ri}/{len(self.rectangles)})....')
                        with r.using_image() as im :
                            proc_results[(r.n,r.file)] = pool.apply_async(write_out_corrected_image_files,
                                                                          (im,r.file.rstrip(UNIV_CONST.IM3_EXT),layers)
                                                                        )
                    for (rn,rfile),res in proc_results.items() :
                        try :
                            res.get()
                        except Exception as e :
                            warnmsg = f'WARNING: writing out corrected images for rectangle {rn} ({rfile.rstrip(UNIV_CONST.IM3_EXT)}) failed '
                            warnmsg+= f'with the error "{e}"'
                            self.logger.warning(warnmsg)
            else :
                for ri,r in enumerate(self.rectangles,start=1) :
                    self.logger.debug(f'{r.file.rstrip(UNIV_CONST.IM3_EXT)} {msg_append} ({ri}/{len(self.rectangles)})....')
                    try :
                        with r.using_image() as im :
                            write_out_corrected_image_files(im,r.file.rstrip(UNIV_CONST.IM3_EXT),layers)
                    except Exception as e :
                        warnmsg = f'WARNING: writing out corrected images for rectangle {r.n} ({r.file.rstrip(UNIV_CONST.IM3_EXT)}) failed '
                        warnmsg+= f'with the error "{e}"'
                        self.logger.warning(warnmsg) 

    #################### PROPERTIES ####################

    @property
    def workflowkwargs(self) :
        return {
            **super().workflowkwargs,
            'layers':self.layers,
        }

    #################### CLASS METHODS ####################

    @classmethod
    def automatic_output_dir(cls,SlideID,root) :
        flatw_dir_name = 'flatw'
        if len(root.name.split('_'))>2 :
            flatw_dir_name+=f'_{"_".join([s for s in root.split("_")[2:]])}'
        return root.parent / flatw_dir_name / SlideID

    @classmethod
    def getoutputfiles(cls,SlideID,root,shardedim3root,layers,**otherworkflowkwargs) :
        all_rawfile_stems = [rfp.name.rstrip(UNIV_CONST.RAW_EXT) for rfp in (shardedim3root/SlideID).glob(f'*{UNIV_CONST.RAW_EXT}')]
        outputfiles = []
        for rfs in all_rawfile_stems :
            if type(layers)==range : #if it's a range then it's just the multilayer images
                outputfilename = f'{rfs}{UNIV_CONST.FLATW_EXT}'
                outputfiles.append(cls.automatic_output_dir(SlideID,root) / outputfilename)
            else :
                for ln in layers :
                    if ln==-1 : #then the multilayer files should be saved
                        outputfilename = f'{rfs}{UNIV_CONST.FLATW_EXT}'
                        outputfiles.append(cls.automatic_output_dir(SlideID,root) / outputfilename)
                    else :
                        outputfilename = f'{rfs}{UNIV_CONST.FLATW_EXT}{ln:02d}'
                        outputfiles.append(cls.automatic_output_dir(SlideID,root) / outputfilename)
        return outputfiles

    @classmethod
    def logmodule(cls) : 
        return "imagecorrection"
    @classmethod
    def workflowdependencyclasses(cls):
        return super().workflowdependencyclasses()
    @classmethod
    def initkwargsfromargumentparser(cls, parsed_args_dict):
        to_return = super().initkwargsfromargumentparser(parsed_args_dict)
        to_return['filetype']='raw' # only ever run image correction on raw files
        to_return['skip_et_corrections']=True # never apply corrections for exposure time
        return to_return

#################### FILE-SCOPE FUNCTIONS ####################

def write_out_corrected_image_files(im,filenamestem,layers) :
    for ln in layers :
        if ln==-1 :
            write_image_to_file(im,filenamestem+UNIV_CONST.FLATW_EXT)
        else :
            write_image_to_file(im[:,:,ln-1],f'{filenamestem}{UNIV_CONST.FLATW_EXT}{ln:02d}')

def main(args=None) :
    ImageCorrectionSample.runfromargumentparser(args)

if __name__=='__main__' :
    main()
