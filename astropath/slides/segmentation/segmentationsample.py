#imports
import os, shutil
from deepcell.applications import NuclearSegmentation
from ...utilities.config import CONST as UNIV_CONST
from ...shared.argumentparser import SegmentationAlgorithmArgumentParser,WorkingDirArgumentParser
from ...shared.sample import ReadRectanglesComponentTiffFromXML, WorkflowSample, ParallelSample
from .config import SEG_CONST
from .utilities import rebuild_model_files_if_necessary, write_nifti_file_for_rect_im
from .utilities import convert_nnunet_output, run_deepcell_nuclear_segmentation

#some constants
NNUNET_SEGMENT_FILE_APPEND = 'nnunet_nuclear_segmentation.npz'
DEEPCELL_SEGMENT_FILE_APPEND = 'deepcell_nuclear_segmentation.npz'

class SegmentationSample(ReadRectanglesComponentTiffFromXML,WorkflowSample,ParallelSample,
                         SegmentationAlgorithmArgumentParser,WorkingDirArgumentParser) :
    """
    Write out nuclear segmentation maps based on the DAPI layers of component tiffs for a single sample
    Algorithms available include pre-trained nnU-Net and DeepCell/mesmer models 
    """

    #################### PUBLIC FUNCTIONS ####################

    def __init__(self,*args,workingdir=None,algorithm='nnunet',**kwargs) :
        # only need to load the DAPI layers of the rectangles, so send that to the __init__
        if kwargs.get('layer') is not None :
            raise RuntimeError(f'ERROR: sample layer was set to {kwargs.get("layer")}')
        kwargs['layer'] = 1 
        super().__init__(*args,**kwargs)
        self.__algorithm = algorithm
        #set the working directory path based on the algorithm being run (if it wasn't set by a command line arg)
        self.__workingdir = SegmentationSample.output_dir(workingdir,self.im3root,self.SlideID,self.__algorithm)

    def inputfiles(self,**kwargs) :
        return [*super().inputfiles(**kwargs),
                *(r.imagefile for r in self.rectangles),
               ]

    def run(self) :
        if not self.__workingdir.is_dir() :
            self.__workingdir.mkdir(parents=True)
        if self.__algorithm=='nnunet' :
            self.__run_nnunet()
        elif self.__algorithm=='deepcell' :
            self.__run_deepcell()
        else :
            raise ValueError(f'ERROR: algorithm choice {self.__algorithm} is not recognized!')

    #################### PROPERTIES ####################

    @property
    def workflowkwargs(self) :
        return {
            **super().workflowkwargs,
            'workingdir':self.__workingdir,
            'algorithm':self.__algorithm,
        }

    #################### CLASS METHODS ####################

    @classmethod
    def output_dir(cls,workingdir,im3root,SlideID,algorithm) :
        #default output is im3folder/segmentation/algorithm
        outputdir = workingdir
        if outputdir is None :
            outputdir = im3root/SlideID/'im3'/SEG_CONST.SEGMENTATION_DIR_NAME/algorithm
        elif not outputdir.name==SlideID :
            #put non-default output in a subdirectory named for the slide
            outputdir = outputdir/SlideID
        return outputdir

    @classmethod
    def getoutputfiles(cls,SlideID,im3root,informdataroot,workingdir,algorithm,**otherworkflowkwargs) :
        outputdir=cls.automatic_output_dir(workingdir,im3root,SlideID,algorithm)
        append = None
        if algorithm=='nnunet' :
            append = NNUNET_SEGMENT_FILE_APPEND
        elif algorithm=='deepcell' :
            append = DEEPCELL_SEGMENT_FILE_APPEND
        file_stems = [fp.name[:-len('_component_data.tif')] for fp in (informdataroot/'Component_Tiffs').glob('*_component_data.tif')]
        outputfiles = []
        for stem in file_stems :
            outputfiles.append(outputdir/f'{stem}_{append}')
        return outputfiles

    @classmethod
    def logmodule(cls) : 
        return "segmentation"

    @classmethod
    def workflowdependencyclasses(cls, **kwargs):
        return super().workflowdependencyclasses(**kwargs)

    #################### PRIVATE HELPER FUNCTIONS ####################

    def __get_rect_nifti_fp(self,rect) :
        return self.temp_dir/f'{rect.imagefile.name[:-4]}_0000.nii.gz'

    def __get_rect_segmented_nifti_fp(self,rect) :
        return self.__workingdir/f'{rect.imagefile.name[:-4]}.nii.gz'

    def __get_rect_nnunet_segmented_fp(self,rect) :
        seg_fn = f'{rect.file.rstrip(UNIV_CONST.IM3_EXT)}_{NNUNET_SEGMENT_FILE_APPEND}'
        return self.__workingdir/seg_fn

    def __get_rect_deepcell_segmented_fp(self,rect) :
        seg_fn = f'{rect.file.rstrip(UNIV_CONST.IM3_EXT)}_{DEEPCELL_SEGMENT_FILE_APPEND}'
        return self.__workingdir/seg_fn

    def __run_nnunet(self) :
        """
        Run nuclear segmentation using the pre-trained nnU-Net algorithm
        """
        #make sure that the necessary model files exist
        rebuild_model_files_if_necessary()
        #create the temporary directory that will hold the NIfTI files
        self.temp_dir = self.__workingdir/'nnunet_nifti_input'
        if not self.temp_dir.is_dir() :
            self.temp_dir.mkdir(parents=True)
        #write a NIfTI file for every rectangle's DAPI layer for input to the algorithm
        self.logger.debug('Writing NIfTI files for nnU-Net input....')
        rects_to_run = []
        for ir,rect in enumerate(self.rectangles,start=1) :
            #skip any rectangles that already have segmentation input or output
            if ( self.__get_rect_nifti_fp(rect).is_file() or self.__get_rect_segmented_nifti_fp(rect).is_file() 
                or self.__get_rect_nnunet_segmented_fp(rect).is_file() ) :
                msg = f'Skipping writing NIfTI file for {rect.imagefile.name} ({ir} of {len(self.rectangles)})'
                if ( self.__get_rect_segmented_nifti_fp(rect).is_file() or 
                    self.__get_rect_nnunet_segmented_fp(rect).is_file() ) :
                    msg+=' (segmentation output already exists)'
                elif self.__get_rect_nifti_fp(rect).is_file() :
                    msg+=' (NIfTI file already exists in temp. directory)'
                self.logger.debug(msg)
                continue
            else :
                rects_to_run.append((ir,rect,self.__get_rect_nifti_fp(rect)))
        if self.njobs is not None and self.njobs>1 :
            proc_results = {}
            with self.pool() as pool :
                for ir,rect,nifti_file_path in rects_to_run :
                    with rect.using_image() as im :
                        msg = f'Writing NIfTI file for {rect.imagefile.name} ({ir} of {len(self.rectangles)})'
                        self.logger.debug(msg)
                        proc_results[(ir,rect.imagefile.name)] = pool.apply_async(write_nifti_file_for_rect_im,
                                                                                  (im,nifti_file_path))
                for (ir,rname),res in proc_results.items() :
                    try :
                        _ = res.get()
                    except Exception as e :
                        errmsg = f'ERROR: failed to write NIfTI file for {rname} '
                        errmsg+= f'({ir} of {len(self.rectangles)}). Exception will be reraised.'
                        self.logger.error(errmsg)
                        raise e
        else :
            for ir,rect,nifti_file_path in rects_to_run :
                with rect.using_image() as im :
                    self.logger.debug(f'Writing NIfTI file for {rect.imagefile.name} ({ir} of {len(self.rectangles)})')
                    write_nifti_file_for_rect_im(im,nifti_file_path)
        #run the nnU-Net nuclear segmentation algorithm
        os.environ['RESULTS_FOLDER'] = str(SEG_CONST.NNUNET_MODEL_TOP_DIR.resolve())
        nnunet_cmd = f'nnUNet_predict -i {str(self.temp_dir.resolve())} -o {str(self.__workingdir.resolve())}'
        nnunet_cmd+= f' -t 500 -m 2d --num_threads_preprocessing {self.njobs} --num_threads_nifti_save {self.njobs}'
        self.logger.debug('Running nuclear segmentation with nnU-Net....')
        try :
            os.system(nnunet_cmd)
        except Exception as e :
            errmsg = 'ERROR: some exception was raised while running nnU-Net. Output will be collected and then '
            errmsg = 'the exception will be reraised. See log lines above for what was completed.'
            self.logger.error(errmsg)
            raise e
        finally :
            completed_files = 0
            rects_to_run = []
            for ir,rect in enumerate(self.rectangles,start=1) :
                if self.__get_rect_segmented_nifti_fp(rect).is_file() :
                    rects_to_run.append(
                        (ir,rect,self.__get_rect_segmented_nifti_fp(rect),
                        self.__get_rect_nnunet_segmented_fp(rect))
                    )
            if self.njobs is not None and self.njobs>1 :
                proc_results = {}
                with self.pool() as pool :
                    for ir,rect,segmented_nifti_path,segmented_file_path in rects_to_run :
                        msg = f'Converting nnU-Net output for {rect.imagefile.name} ({ir} of {len(self.rectangles)})'
                        self.logger.debug(msg)
                        proc_results[(ir,rect.imagefile.name)] = pool.apply_async(convert_nnunet_output,
                                                                                  (segmented_nifti_path,
                                                                                  segmented_file_path))
                    for (ir,rname),res in proc_results.items() :
                        try :
                            _ = res.get()
                        except Exception as e :
                            errmsg = f'ERROR: failed to convert nnU-Net output for {rname} '
                            errmsg+= f'({ir} of {len(self.rectangles)}). Exception will be reraised.'
                            self.logger.error(errmsg)
                            raise e
            else :
                for ir,rect,segmented_nifti_path,segmented_file_path in rects_to_run :
                    msg = f'Converting nnU-Net output for {rect.imagefile.name} ({ir} of {len(self.rectangles)})'
                    self.logger.debug(msg)
                    convert_nnunet_output(segmented_nifti_path,segmented_file_path)
            for ir,rect in enumerate(self.rectangles,start=1) :
                if self.__get_rect_nnunet_segmented_fp(rect).is_file() :
                    completed_files+=1
                    if self.__get_rect_nifti_fp(rect).is_file() :
                        self.__get_rect_nifti_fp(rect).unlink()
                    if self.__get_rect_segmented_nifti_fp(rect).is_file() :
                        self.__get_rect_segmented_nifti_fp(rect).unlink()
            if completed_files==len(self.rectangles) :
                shutil.rmtree(self.temp_dir)
                plans_file = self.__workingdir/'plans.pkl'
                if plans_file.is_file() :
                    plans_file.unlink()
                postproc_file = self.__workingdir/'postprocessing.json'
                if postproc_file.is_file() :
                    postproc_file.unlink()
                self.logger.info(f'All files segmented using nnU-Net with output in {self.__workingdir}')
            else :
                msg = f'{completed_files} of {len(self.rectangles)} files segmented using nnU-Net. '
                msg+= 'Rerun the same command to retry.'
                self.logger.info(msg)

    def __run_deepcell(self) :
        """
        Run nuclear segmentation using DeepCell's nuclear segmentation algorithm
        """
        self.logger.debug('Running nuclear segmentation with DeepCell....')
        if self.njobs is not None and self.njobs>1 :
            self.logger.warning(f'WARNING: njobs is {self.njobs} but DeepCell segmentation cannot be run in parallel.')
        app = NuclearSegmentation()
        completed_files = 0
        rects_to_run = []
        for ir,rect in enumerate(self.rectangles,start=1) :
            #skip any rectangles that already have segmentation output
            if self.__get_rect_deepcell_segmented_fp(rect).is_file() :
                msg = f'Skipping {rect.imagefile.name} ({ir} of {len(self.rectangles)}) '
                msg+= '(segmentation output already exists)'
                self.logger.debug(msg)
                completed_files+=1
                continue
            rects_to_run.append((ir,rect,self.__get_rect_deepcell_segmented_fp(rect)))
        try :
            for ir,rect,segmented_file_path in rects_to_run :
                with rect.using_image() as im :
                    msg = f'Running DeepCell segmentation for {rect.imagefile.name} '
                    msg+= f'({ir} of {len(self.rectangles)})'
                    self.logger.debug(msg)
                    run_deepcell_nuclear_segmentation(im,app,self.pscale,segmented_file_path)
            for rect in self.rectangles :
                if self.__get_rect_deepcell_segmented_fp(rect).is_file() :
                    completed_files+=1
        except Exception as e :
            raise e
        finally :
            if completed_files==len(self.rectangles) :
                self.logger.info(f'All files segmented using DeepCell with output in {self.__workingdir}')
            else :
                msg = f'{completed_files} of {len(self.rectangles)} files segmented using DeepCell. '
                msg+= 'Rerun the same command to retry.'
                self.logger.info(msg)

#################### FILE-SCOPE FUNCTIONS ####################

def main(args=None) :
    SegmentationSample.runfromargumentparser(args)

if __name__=='__main__' :
    main()
