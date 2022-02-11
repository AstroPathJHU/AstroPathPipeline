#imports
import os, shutil
from batchgenerators.utilities.file_and_folder_operations import join
from ...utilities.config import CONST as UNIV_CONST
from ...utilities.optionalimports import deepcell, nnunet
from ...shared.argumentparser import WorkingDirArgumentParser
from ...shared.sample import ReadRectanglesComponentTiffFromXML, WorkflowSample, ParallelSample
from .config import SEG_CONST
from .utilities import rebuild_model_files_if_necessary, write_nifti_file_for_rect_im
from .utilities import convert_nnunet_output, run_deepcell_nuclear_segmentation

#some constants
NNUNET_SEGMENT_FILE_APPEND = 'nnunet_nuclear_segmentation.npz'
DEEPCELL_SEGMENT_FILE_APPEND = 'deepcell_nuclear_segmentation.npz'

class SegmentationSampleBase(ReadRectanglesComponentTiffFromXML,SampleWithSegmentations,
                             WorkflowSample,ParallelSample,WorkingDirArgumentParser) :
    """
    Write out nuclear segmentation maps based on the DAPI layers of component tiffs for a single sample
    Algorithms available include pre-trained nnU-Net and DeepCell/mesmer models 
    """

    #################### PUBLIC FUNCTIONS ####################

    def __init__(self,*args,workingdir=None,**kwargs) :
        # only need to load the DAPI layers of the rectangles, so send that to the __init__
        if kwargs.get('layer') is not None :
            raise RuntimeError(f'ERROR: sample layer was set to {kwargs.get("layer")}')
        kwargs['layer'] = 1 
        super().__init__(*args,**kwargs)
        self.__workingdirarg = workingdir

    def inputfiles(self,**kwargs) :
        return [*super().inputfiles(**kwargs),
                *(r.imagefile for r in self.rectangles),
               ]

    def run(self,**kwargs) :
        if not self.__workingdir.is_dir() :
            self.__workingdir.mkdir(parents=True)
        self.runsegmentation(**kwargs)

    @methodtools.lru_cache()
    def runsegmentation(self, **kwargs): pass

    #################### PROPERTIES ####################

    @property
    def workflowkwargs(self) :
        return {
            **super().workflowkwargs,
            'workingdir':self.__workingdirarg,
            'algorithm':self.segmentationalgorithm(),
        }

    @property
    def __workingdir(self):
        #set the working directory path based on the algorithm being run (if it wasn't set by a command line arg)
        return self.output_dir(self.__workingdirarg,self.im3root,self.SlideID,self.segmentationalgorithm())

    #################### CLASS METHODS ####################

    @classmethod
    def output_dir(cls,workingdir,im3root,SlideID,algorithm) :
        #default output is im3folder/segmentation/algorithm
        outputdir = workingdir
        if outputdir is None :
            outputdir = im3root/SlideID/'im3'/SEG_CONST.SEGMENTATION_DIR_NAME/algorithm
        else :
            if outputdir.name!=SlideID :
                #put non-default output in a subdirectory named for the slide
                outputdir = outputdir/SlideID
        return outputdir

    @classmethod
    def getoutputfiles(cls,SlideID,im3root,informdataroot,workingdir,algorithm,**otherworkflowkwargs) :
        outputdir=cls.output_dir(workingdir,im3root,SlideID,algorithm)
        append = None
        if algorithm=='nnunet' :
            append = NNUNET_SEGMENT_FILE_APPEND
        elif algorithm=='deepcell' :
            append = DEEPCELL_SEGMENT_FILE_APPEND
        file_stems = [fp.name[:-len('_component_data.tif')] for fp in (informdataroot/SlideID/'inform_data'/'Component_Tiffs').glob('*_component_data.tif')]
        outputfiles = []
        for stem in file_stems :
            outputfiles.append(outputdir/f'{stem}_{append}')
        return outputfiles

    @classmethod
    def workflowdependencyclasses(cls, **kwargs):
        return super().workflowdependencyclasses(**kwargs)

class SegmentationSampleNNUNet(SegmentationSampleBase) :

    def __init__(self,*args,**kwargs) :
        super().__init__(*args,algorithm='nnunet',**kwargs)

    @classmethod
    def logmodule(cls) : 
        return "segmentationnnunet"

    def runsegmentation(self) :
        """
        Run nuclear segmentation using the pre-trained nnU-Net algorithm
        """
        predict_from_folder = nnunet.inference.predict.predict_from_folder
        default_plans_identifier = nnunet.paths.default_plans_identifier
        default_trainer = nnunet.paths.default_trainer

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
        self.logger.debug('Running nuclear segmentation with nnU-Net....')
        my_network_training_output_dir = str(SEG_CONST.NNUNET_MODEL_TOP_DIR.resolve()/'nnUNet')
        task_name = 'Task500_Pathology_DAPI'
        model_folder_name = join(my_network_training_output_dir, '2d', task_name, default_trainer + "__" +
                                 default_plans_identifier)
        try :
            predict_from_folder(model_folder_name, str(self.temp_dir.resolve()), str(self.__workingdir.resolve()), 
                                None, False, self.njobs, self.njobs, None, 0, 1, True, overwrite_existing=False, 
                                mode='normal', overwrite_all_in_gpu=None, mixed_precision=True,
                                step_size=0.5, checkpoint_name='model_final_checkpoint')
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


    #################### PRIVATE HELPER FUNCTIONS ####################

    def __get_rect_nifti_fp(self,rect) :
        return self.temp_dir/f'{rect.imagefile.name[:-4]}_0000.nii.gz'

    def __get_rect_segmented_nifti_fp(self,rect) :
        return self.__workingdir/f'{rect.imagefile.name[:-4]}.nii.gz'

    def __get_rect_nnunet_segmented_fp(self,rect) :
        seg_fn = f'{rect.file.rstrip(UNIV_CONST.IM3_EXT)}_{NNUNET_SEGMENT_FILE_APPEND}'
        return self.__workingdir/seg_fn

class SegmentationSampleDeepCell(SegmentationSampleBase) :
    def __init__(self,*args,**kwargs) :
        super().__init__(*args,**kwargs)

    @classmethod
    def segmentationalgorithm(cls):
      return "deepcell"

    @classmethod
    def logmodule(cls) : 
        return "segmentationdeepcell"

    def runsegmentation(self) :
        """
        Run nuclear segmentation using DeepCell's nuclear segmentation algorithm
        """
        NuclearSegmentation = deepcell.applications.NuclearSegmentation

        self.logger.debug('Running nuclear segmentation with DeepCell....')
        if self.njobs is not None and self.njobs>1 :
            self.logger.warning(f'WARNING: njobs is {self.njobs} but DeepCell segmentation cannot be run in parallel.')
        app = NuclearSegmentation()
        rects_to_run = []
        for ir,rect in enumerate(self.rectangles,start=1) :
            #skip any rectangles that already have segmentation output
            if self.__get_rect_deepcell_segmented_fp(rect).is_file() :
                msg = f'Skipping {rect.imagefile.name} ({ir} of {len(self.rectangles)}) '
                msg+= '(segmentation output already exists)'
                self.logger.debug(msg)
                continue
            rects_to_run.append((ir,rect,self.__get_rect_deepcell_segmented_fp(rect)))
        completed_files = 0
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

    #################### PRIVATE HELPER FUNCTIONS ####################

    def __get_rect_deepcell_segmented_fp(self,rect) :
        seg_fn = f'{rect.file.rstrip(UNIV_CONST.IM3_EXT)}_{DEEPCELL_SEGMENT_FILE_APPEND}'
        return self.__workingdir/seg_fn

#################### FILE-SCOPE FUNCTIONS ####################

def segmentationsamplennunet(args=None) :
    SegmentationSampleNNUNet.runfromargumentparser(args)

def segmentationsampledeepcell(args=None) :
    SegmentationSampleDeepCell.runfromargumentparser(args)
