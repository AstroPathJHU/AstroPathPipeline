#imports
import methodtools, os, shutil
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import join
from ...utilities.config import CONST as UNIV_CONST
from ...utilities.optionalimports import deepcell, nnunet
from ...shared.sample import ParallelSample, ReadRectanglesComponentAndIHCTiffFromXML
from ...shared.sample import SampleWithSegmentationFolder, WorkflowSample
from .config import SEG_CONST
from .utilities import rebuild_model_files_if_necessary, write_nifti_file_for_rect_im
from .utilities import convert_nnunet_output, run_deepcell_nuclear_segmentation, run_mesmer_segmentation

#some constants
NNUNET_SEGMENT_FILE_APPEND = 'nnunet_nuclear_segmentation.npz'
DEEPCELL_SEGMENT_FILE_APPEND = 'deepcell_nuclear_segmentation.npz'
MESMER_SEGMENT_FILE_APPEND = 'mesmer_segmentation.npz'
GROUP_SIZE = 48

class SegmentationSampleBase(ReadRectanglesComponentAndIHCTiffFromXML,SampleWithSegmentationFolder,
                             WorkflowSample,ParallelSample) :
    """
    Write out nuclear segmentation maps based on the DAPI layers of component tiffs for a single sample
    Algorithms available include pre-trained nnU-Net and DeepCell/mesmer models 
    """

    #################### PUBLIC FUNCTIONS ####################

    def __init__(self,*args,layercomponenttiff=1,**kwargs) :
        # only need to load the DAPI layers of the rectangles, so send that to the __init__
        if layercomponenttiff != 1 :
            raise RuntimeError(f'ERROR: sample layer was set to {kwargs.get("layer")}')
        super().__init__(*args,layercomponenttiff=layercomponenttiff,**kwargs)

    def inputfiles(self,**kwargs) :
        return [*super().inputfiles(**kwargs),
                *(r.componenttifffile for r in self.rectangles),
               ]

    def run(self,**kwargs) :
        if not self.segmentationfolder.is_dir() :
            self.segmentationfolder.mkdir(parents=True)
        self.runsegmentation(**kwargs)

    @methodtools.lru_cache()
    def runsegmentation(self, **kwargs): pass

    #################### CLASS METHODS ####################

    @classmethod
    def getoutputfiles(cls,SlideID,im3root,informdataroot,segmentationfolder,**otherworkflowkwargs) :
        outputdir=cls.segmentation_folder(segmentationfolder,im3root,SlideID)
        append = None
        if cls.segmentationalgorithm()=='nnunet' :
            append = NNUNET_SEGMENT_FILE_APPEND
        elif cls.segmentationalgorithm()=='deepcell' :
            append = DEEPCELL_SEGMENT_FILE_APPEND
        elif cls.segmentationalgorithm()=='mesmer' :
            append = MESMER_SEGMENT_FILE_APPEND
        file_stems = [fp.name[:-len('_component_data.tif')] 
                      for fp in (informdataroot/SlideID/'inform_data'/'Component_Tiffs').glob('*_component_data.tif')]
        outputfiles = []
        for stem in file_stems :
            outputfiles.append(outputdir/f'{stem}_{append}')
        return outputfiles

    @classmethod
    def workflowdependencyclasses(cls, **kwargs):
        return super().workflowdependencyclasses(**kwargs)

class SegmentationSampleNNUNet(SegmentationSampleBase) :

    def __init__(self,*args,**kwargs) :
        super().__init__(*args,**kwargs)

    @classmethod
    def segmentationalgorithm(cls) :
      return 'nnunet'

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
        self.temp_dir = self.segmentationfolder/'nnunet_nifti_input'
        if not self.temp_dir.is_dir() :
            self.temp_dir.mkdir(parents=True)
        #write a NIfTI file for every rectangle's DAPI layer for input to the algorithm
        self.logger.debug('Writing NIfTI files for nnU-Net input....')
        rects_to_run = []
        for ir,rect in enumerate(self.rectangles,start=1) :
            #skip any rectangles that already have segmentation input or output
            if ( self.__get_rect_nifti_fp(rect).is_file() or self.__get_rect_segmented_nifti_fp(rect).is_file() 
                or self.__get_rect_nnunet_segmented_fp(rect).is_file() ) :
                msg = f'Skipping writing NIfTI file for {rect.componenttifffile.name} ({ir} of {len(self.rectangles)})'
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
                    with rect.using_component_tiff() as im :
                        msg = f'Writing NIfTI file for {rect.componenttifffile.name} ({ir} of {len(self.rectangles)})'
                        self.logger.debug(msg)
                        proc_results[(ir,rect.componenttifffile.name)] = pool.apply_async(write_nifti_file_for_rect_im,
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
                with rect.using_component_tiff() as im :
                    msg = f'Writing NIfTI file for {rect.componenttifffile.name} ({ir} of {len(self.rectangles)})'
                    self.logger.debug(msg)
                    write_nifti_file_for_rect_im(im,nifti_file_path)
        #run the nnU-Net nuclear segmentation algorithm
        os.environ['RESULTS_FOLDER'] = str(SEG_CONST.NNUNET_MODEL_TOP_DIR.resolve())
        self.logger.debug('Running nuclear segmentation with nnU-Net....')
        my_network_training_output_dir = str(SEG_CONST.NNUNET_MODEL_TOP_DIR.resolve()/'nnUNet')
        task_name = 'Task500_Pathology_DAPI'
        model_folder_name = join(my_network_training_output_dir, '2d', task_name, default_trainer + "__" +
                                 default_plans_identifier)
        try :
            predict_from_folder(model_folder_name, str(self.temp_dir.resolve()), str(self.segmentationfolder.resolve()), 
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
                        msg = f'Converting nnU-Net output for {rect.componenttifffile.name} '
                        msg+= f'({ir} of {len(self.rectangles)})'
                        self.logger.debug(msg)
                        proc_results[(ir,rect.componenttifffile.name)] = pool.apply_async(convert_nnunet_output,
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
                    msg = f'Converting nnU-Net output for {rect.componenttifffile.name} '
                    msg+= f'({ir} of {len(self.rectangles)})'
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
                plans_file = self.segmentationfolder/'plans.pkl'
                if plans_file.is_file() :
                    plans_file.unlink()
                postproc_file = self.segmentationfolder/'postprocessing.json'
                if postproc_file.is_file() :
                    postproc_file.unlink()
                self.logger.info(f'All files segmented using nnU-Net with output in {self.segmentationfolder}')
            else :
                msg = f'{completed_files} of {len(self.rectangles)} files segmented using nnU-Net. '
                msg+= 'Rerun the same command to retry.'
                self.logger.info(msg)

    def __get_rect_nifti_fp(self,rect) :
        return self.temp_dir/f'{rect.componenttifffile.name[:-4]}_0000.nii.gz'

    def __get_rect_segmented_nifti_fp(self,rect) :
        return self.segmentationfolder/f'{rect.componenttifffile.name[:-4]}.nii.gz'

    def __get_rect_nnunet_segmented_fp(self,rect) :
        seg_fn = f'{rect.file.rstrip(UNIV_CONST.IM3_EXT)}_{NNUNET_SEGMENT_FILE_APPEND}'
        return self.segmentationfolder/seg_fn

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
            if self.__get_rect_segmented_fp(rect).is_file() :
                msg = f'Skipping {rect.componenttifffile.name} ({ir} of {len(self.rectangles)}) '
                msg+= '(segmentation output already exists)'
                self.logger.debug(msg)
                continue
            rects_to_run.append((ir,rect,self.__get_rect_segmented_fp(rect)))
        completed_files = 0
        try :
            deepcell_batch_images = []
            deepcell_batch_segmented_filepaths = []
            for realir,(ir,rect,segmented_file_path) in enumerate(rects_to_run,start=1) :
                #add to the batch
                msg = f'Adding {rect.componenttifffile.name} ({ir} of {len(self.rectangles)}) '
                msg+= 'to the next group of images....'
                self.logger.debug(msg)
                with rect.using_component_tiff() as im :
                    dapi_layer = im
                im_for_deepcell = np.expand_dims(dapi_layer,axis=-1)
                deepcell_batch_images.append(im_for_deepcell)
                deepcell_batch_segmented_filepaths.append(segmented_file_path)
                #run segmentations for a whole batch
                if (len(deepcell_batch_images)>=GROUP_SIZE) or (realir==len(rects_to_run)) :
                    msg = f'Running DeepCell segmentation for the current group of {len(deepcell_batch_images)} images'
                    self.logger.debug(msg)
                    run_deepcell_nuclear_segmentation(np.array(deepcell_batch_images),
                                                      app,
                                                      self.pscale,
                                                      deepcell_batch_segmented_filepaths)
                    deepcell_batch_images = []
                    deepcell_batch_segmented_filepaths = []
            for rect in self.rectangles :
                if self.__get_rect_segmented_fp(rect).is_file() :
                    completed_files+=1
        except Exception as e :
            raise e
        finally :
            if completed_files==len(self.rectangles) :
                self.logger.info(f'All files segmented using DeepCell with output in {self.segmentationfolder}')
            else :
                msg = f'{completed_files} of {len(self.rectangles)} files segmented using DeepCell. '
                msg+= 'Rerun the same command to retry.'
                self.logger.info(msg)

    def __get_rect_segmented_fp(self,rect) :
        seg_fn = f'{rect.file.rstrip(UNIV_CONST.IM3_EXT)}_{DEEPCELL_SEGMENT_FILE_APPEND}'
        return self.segmentationfolder/seg_fn

class SegmentationSampleMesmer(SegmentationSampleBase) :
    
    def __init__(self,*args,**kwargs) :
        super().__init__(*args,**kwargs)

    @classmethod
    def segmentationalgorithm(cls):
      return "mesmer"

    @classmethod
    def logmodule(cls) : 
        return "segmentationmesmer"

    def runsegmentation(self) :
        """
        Run whole-cell segmentation using the Mesmer segmentation algorithm
        """
        Mesmer = deepcell.applications.Mesmer
        pca_vec_to_dot = np.expand_dims(SEG_CONST.IHC_PCA_BLACK_COMPONENT,0).T
        self.logger.debug('Running whole-cell and nuclear segmentation with Mesmer....')
        if self.njobs is not None and self.njobs>1 :
            self.logger.warning(f'WARNING: njobs is {self.njobs} but Mesmer segmentation cannot be run in parallel.')
        app = Mesmer()
        rects_to_run = []
        for ir,rect in enumerate(self.rectangles,start=1) :
            #skip any rectangles that already have segmentation output
            if self.__get_rect_segmented_fp(rect).is_file() :
                msg = f'Skipping {rect.ihctifffile.name} ({ir} of {len(self.rectangles)}) '
                msg+= '(segmentation output already exists)'
                self.logger.debug(msg)
                continue
            rects_to_run.append((ir,rect,self.__get_rect_segmented_fp(rect)))
        completed_files = 0
        try :
            mesmer_batch_images = []
            mesmer_batch_segmented_filepaths = []
            for realir,(ir,rect,segmented_file_path) in enumerate(rects_to_run,start=1) :
                #add to the batch
                msg = f'Adding {rect.ihctifffile.name} ({ir} of {len(self.rectangles)}) to the next group of images....'
                self.logger.debug(msg)
                with rect.using_component_tiff() as im :
                    dapi_layer = im
                with rect.using_ihc_tiff() as im :
                    im_membrane = -1.0*(np.dot(im,pca_vec_to_dot))[:,:,0]
                    membrane_layer = (im_membrane-np.min(im_membrane))/SEG_CONST.IHC_MEMBRANE_LAYER_NORM
                im_for_mesmer = np.array([dapi_layer,membrane_layer]).transpose(1,2,0)
                mesmer_batch_images.append(im_for_mesmer)
                mesmer_batch_segmented_filepaths.append(segmented_file_path)
                #run segmentations for a whole batch
                if (len(mesmer_batch_images)>=GROUP_SIZE) or (realir==len(rects_to_run)) :
                    msg = f'Running Mesmer segmentation for the current group of {len(mesmer_batch_images)} images'
                    self.logger.debug(msg)
                    run_mesmer_segmentation(np.array(mesmer_batch_images),
                                            app,
                                            self.pscale,
                                            mesmer_batch_segmented_filepaths)
                    mesmer_batch_images = []
                    mesmer_batch_segmented_filepaths = []
            for rect in self.rectangles :
                if self.__get_rect_segmented_fp(rect).is_file() :
                    completed_files+=1
        except Exception as e :
            raise e
        finally :
            if completed_files==len(self.rectangles) :
                self.logger.info(f'All files segmented using Mesmer with output in {self.segmentationfolder}')
            else :
                msg = f'{completed_files} of {len(self.rectangles)} files segmented using Mesmer. '
                msg+= 'Rerun the same command to retry.'
                self.logger.info(msg)
        pass

    #################### PRIVATE HELPER FUNCTIONS ####################

    def __get_rect_segmented_fp(self,rect) :
        seg_fn = f'{rect.file.rstrip(UNIV_CONST.IM3_EXT)}_{MESMER_SEGMENT_FILE_APPEND}'
        return self.segmentationfolder/seg_fn

#################### FILE-SCOPE FUNCTIONS ####################

def segmentationsamplennunet(args=None) :
    SegmentationSampleNNUNet.runfromargumentparser(args)

def segmentationsampledeepcell(args=None) :
    SegmentationSampleDeepCell.runfromargumentparser(args)

def segmentationsamplemesmer(args=None) :
    SegmentationSampleMesmer.runfromargumentparser(args)
