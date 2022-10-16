#imports
import os, shutil
from batchgenerators.utilities.file_and_folder_operations import join
from ...utilities.optionalimports import nnunet
from ...utilities.config import CONST as UNIV_CONST
from .config import SEG_CONST
from .utilities import rebuild_model_files_if_necessary, write_nifti_file_for_rect_im, convert_nnunet_output
from .segmentationsample import SegmentationSampleDAPIComponentTiff

class SegmentationSampleNNUNet(SegmentationSampleDAPIComponentTiff) :

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
        seg_fn = f'{rect.file.stem}_{SEG_CONST.NNUNET_SEGMENT_FILE_APPEND}'
        return self.segmentationfolder/seg_fn

def main(args=None) :
    SegmentationSampleNNUNet.runfromargumentparser(args)

if __name__=='__main__' :
    main()
