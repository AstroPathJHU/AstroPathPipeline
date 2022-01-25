#imports
import os
import numpy as np, SimpleITK as sitk
from skimage.segmentation import find_boundaries
from deepcell.applications import NuclearSegmentation
from ...utilities.config import CONST as UNIV_CONST
from ...shared.argumentparser import WorkingDirArgumentParser
from ...shared.sample import ReadRectanglesComponentTiffFromXML, WorkflowSample, ParallelSample
from .config import SEG_CONST
from .utilities import rebuild_model_files_if_necessary

class SegmentationSample(ReadRectanglesComponentTiffFromXML,WorkflowSample,ParallelSample,WorkingDirArgumentParser) :
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
        self.__workingdir = workingdir
        if self.__workingdir is None :
            self.__workingdir = self.im3folder/SEG_CONST.SEGMENTATION_DIR_NAME/self.__algorithm
            if not self.__workingdir.is_dir() :
                self.__workingdir.mkdir(parents=True)

    def inputfiles(self,**kwargs) :
        return [*super().inputfiles(**kwargs),
                *(r.imagefile for r in self.rectangles),
               ]

    def run(self) :
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
            'workingdir':self.__workingdir
        }

    #################### CLASS METHODS ####################

    @classmethod
    def makeargumentparser(cls):
        p = super().makeargumentparser()
        p.add_argument('--algorithm', choices=['nnunet','deepcell'], default='nnunet',
                       help='''Which segmentation algorithm to apply''')
        return p

    @classmethod
    def initkwargsfromargumentparser(cls, parsed_args_dict):
        return {**super().initkwargsfromargumentparser(parsed_args_dict),
                'algorithm':parsed_args_dict.pop('algorithm')
            }

    @classmethod
    def getoutputfiles(cls,SlideID,root,workingdir,**otherworkflowkwargs) :
        pass

    @classmethod
    def logmodule(cls) : 
        return "segmentation"

    @classmethod
    def workflowdependencyclasses(cls, **kwargs):
        return super().workflowdependencyclasses(**kwargs)

    #################### PRIVATE HELPER FUNCTIONS ####################

    def __run_nnunet(self) :
        """
        Run nuclear segmentation using the pre-trained nnU-Net algorithm
        """
        #make sure that the necessary model files exist
        rebuild_model_files_if_necessary()
        #create the temporary directory that will hold the NIfTI files
        temp_dir = self.__workingdir/'nnunet_nifti_input'
        if not temp_dir.is_dir() :
            temp_dir.mkdir(parents=True)
        #write a NIfTI file for every rectangle's DAPI layer for input to the algorithm
        self.logger.debug('Writing NIfTI files for nnU-Net input....')
        for ir,rect in enumerate(self.rectangles,start=1) :
            nifti_file_path = temp_dir/f'{rect.imagefile.name[:-4]}_0000.nii.gz'
            segmented_nifti_path = self.__workingdir/f'{rect.imagefile.name[:-4]}.nii.gz'
            seg_fn = f'{rect.file.rstrip(UNIV_CONST.IM3_EXT)}_nnunet_nuclear_segmentation.npz'
            segmented_file_path = self.__workingdir/seg_fn
            #skip any rectangles that already have segmentation input or output
            if nifti_file_path.is_file() or segmented_nifti_path.is_file() or segmented_file_path.is_file() :
                msg = f'Skipping writing NIfTI file for {rect.imagefile.name} ({ir} of {len(self.rectangles)})'
                if segmented_nifti_path.is_file() or segmented_file_path.is_file() :
                    msg+=' (segmentation output already exists)'
                elif nifti_file_path.is_file() :
                    msg+=' (NIfTI file already exists in temp. directory)'
                self.logger.debug(msg)
                continue
            with rect.using_image() as im :
                self.logger.debug(f'Writing NIfTI file for {rect.imagefile.name} ({ir} of {len(self.rectangles)})')
                img = im[:,:,np.newaxis]
                img = img.transpose(2,0,1)
                itk_img = sitk.GetImageFromArray(img)
                itk_img.SetSpacing([1,1,999])
                sitk.WriteImage(itk_img, str(nifti_file_path))
        #run the nnU-Net nuclear segmentation algorithm
        os.environ['RESULTS_FOLDER'] = str(SEG_CONST.NNUNET_MODEL_TOP_DIR.resolve())
        nnunet_cmd = f'nnUNet_predict -i {str(temp_dir.resolve())} -o {str(self.__workingdir.resolve())}'
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
            for ir,rect in enumerate(self.rectangles,start=1) :
                nifti_file_path = temp_dir/f'{rect.imagefile.name[:-4]}_0000.nii.gz'
                segmented_nifti_path = self.__workingdir/f'{rect.imagefile.name[:-4]}.nii.gz'
                seg_fn = f'{rect.file.rstrip(UNIV_CONST.IM3_EXT)}_nnunet_nuclear_segmentation.npz'
                segmented_file_path = self.__workingdir/seg_fn
                if segmented_nifti_path.is_file() :
                    msg = f'Converting nnU-Net output for {rect.imagefile.name} ({ir} of {len(self.rectangles)})'
                    self.logger.debug(msg)
                    itk_read_img = sitk.ReadImage(str(segmented_nifti_path),imageIO='NiftiImageIO')
                    output_img = np.zeros((itk_read_img.GetHeight(),itk_read_img.GetWidth()),dtype=np.float32)
                    for ix in range(output_img.shape[1]) :
                        for iy in range(output_img.shape[0]) :
                            output_img[iy,ix] = itk_read_img.GetPixel((ix,iy,0))
                    output_img[output_img>1] = 2
                    output_img = output_img.astype(np.uint8)
                    np.savez_compressed(segmented_file_path,output_img)
                if segmented_file_path.is_file() :
                    completed_files+=1
                    nifti_file_path.unlink()
                    segmented_nifti_path.unlink()
            if completed_files==len(self.rectangles) :
                self.logger.info('All files segmented using nnU-Net')
            else :
                msg = f'{completed_files} of {len(self.rectangles)} files segmented using nnU-Net. '
                msg+= 'Rerun the same command to retry.'
                self.logger.info(msg)

    def __run_deepcell(self) :
        """
        Run nuclear segmentation using DeepCell's nuclear segmentation algorithm
        """
        self.logger.debug('Running nuclear segmentation with DeepCell....')
        app = NuclearSegmentation()
        completed_files = 0
        try :
            for ir,rect in enumerate(self.rectangles[:5],start=1) :
                seg_fn = f'{rect.file.rstrip(UNIV_CONST.IM3_EXT)}_deepcell_nuclear_segmentation.npz'
                segmented_file_path = self.__workingdir/seg_fn
                #skip any rectangles that already have segmentation output
                if segmented_file_path.is_file() :
                    msg = f'Skipping {rect.imagefile.name} ({ir} of {len(self.rectangles)}) '
                    msg+= '(segmentation output already exists)'
                    self.logger.debug(msg)
                    completed_files+=1
                    continue
                with rect.using_image() as im :
                    self.logger.debug(f'Segmenting {rect.imagefile.name} ({ir} of {len(self.rectangles)})')
                    img = np.expand_dims(im,axis=-1)
                img = np.expand_dims(img,axis=0)
                labeled_img = app.predict(img,image_mpp=1./self.pscale)
                labeled_img = labeled_img[0,:,:,0]
                boundaries = find_boundaries(labeled_img)
                output_img = np.zeros(labeled_img.shape,dtype=np.uint8)
                output_img[labeled_img!=0] = 2
                output_img[boundaries] = 1
                np.savez_compressed(segmented_file_path,output_img)
                if segmented_file_path.is_file() :
                    completed_files+=1
        except Exception as e :
            raise e
        finally :
            if completed_files==len(self.rectangles) :
                self.logger.info('All files segmented using DeepCell')
            else :
                msg = f'{completed_files} of {len(self.rectangles)} files segmented using DeepCell. '
                msg+= 'Rerun the same command to retry.'
                self.logger.info(msg)

#################### FILE-SCOPE FUNCTIONS ####################

def main(args=None) :
    SegmentationSample.runfromargumentparser(args)

if __name__=='__main__' :
    main()
