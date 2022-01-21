#imports
import os
import numpy as np, SimpleITK as sitk
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
            segmented_file_path = self.__workingdir/f'{rect.imagefile.name[:-4]}_nnunet_nuclear_segmentation.npz'
            #skip any rectangles that already have segmentation input or output
            if nifti_file_path.is_file() or segmented_nifti_path.is_file() or segmented_file_path.is_file() :
                msg = f'\tSkipping writing NIfTI file for {rect.imagefile.name} ({ir} of {len(self.rectangles)})'
                if segmented_nifti_path.is_file() or segmented_file_path.is_file() :
                    msg+=' (segmentation output already exists)'
                elif nifti_file_path.is_file() :
                    msg+=' (NIfTI file already exists in temp. directory)'
                self.logger.debug(msg)
                continue
            with rect.using_image() as im :
                self.logger.debug(f'\tWriting NIfTI file for {rect.imagefile.name} ({ir} of {len(self.rectangles)})')
                img = im[:,:,np.newaxis]
                img = img.transpose(2,0,1)
                itk_img = sitk.GetImageFromArray(img)
                itk_img.SetSpacing([1,1,999])
                sitk.WriteImage(itk_img, str(nifti_file_path))
        #run the nnU-Net nuclear segmentation algorithm
        os.environ['nnUNet_raw_data_base'] = str(SEG_CONST.NNUNET_MODEL_TOP_DIR.resolve())
        os.environ['nnUNet_preprocessed'] = str(SEG_CONST.NNUNET_MODEL_TOP_DIR.resolve())
        os.environ['RESULTS_FOLDER'] = str(SEG_CONST.NNUNET_MODEL_TOP_DIR.resolve())
        nnunet_cmd = f'nnUNet_predict -i {str(temp_dir.resolve())} -o {str(self.__workingdir.resolve())} -t 500 -m 2d'
        try :
            os.system(nnunet_cmd)
        except Exception as e :
            errmsg = 'ERROR: some exception was raised while running nnU-Net. Output will be collected and then '
            errmsg = 'the exception will be reraised. See log lines above for what was completed.'
            self.logger.error(errmsg)
            raise e
        finally :
            completed_files = 0
            for rect in self.rectangles :
                nifti_file_path = temp_dir/f'{rect.imagefile.name[:-4]}_0000.nii.gz'
                segmented_nifti_path = self.__workingdir/f'{rect.imagefile.name[:-4]}.nii.gz'
                segmented_file_path = self.__workingdir/f'{rect.imagefile.name[:-4]}_nnunet_nuclear_segmentation.npz'
                if segmented_nifti_path.is_file() :
                    itk_read_img = sitk.ReadImage(str(segmented_nifti_path),imageIO='NiftiImageIO')
                    output_img = np.zeros((itk_read_img.GetHeight(),itk_read_img.GetWidth()),dtype=np.float32)
                    for ix in range(output_img.shape[1]) :
                        for iy in range(output_img.shape[0]) :
                            output_img[iy,ix] = itk_read_img.GetPixel((ix,iy,0))
                    output_img = output_img.astype(np.uint8)
                    np.savez_compressed(segmented_file_path,output_img)
                if segmented_file_path.is_file() :
                    completed_files+=1
                    nifti_file_path.unlink()
            if completed_files==len(self.rectangles) :
                self.logger.info('All files segmented using nnU-Net')
            else :
                msg = f'{completed_files} of {len(self.rectangles)} files segmented using nnU-Net. '
                msg+= 'Rerun the same command to retry.'
                self.logger.info(msg)

    def __run_deepcell(self) :
        pass

#################### FILE-SCOPE FUNCTIONS ####################

def main(args=None) :
    SegmentationSample.runfromargumentparser(args)

if __name__=='__main__' :
    main()
