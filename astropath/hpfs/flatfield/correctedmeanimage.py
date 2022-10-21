#imports
import numpy as np
from ...utilities.miscfileio import cd
from ...utilities.img_file_io import smooth_image_worker
from ...utilities.img_file_io import write_image_to_file
from .config import CONST
from .plotting import plot_image_layers, flatfield_image_pixel_intensity_plot, corrected_mean_image_PI_and_IV_plots
from .latexsummary import AppliedFlatfieldLatexSummary
from .meanimage import MeanImageBase, MeanImageComponentTiff, MeanImageIm3

class CorrectedMeanImageBase(MeanImageBase) :
    """
    Base class to work with a mean image that will be corrected with a set of flatfield correction factors
    """

    def __init__(self,*args,**kwargs) :
        super().__init__(*args,**kwargs)
        self.__flatfield_image = None
        self.__flatfield_image_err = None
        self.__corrected_mean_image = None
        self.__corrected_mean_image_err = None

    def apply_flatfield_corrections(self,flatfield) :
        """
        Correct the meanimage using the given flatfield object
        """
        flatfield_image = flatfield.flatfield_image
        flatfield_err = flatfield.flatfield_image_err
        if self.mean_image is None :
            raise RuntimeError('ERROR: mean image is not yet set but apply_flatfield_corrections was called!')
        elif self.mean_image.shape != flatfield_image.shape :
            errmsg = f'ERROR: shape mismatch in apply_flatfield_corrections, meanimage shape={self.mean_image.shape} '
            errmsg+= f'but flatfield shape={flatfield_image.shape}'
            raise ValueError(errmsg)
        self.__flatfield_image = flatfield_image
        self.__flatfield_image_err = flatfield_err
        self.__corrected_mean_image = self.mean_image/flatfield_image
        self.__corrected_mean_image_err = self.__corrected_mean_image*np.sqrt(np.power(self.__flatfield_image_err/self.__flatfield_image,2)*np.power(self.std_err_of_mean_image/self.mean_image,2))

    def write_output(self,samp,workingdirpath) :
        """
        Write out the relevant information for the corrected mean image to the given working directory
        """
        if not workingdirpath.is_dir() :
            workingdirpath.mkdir(parents=True)
        #save the flatfield image and its uncertainty
        self.logger.info('Saving flatfield image and its uncertainty....')
        with cd(workingdirpath) :
            write_image_to_file(self.__flatfield_image,'flatfield.bin')
            write_image_to_file(self.__flatfield_image_err,'flatfield_uncertainty.bin')
        #write out the mask stack
        if self.mask_stack is not None :
            self.logger.info('Writing out mask stack for mean image....')
            with cd(workingdirpath) :
                write_image_to_file(self.mask_stack,'corrected_mean_image_mask_stack.bin')
        #write out the corrected mean image and its uncertainty
        self.logger.info('Writing out the corrected mean image and its uncertainty')
        with cd(workingdirpath) :
            write_image_to_file(self.__corrected_mean_image,'corrected_mean_image.bin')
            write_image_to_file(self.__corrected_mean_image_err,'corrected_mean_image_uncertainty.bin')
        try :
            #make some plots of the image layers and the pixel intensities
            self.logger.info('Writing out image layer plots....')
            plotdir_path = workingdirpath / 'corrected_meanimage_plots'
            plotdir_path.mkdir(exist_ok=True)
            plot_image_layers(self.__flatfield_image,'flatfield',plotdir_path)
            plot_image_layers(self.__flatfield_image_err,'flatfield_uncertainty',plotdir_path)
            if self.mask_stack is not None :
                plot_image_layers(self.mask_stack,'corrected_mean_image_mask_stack',plotdir_path)
            plot_image_layers(self.mean_image,'mean_image',plotdir_path)
            plot_image_layers(self.std_err_of_mean_image,'mean_image_uncertainty',plotdir_path)
            plot_image_layers(self.__corrected_mean_image,'corrected_mean_image',plotdir_path)
            plot_image_layers(self.__corrected_mean_image_err,'corrected_mean_image_uncertainty',plotdir_path)
            self.logger.info('Building smoothed mean images pre/post correction....')
            sm_mean_image = smooth_image_worker(self.mean_image,CONST.FLATFIELD_WIDE_GAUSSIAN_FILTER_SIGMA,gpu=True)
            sm_corr_mean_image = smooth_image_worker(self.__corrected_mean_image,
                                                    CONST.FLATFIELD_WIDE_GAUSSIAN_FILTER_SIGMA,gpu=True)
            self.logger.info('Plotting pixel intensities....')
            flatfield_image_pixel_intensity_plot(samp,self.__flatfield_image,save_dirpath=plotdir_path)
            corrected_mean_image_PI_and_IV_plots(samp,sm_mean_image,sm_corr_mean_image,
                                                central_region=False,save_dirpath=plotdir_path)
            corrected_mean_image_PI_and_IV_plots(samp,sm_mean_image,sm_corr_mean_image,
                                                central_region=True,save_dirpath=plotdir_path)
            #make the summary PDF
            self.logger.info('Making the summary pdf....')
            latex_summary = AppliedFlatfieldLatexSummary(self.__flatfield_image,sm_mean_image,sm_corr_mean_image,
                                                        plotdir_path)
            latex_summary.build_tex_file()
            check = latex_summary.compile()
            if check!=0 :
                warnmsg = 'WARNING: failed while compiling flatfield summary LaTeX file into a PDF. '
                warnmsg+= f'tex file will be in {latex_summary.failed_compilation_tex_file_path}'
                self.logger.warning(warnmsg)
        except Exception as e :
            warnmsg = 'WARNING: failed to write out some optional plots (scripts will need to be run separately). '
            warnmsg+= f'Exception: {e}'
            self.logger.warning(warnmsg)

class CorrectedMeanImageComponentTiff(CorrectedMeanImageBase,MeanImageComponentTiff) :
    """
    Corrected mean image with a flatfield and mean image determined from component tiff files
    """
    pass

class CorrectedMeanImageIm3(CorrectedMeanImageBase,MeanImageIm3) :
    """
    Corrected mean image with a flatfield and mean image determined from raw IM3 files
    """
    pass
