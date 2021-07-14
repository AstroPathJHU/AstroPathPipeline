#imports
from .utilities import calculate_statistics_for_image
from .config import CONST
from ...shared.latexsummary import LatexSummaryWithPlotdir, LatexSummaryForSlideWithPlotdir, LatexDataTable
import numpy as np

class ThresholdingLatexSummary(LatexSummaryForSlideWithPlotdir) :
    """
    Class to make the background thresholding summary file for a single slide
    """

    def __init__(self,slideID,threshold_plot_dirpath) :
        """
        threshold_plot_dirpath = path to the directory with all of the thresholding summary plots in it
        """
        super().__init__('Background Thresholding Summary',CONST.THRESHOLDING_SUMMARY_PDF_FILENAME,slideID,threshold_plot_dirpath)

    @property
    def sections(self) :
        return super().sections+[self.rect_locations,self.thresholds_by_layer,self.individual_layer_thresholding_plots]

    @property
    def rect_locations(self) :
        lines = []
        lines.append(self.section_start('Locations of images used'))
        figlabel = 'fig:rectangle_locations'
        l = 'All High Power Fields (HPFs) located on the edges of the tissue were used to find the best overall background thresholds in every image layer. '
        l+= f'Figure~\\ref{{{figlabel}}} shows the locations of the tissue edge HPFs in red and the locations of the ``bulk" (non-edge) HPFs in blue.'
        lines.append(l)
        lines.append('\n')
        caption = f'Locations of HPFs in slide {self.slideID_tex}, with tissue edge HPFs shown in red and non-edge HPFs shown in blue.'
        lines+=self.image_figure_lines(f'{self.plot_dirpath_tex}/{self.slideID}_rectangle_locations',caption,figlabel)
        return lines

    @property
    def thresholds_by_layer(self) :
        lines = []
        lines.append(self.section_start('Distributions of thresholds found by layer'))
        figlabel = 'fig:thresholds_by_layer'
        l = f'Figure~\\ref{{{figlabel}}} shows the 10th and 90th percentiles in the set of all individual HPF thresholds found in each image layer, '
        l+= 'as well as the final overall chosen thresholds, as a function of image layer, in both counts and counts/ms.'
        lines.append(l)
        lines.append('\n')
        caption = '10th and 90th percentiles (in red and blue, respectively) of the entire set of individual HPF thresholds found '
        caption+= 'in each image layer in raw counts (upper) and in counts/ms (lower). Also shown in black are the overall optimal thresholds chosen.'
        lines+=self.image_figure_lines(f'{self.plot_dirpath_tex}/{self.slideID}_background_thresholds_by_layer',caption,figlabel)
        return lines

    @property
    def individual_layer_thresholding_plots(self) :
        all_fns = []
        for fn in self.plot_dirpath.glob('layer_*_background_threshold_plots.png') :
            all_fns.append(fn.name)
        figure_lines = []
        for ifn,fn in enumerate(sorted(all_fns,key=lambda x:int(str(x).split('_')[1]))) :
            figlabel=f'fig:layer_{ifn+1}_threshold_plots'
            figure_lines+=self.image_figure_lines(f'{self.plot_dirpath_tex}/{fn}',label=figlabel,widths=0.9)
            if ifn==1 or (ifn>1 and (ifn-1)%3==0) :
                figure_lines.append('\\clearpage\n\n')
        lines = []
        lines.append(self.section_start('Detailed thresholding results for each image layer'))
        l = f'Figures~\\ref{{fig:layer_1_threshold_plots}}-\\ref{{fig:layer_{len(all_fns)}_threshold_plots}} show detailed views of the thresholding results '
        l+= 'in each image layer. The left columns in those figures show histograms of all individual HPF thresholds found for a given image layer, along with '
        l+= 'the means and medians of the distributions. The right columns in those figures show pixel intensity histograms for the same image layer, '
        l+= 'on a log-log scale, summed over all tissue edge HPFs, with the signal and background pixels shown in different colors.'
        lines.append(l)
        lines.append('\n')
        lines+=figure_lines
        return lines

class MaskingLatexSummary(LatexSummaryForSlideWithPlotdir) :
    """
    Class to make the blur and saturation masking summary file for a single slide
    """

    def __init__(self,slideID,masking_plot_dirpath) :
        """
        masking_plot_dirpath = path to the directory with all of the masking plots in it
        """
        super().__init__('Image Masking Summary',CONST.MASKING_SUMMARY_PDF_FILENAME,
                         slideID,masking_plot_dirpath,['*_masking_plots.png','*_flagged_hpf_locations.png'])

    @property
    def sections(self) :
        return super().sections+[self.flagged_hpf_locations,self.reproduced_plots]

    @property
    def flagged_hpf_locations(self) :
        lines = []
        lines.append(self.section_start('Flagged HPF locations'))
        figlabel = 'fig:flagged_hpf_locations'
        lines.append(f'Figure~\\ref{{{figlabel}}} shows the locations of every HPF in {self.slideID_tex}, color-coded by their reason for being flagged.')
        lines.append('\n')
        caption = f'Locations of all HPFs in {self.slideID_tex}. Those shown in gray did not have any blur or saturation flagged in them. '
        caption+= 'Those shown in blue had some region(s) flagged due to blur (either dust or folded tissue). Those shown in gold had some '
        caption+= 'region(s) flagged due to saturation in at least one layer group. Those shown in green had some region(s) flagged due to '
        caption+= 'both blur and saturation.'
        lines+=self.image_figure_lines(f'{self.plot_dirpath_tex}/{self.slideID}_flagged_hpf_locations',caption,figlabel)
        return lines

    @property
    def reproduced_plots(self) :
        all_fns = []
        for fn in self.plot_dirpath.glob('*_masking_plots.png') :
            all_fns.append(fn.name)
        lines = []
        lines.append(self.section_start('Example masking plots'))
        lines.append('\n')
        l = f'Figures~\\ref{{fig:first_masking_plot}}-\\ref{{fig:last_masking_plot}} below show examples of how the image masking proceeded for {len(all_fns)} '
        l+= f'individual images in {self.slideID_tex}. The examples shown are for the images in the sample with the largest numbers of pixels flagged due to '
        l+= f'blur and/or saturation.'
        lines.append(l+'\n\n')
        l = 'Every row (except the last) in each plot shows the same information, for the different broadband filter groups. The leftmost column shows the raw '
        l+= 'brightest image layer in the layer group. The second column from the left shows a grayscale overlay of that same brightest image layer with the '
        l+= 'tissue fold mask found for that independent layer group. In these second column plots, anything shown in red was flagged in the layer group but not '
        l+= 'in the overall final mask, anything shown in yellow was flagged in the layer group as well as the overall final mask, and anything shown in green '
        l+= 'was not flagged in the layer group but was flagged in the overall mask. Everything else is shown in grayscale. Note that '
        l+= '``flagged in the layer group" refers to being flagged as folded tissue, so regions that are flagged due to dust or saturation in the final mask '
        l+= 'will tend to show up green in these second column plots.'
        lines.append(l+'\n\n')
        l = "The third column shows a histogram of the exposure times in the layer group for all images in the slide, with the example image's exposure time "
        l+= 'marked with a red line. This is helpful to compare the exposure times of any images where saturation is flagged to those in the rest of the sample. '
        l+= 'The fourth and fifth columns show the normalized Laplacian variance of the brightest image layer in the layer group, and a histogram thereof, '
        l+= 'respectively. The final rightmost column shows the stack of individual image layer tissue fold masks in each layer group, and states where the cut '
        l+= 'on the number of layers required was.'
        lines.append(l+'\n\n')
        full_mask_fn_tex = CONST.BLUR_AND_SATURATION_MASK_FILE_NAME_STEM.replace("_","\\_")
        labelled_mask_regions_fn_tex = CONST.LABELLED_MASK_REGIONS_CSV_FILENAME.replace("_","\\_")
        l = f'The bottom row in each plot shows the layers of the compressed ``{full_mask_fn_tex}" file, with the '
        l+= f'numerical values pictured corresponding to entries in the ``{labelled_mask_regions_fn_tex}" file for the sample.'
        lines.append(l+'\n\n')
        for ifn,fn in enumerate(all_fns) :
            figlabel=None
            if ifn==0 :
                figlabel = 'fig:first_masking_plot'
            elif ifn==len(all_fns)-1 :
                figlabel = 'fig:last_masking_plot'
            img_key = fn.rstrip("_masking_plots.png").replace("_","\\_")
            caption = f'Masking plots for {img_key}'
            width = 1.0
            lines+=self.image_figure_lines(f'{self.plot_dirpath_tex}/{fn}',caption,figlabel,width)
            lines.append('\\clearpage\n\n')
        return lines

class MeanImageLatexSummary(LatexSummaryForSlideWithPlotdir) :
    """
    Class to make the meanimage summary file for a single slide
    """

    def __init__(self,slideID,image_layer_plot_dirpath) :
        """
        image_layer_plot_dirpath = path to the directory with all of the image layer plots in it
        """
        super().__init__('Mean Image Summary',CONST.MEANIMAGE_SUMMARY_PDF_FILENAME,slideID,image_layer_plot_dirpath)

    @property
    def sections(self) :
        sections = super().sections+[self.mean_image_plots,self.std_err_mean_image_plots]
        n_mask_stack_plots = 0
        for fn in self.plot_dirpath.glob(f'{self.slideID}-{CONST.MASK_STACK_BIN_FILE_NAME_STEM.rstrip(".bin")}_layer_*.png') :
            n_mask_stack_plots+=1
        if n_mask_stack_plots>0 :
            sections.append(self.mask_stack_plots)
        return sections

    @property
    def mean_image_plots(self) :
        lines = []
        lines.append(self.section_start('Layers of the mean image'))
        figlabel = 'fig:mean_image_layers'
        l = f'Figure~\\ref{{{figlabel}}} shows all individual layers of the computed mean image for {self.slideID_tex}. '
        l+= 'The units in these plots are units of intensity in average counts/ms.'
        lines.append(l)
        lines.append('\n')
        pattern = f'{self.slideID}-{CONST.MEAN_IMAGE_BIN_FILE_NAME_STEM.rstrip(".bin")}_layer_*.png'
        caption = f'All layers of the mean image computed for {self.slideID_tex}'
        lines+=self.image_layer_grid_plot_tex_lines(pattern,caption,figlabel)
        return lines

    @property
    def std_err_mean_image_plots(self) :
        lines = []
        lines.append(self.section_start('Layers of the standard error on the mean image'))
        figlabel = 'fig:mean_image_std_err_layers'
        l = f'Figure~\\ref{{{figlabel}}} shows the standard error of the {self.slideID_tex} mean image in all individual image layers. '
        l+= 'The units in these plots are also units of intensity in average counts/ms; they represent the uncertainties on the mean image layers shown in '
        l+= 'Figure~\\ref{fig:mean_image_layers}. Any overly bright regions that were not masked out will be especially apparent in these plots.'
        lines.append(l)
        lines.append('\n')
        pattern = f'{self.slideID}-{CONST.STD_ERR_OF_MEAN_IMAGE_BIN_FILE_NAME_STEM.rstrip(".bin")}_layer_*.png'
        caption = f'The standard error of the mean image for {self.slideID_tex} in all layers'
        lines+=self.image_layer_grid_plot_tex_lines(pattern,caption,figlabel)
        return lines

    @property
    def mask_stack_plots(self) :
        lines = []
        lines.append(self.section_start('Layers of the mask stack'))
        figlabel = 'fig:mask_stack_layers'
        l = f'Figure~\\ref{{{figlabel}}} shows every layer of the mask stack for {self.slideID_tex}. The units in these plots are the number of '
        l+= 'individual images contributing to the mean image at every location. They should be identical unless one or more image layer groups exhibited a '
        l+= 'great deal of saturation that was masked out. Referencing the general number of images stacked to find the mean illumination of each pixel helps '
        l+= 'contextualize the results above.'
        lines.append(l)
        lines.append('\n')
        pattern = f'{self.slideID}-{CONST.MASK_STACK_BIN_FILE_NAME_STEM.rstrip(".bin")}_layer_*.png'
        caption = f'The stack of all masks for images used to compute the {self.slideID_tex} mean image in all layers'
        lines+=self.image_layer_grid_plot_tex_lines(pattern,caption,figlabel)
        return lines

class FlatfieldLatexSummary(LatexSummaryWithPlotdir) :
    """
    Class to make a LatexSummary about a batch's flatfield model
    """

    def __init__(self,flatfield_image,plot_dirpath,batchID=None) :
        """
        flatfield_image = the actual flatfield image that was created (used to make a data table of the intensity ranges)
        plot_dirpath = path to the directory that has all the individual .png plots in it
        batchID = the batchID for the model in question (used in figure/filenames, optional)
        """
        self.__flatfield_image = flatfield_image
        self.__batchID = batchID
        title = 'Flatfield Summary'
        filename = CONST.FLATFIELD_SUMMARY_PDF_FILENAME_STEM
        if self.__batchID is not None :
            title+=f' for Batch {self.__batchID:02d}'
            filename+=f'_BatchID_{self.__batchID:02d}'
        filename+='.pdf'
        super().__init__(title,filename,plot_dirpath)

    @property
    def sections(self) :
        return super().sections+[self.pixel_intensities_and_data_table,self.flatfield_layers,self.flatfield_uncertainty_layers,self.mask_stack_layers]

    @property
    def pixel_intensities_and_data_table(self) :
        o_max, o_min, o_spread, o_stddev, c_max, c_min, c_spread, c_stddev = calculate_statistics_for_image(self.__flatfield_image)
        lines = []
        lines.append(self.section_start('Flatfield Correction Factors'))
        lines.append('\n')
        figlabel = 'fig:flatflield_pixel_intensities'
        l = f'Figure~\\ref{{{figlabel}}} shows the 5th and 95th percentile, as well as the standard deviation, of the pixel-by-pixel '
        l+= f'correction factors in each layer of the flatfield model'
        if self.__batchID is not None :
            l+=f' created for batch {self.__batchID:02d}'
        l+='. The red lines and green shaded areas show the values calculated over the entire area of the correction image, and the blue lines and yellow '
        l+='shading show the values calculated considering only the central 64\\% ``primary region" of the correction image.'
        lines.append(l)
        lines.append('\n')
        path = f'{self.plot_dirpath_tex}/flatfield'
        if self.__batchID is not None :
            path+=f'_BatchID_{self.__batchID:02d}'
        path+='_pixel_intensities'
        caption = '5th and 95th percentile and standard deviation of flatfield correction factors in each image layer. Red lines and green '
        caption+= 'shaded areas show statistics calculated using the entire area of each image layer. Blue lines and yellow shading show '
        caption+= 'statistics calculated in the central ``primary region" of each image layer.'
        lines+=self.image_figure_lines(path,caption,figlabel,widths=1.0)
        lines.append('\n')
        tablabel = 'tab:average_correction_factors'
        l = f'Table~\\ref{{{tablabel}}} lists the overall largest and smallest correction factors in any layer of the entire image and its '
        l+= 'primary region. It also denotes the average, over all image layers, of the 5th-95th percentile spread and standard deviation of the correction '
        l+= 'factors in the entire image and its primary region.'
        lines.append(l)
        lines.append('\n')
        caption = 'Summarizing statistics for the flatfield correction model. The central and rightmost columns list values calculated using '
        caption+= 'the entire area of the image and the central 64\\% ``primary region" of the image, respectively. From the upper to lower '
        caption+= 'row the values listed are the maximum and minimum correction factors in any image layer, the average over all image layers '
        caption+= 'of the spread from the 5th-95th percentile correction factors, and the average over all layers of the standard deviation '
        caption+= 'of the correction factors.'
        datatable = LatexDataTable(caption,tablabel)
        heading = 'Statistic & Whole image area & Central 64\\% ``primary region"'
        rows = [f'Maximum correction factor & {o_max:.05f} & {c_max:.05f}',
                f'Minimum correction factor & {o_min:.05f} & {c_min:.05f}',
                f'5th-95th pctile spread, avg. over all layers & {o_spread:.05f} & {c_spread:.05f}',
                f'Standard deviation, avg. over all layers & {o_stddev:.05f} & {c_stddev:.05f}'
               ]
        datatable.add_tabular(heading,rows)
        lines+=datatable.get_table_lines()
        return lines

    @property
    def flatfield_layers(self) :
        lines = []
        lines.append('\\section{Layers of the flatfield correction image}\n')
        figlabel = 'fig:flatfield_image_layers'
        lines.append(f'Figure~\\ref{{{figlabel}}} shows the flatfield correction factors found for every image layer.\n')
        lines.append('\n')
        pattern = f'{CONST.FLATFIELD_DIRNAME_STEM}'
        if self.__batchID is not None :
            pattern+= f'{self.__batchID:02d}_'
        pattern+='layer_*.png'
        caption = 'Flatfield correction factors in each image layer'
        lines+=self.image_layer_grid_plot_tex_lines(pattern,caption,figlabel)
        return lines

    @property
    def flatfield_uncertainty_layers(self) :
        lines = []
        lines.append(self.section_start('Layers of the flatfield correction image'))
        figlabel = 'fig:flatfield_uncertainty_image_layers'
        lines.append(f'Figure~\\ref{{{figlabel}}} shows the uncertainties on the flatfield correction factors for every image layer.\n')
        lines.append('\n')
        pattern = f'{CONST.FLATFIELD_DIRNAME_STEM}'
        if self.__batchID is not None :
            pattern+= f'{self.__batchID:02d}_'
        pattern+='uncertainty_layer_*.png'
        caption = 'Uncertainties on the flatfield correction factors in each image layer'
        lines+=self.image_layer_grid_plot_tex_lines(pattern,caption,figlabel)
        return lines

    @property
    def mask_stack_layers(self) :
        lines = []
        lines.append(self.section_start('Layers of the flatfield correction image'))
        figlabel = 'fig:flatfield_mask_stack_layers'
        l = f'Figure~\\ref{{{figlabel}}} shows every layer of the overall mask stack used to measure the flatfield correction factors. '
        l+= 'These plots give some insight as to how many total images contribute to the measurements made.'
        lines.append(l+'\n')
        lines.append('\n')
        pattern = f'{CONST.FLATFIELD_DIRNAME_STEM}'
        if self.__batchID is not None :
            pattern+= f'{self.__batchID:02d}_'
        pattern+='mask_stack_layer_*.png'
        caption = 'The stack of all image masks combined over every sample used to measure the flatfield corrections'
        lines+=self.image_layer_grid_plot_tex_lines(pattern,caption,figlabel)
        return lines

class AppliedFlatfieldLatexSummary(LatexSummaryWithPlotdir) :
    """
    Class to make a LatexSummary about the effects of applying a flatfield model 
    to a meanimage created from an orthogonal set of images in the same sample
    """

    def __init__(self,flatfield_image,smoothed_mean_image,smoothed_corrected_mean_image,plot_dirpath) :
        """
        flatfield_image = the flatfield image that was applied
        smoothed_mean_image = the pre-correction smoothed mean image
        smoothed_corrected_mean_image = the post-correction smoothed mean image
        plot_dirpath = the directory with all of the necessary plots in it
        """
        self.__flatfield_image = flatfield_image
        self.__smoothed_mean_image = smoothed_mean_image
        self.__smoothed_corrected_mean_image = smoothed_corrected_mean_image
        title = 'Applied Flatfield Summary'
        filename = f'{CONST.APPLIED_FLATFIELD_SUMMARY_PDF_FILENAME_STEM}.pdf'
        super().__init__(title,filename,plot_dirpath)

    @property
    def sections(self) :
        return super().sections+[self.mean_image_before_correction,self.flatfield_corrections_applied,self.effects_of_applying_corrections]

    @property
    def mean_image_before_correction(self) :
        lines = []
        lines.append(self.section_start('Mean image before correction'))
        lines.append('\n')
        mean_image_figlabel = 'fig:mean_image_layers'
        mean_image_pattern = 'mean_image_layer_*.png'
        mean_image_caption = 'All layers of the mean image before application of flatfield correction factors'
        mean_image_unc_figlabel = 'fig:mean_image_uncertainty_layers'
        mean_image_unc_pattern = 'mean_image_uncertainty_layer_*.png'
        mean_image_unc_caption = 'The standard error of the mean image in every layer'
        mask_stack_figlabel = 'fig:mask_stack_layers'
        mask_stack_pattern = 'corrected_mean_image_mask_stack_layer_*.png'
        mask_stack_caption = 'The stack of all masks for images used to compute the mean image'
        has_mask_stack = False
        for fn in self.plot_dirpath.glob(mask_stack_pattern) :
            has_mask_stack = True
        l = f'Figure~\\ref{{{mean_image_figlabel}}} shows each layer of the mean image before application of the calculated flatfield correction factors.'
        l+= f' Figure~\\ref{{{mean_image_unc_figlabel}}} shows the uncertainties in each layer of this mean image.'
        if has_mask_stack :
            l+= f' Figure~\\ref{{{mask_stack_figlabel}}} shows each layer of the stack of individual image masks, describing how many images contributed '
            l+= 'to this mean image at each location.'
        lines.append(l+'\n')
        lines.append('\n')
        lines+=self.image_layer_grid_plot_tex_lines(mean_image_pattern,mean_image_caption,mean_image_figlabel)
        lines.append('\n')
        lines+=self.image_layer_grid_plot_tex_lines(mean_image_unc_pattern,mean_image_unc_caption,mean_image_unc_figlabel)
        if has_mask_stack :
            lines.append('\n')
            lines+=self.image_layer_grid_plot_tex_lines(mask_stack_pattern,mask_stack_caption,mask_stack_figlabel)
        return lines

    @property
    def flatfield_corrections_applied(self) :
        lines = []
        lines.append(self.section_start('Flatfield correction factors applied'))
        lines.append('\n')
        figlabel = 'fig:flatflield_pixel_intensities'
        l = f'Figure~\\ref{{{figlabel}}} shows the 5th and 95th percentile, as well as the standard deviation, '
        l+= 'of the pixel-by-pixel correction factors in each layer of the flatfield model. The red lines and green shaded areas show '
        l+= 'the values calculated over the entire area of the correction image, and the blue lines and yellow shading show the values '
        l+= 'calculated considering only the central 64\\% ``primary region" of the correction image.'
        lines.append(l)
        lines.append('\n')
        caption = '5th and 95th percentile and standard deviation of flatfield correction factors in each image layer. Red lines and green '
        caption+= 'shaded areas show statistics calculated using the entire area of each image layer. Blue lines and yellow shading show '
        caption+= 'statistics calculated in the central ``primary region" of each image layer.'
        lines+=self.image_figure_lines(f'{self.plot_dirpath_tex}/flatfield_pixel_intensities',caption,figlabel,widths=1.0)
        lines.append('\n')
        tablabel = 'tab:average_correction_factors'
        l = f'Table\\ref{{{tablabel}}} lists the overall largest and smallest correction factors in any layer of the entire image and its '
        l+= 'primary region. It also denotes the average, over all image layers, of the 5th-95th percentile spread and standard deviation '
        l+= 'of the correction factors in the entire image and its primary region.'
        lines.append(l)
        caption = 'Summarizing statistics for the flatfield correction model. The central and rightmost columns list values calculated using '
        caption+= ' the entire area of the image and the central 64\\% ``primary region" of the image, respectively. From the upper to lower '
        caption+= 'row the values listed are the maximum and minimum correction factors in any image layer, the average over all image layers '
        caption+= 'of the spread from the 5th-95th percentile correction factors, and the average over all layers of the standard deviation '
        caption+= 'of the correction factors.'
        datatable = LatexDataTable(caption,tablabel)
        o_max, o_min, o_spread, o_stddev, c_max, c_min, c_spread, c_stddev = calculate_statistics_for_image(self.__flatfield_image)
        heading = 'Statistic & Whole image area & Central 64\\% ``primary region"'
        rows = [f'Maximum correction factor & {o_max:.05f} & {c_max:.05f}',
                f'Minimum correction factor & {o_min:.05f} & {c_min:.05f}',
                f'5th-95th pctile spread, avg. over all layers & {o_spread:.05f} & {c_spread:.05f}',
                f'Standard deviation, avg. over all layers & {o_stddev:.05f} & {c_stddev:.05f}'
               ]
        datatable.add_tabular(heading,rows)
        lines+=datatable.get_table_lines()
        lines.append('\n')
        flatfield_layers_pattern = 'flatfield_layer_*.png'
        flatfield_layers_caption = 'Flatfield correction factors in each image layer'
        flatfield_layers_figlabel = 'fig:flatfield_layers'
        l = f'Figure~\\ref{{{flatfield_layers_figlabel}}} shows each layer of the flatfield correction factor image. These correction factors were '
        l+= 'applied to the mean image shown in Fig.~\\ref{fig:mean_image_layers}. The mean image and flatfield correction factors were measured using '
        l+= 'equally-sized orthogonal subsets of the HPF images in each sample considered.'
        lines.append(l+'\n')
        lines+=self.image_layer_grid_plot_tex_lines(flatfield_layers_pattern,flatfield_layers_caption,flatfield_layers_figlabel)
        lines.append('\n')
        ff_unc_pattern = 'flatfield_uncertainty_layer_*.png'
        ff_unc_caption = 'Uncertainties on the flatfield correction factors in each image layer'
        ff_unc_figlabel = 'fig:flatfield_uncertainty'
        lines.append(f'Figure~\\ref{{{ff_unc_figlabel}}} shows the uncertainties on the flatfield correction factors.\n')
        lines+=self.image_layer_grid_plot_tex_lines(ff_unc_pattern,ff_unc_caption,ff_unc_figlabel)
        return lines

    @property
    def effects_of_applying_corrections(self) :
        lines = []
        lines.append(self.section_start('Effects of applying flatfield corrections'))
        lines.append('\n')
        cmi_layers_pattern = 'corrected_mean_image_layer_*.png'
        cmi_layers_caption = 'Layers of the mean image after application of the flatfield correction factors'
        cmi_layers_figlabel = 'fig:corrected_mean_image_layers'
        cmi_unc_layers_pattern = 'corrected_mean_image_uncertainty_layer_*.png'
        cmi_unc_layers_caption = 'The uncertainties on the flatfield-corrected mean image'
        cmi_unc_layers_figlabel = 'fig:corrected_mean_image_uncertainty_layers'
        l = f'Figure~\\ref{{{cmi_layers_figlabel}}} shows each layer of the mean image from Fig.~\\ref{{fig:mean_image_layers}} after application'
        l+= f' of the flatfield correction factors shown in Fig.~\\ref{{fig:flatfield_layers}}. Figure ~\\ref{{{cmi_unc_layers_figlabel}}} '
        l+= 'shows the uncertainties on this corrected mean image.'
        lines.append(l+'\n')
        lines+=self.image_layer_grid_plot_tex_lines(cmi_layers_pattern,cmi_layers_caption,cmi_layers_figlabel)
        lines+=self.image_layer_grid_plot_tex_lines(cmi_unc_layers_pattern,cmi_unc_layers_caption,cmi_unc_layers_figlabel)
        figlabel_1 = 'fig:smoothed_mean_image_pixel_intensities'
        figlabel_2 = 'fig:smoothed_mean_image_pixel_intensities_central_region'
        l = 'The mean images before and after application of the flatfield correction factors were smoothed with a wide Gaussian filter to remove '
        l+= f'small variations in pixel intensity. Figure~\\ref{{{figlabel_1}}} shows the 5th and 95th percentile and '
        l+= 'standard deviation of the mean-relative pixel intensity in each layer of the smoothed mean images before and after corrections were applied. '
        l+= f'Figure~\\ref{{{figlabel_2}}} shows the same statistics, but calculated using only the central '
        l+= '64\\% ``primary region" of the image instead of the entire image region.'
        lines.append(l)
        lines.append('\n')
        caption_1 = '5th and 95th percentile and standard deviation of mean-relative pixel intensity in each layer of the smoothed '
        caption_1+= 'mean images. Red lines and green shaded areas show statistics calculated for the smoothed mean image before application of the flatfield '
        caption_1+= 'correction factors. Blue lines and yellow shading show statistics calculated for the smoothed post-correction mean image.'
        lines+=self.image_figure_lines(f'{self.plot_dirpath_tex}/smoothed_mean_image_pixel_intensities',caption_1,figlabel_1)
        lines.append('\n')
        caption_2 = '5th and 95th percentile and standard deviation of mean-relative pixel intensity in the central ``primary '
        caption_2+= 'region" of each layer of the smoothed mean images. Red lines and green shaded areas show statistics calculated for the smoothed '
        caption_2+= 'mean image before application of the flatfield correction factors. Blue lines and yellow shading show statistics calculated for '
        caption_2+= 'the smoothed post-correction mean image.'
        lines+=self.image_figure_lines(f'{self.plot_dirpath_tex}/smoothed_mean_image_pixel_intensities_central_region',caption_2,figlabel_2)
        lines.append('\n')
        figlabel_3 = 'fig:illumination_variation_reduction'
        figlabel_4 = 'fig:illumination_variation_reduction_central_region'
        l = f'Figure~\\ref{{{figlabel_3}}} shows how the spread from the 5th to 95th percentile mean-relative flux and the standard '
        l+= 'deviation of the mean-relative flux in the smoothed mean images changed as a result of application of the correction factors in each image '
        l+= f'layer. Figure.~\\ref{{{figlabel_4}}} shows the same, calculated using the central ``primary region" of the image.'
        lines.append(l)
        lines.append('\n')
        caption_3 = 'Spread from 5th to 95th percentile and standard deviation of mean relative flux observed for the entire region of '
        caption_3+= 'the smoothed mean image before and after application of the flatfield correction factors.'
        lines+=self.image_figure_lines(f'{self.plot_dirpath_tex}/illumination_variation_reduction',caption_3,figlabel_3,widths=0.8)
        lines.append('\n')
        caption_4 = 'Spread from 5th to 95th percentile and standard deviation of mean relative flux observed for the central ``primary" '
        caption_4+= 'region of the smoothed mean image before and after application of the flatfield correction factors.'
        lines+=self.image_figure_lines(f'{self.plot_dirpath_tex}/illumination_variation_reduction_central_region',caption_4,figlabel_4,widths=0.8)
        lines.append('\n')
        tablabel = 'tab:illumination_variation_reduction'
        l = f'Table~\\ref{{{tablabel}}} lists the reductions in illumination variation, averaged over all image layers, observed '
        l+= 'in the smoothed mean images as a result of applying the flatfield corrections. The illumination variations listed are the 5th-95th percentile '
        l+= 'spread and standard deviation of the mean-relative pixel intensity in the smoothed mean images before and after correction. Values are shown '
        l+= 'for the entire images as well as for the central ``primary regions" of the images.'
        lines.append(l)
        caption = 'Summarizing statistics describing the effects of applying flatfield corrections to a mean image created from '
        caption+= "an orthogonal subsample of HPF images. Each statistic in the upper portion of the table describes the distribution of a smoothed mean "
        caption+= "image's mean-relative pixel intensities. The maxima and minima listed are the maximum/minimum mean-relative intensity in any image layer, "
        caption+= "and the 5th-95th percentile spread and standard deviation listed are the averages over all image layers. Values are given for both the entire "
        caption+= "image region and the central primary region, both before and after application of the flatfield corrections. The lower table lists the percent "
        caption+= "changes in the same statistics between the pre- and post-correction smoothed mean images, both for the overall image and in the central "
        caption+= "primary region."
        datatable = LatexDataTable(caption,tablabel)
        mean_rel_pre_smi = self.__smoothed_mean_image/(np.mean(self.__smoothed_mean_image,axis=(0,1))[np.newaxis,np.newaxis,:])
        o_pre_max, o_pre_min, o_pre_spread, o_pre_stddev, c_pre_max, c_pre_min, c_pre_spread, c_pre_stddev = calculate_statistics_for_image(mean_rel_pre_smi)
        mean_rel_post_smi = self.__smoothed_corrected_mean_image/(np.mean(self.__smoothed_corrected_mean_image,axis=(0,1))[np.newaxis,np.newaxis,:])
        o_post_max, o_post_min, o_post_spread, o_post_stddev, c_post_max, c_post_min, c_post_spread, c_post_stddev = calculate_statistics_for_image(mean_rel_post_smi)
        o_spread_change = 100.*(o_post_spread-o_pre_spread)/o_pre_spread
        c_spread_change = 100.*(c_post_spread-c_pre_spread)/c_pre_spread
        o_stddev_change = 100.*(o_post_stddev-o_pre_stddev)/o_pre_stddev
        c_stddev_change = 100.*(c_post_stddev-c_pre_stddev)/c_pre_stddev
        headings = ['Statistic & Pre-correction,  & Pre-correction,        & Post-correction, & Post-correction,',
                    '          & whole image area & central primary region & whole image area & central primary region'
                   ]
        rows = [f'Maximum mean-relative pixel intensity & {o_pre_max:.05f} & {c_pre_max:.05f} & {o_post_max:.05f} & {c_post_max:.05f}',
                f'Minimum mean-relative pixel intensity & {o_pre_min:.05f} & {c_pre_min:.05f} & {o_post_min:.05f} & {c_post_min:.05f}',
                f'5th-95th \\%ile spread, avg. over all layers & {o_pre_spread:.05f} & {c_pre_spread:.05f} & {o_post_spread:.05f} & {c_post_spread:.05f}',
                f'Standard deviation, avg. over all layers & {o_pre_stddev:.05f} & {c_pre_stddev:.05f} & {o_post_stddev:.05f} & {c_post_stddev:.05f}'
               ]
        datatable.add_tabular(headings,rows)
        headings = [f'Statistic & change due to correction,  & change due to correction,',
                     '          & whole image area           & central primary region'
                   ]
        rows = [f'5th-95th \\%ile spread, avg. over all layers & {o_spread_change:.02f}\\% & {c_spread_change:.02f}\\%',
                f'Standard deviation, avg. over all layers & {o_stddev_change:.02f}\\% & {c_stddev_change:.02f}\\%'
               ]
        datatable.add_tabular(headings,rows)
        lines+=datatable.get_table_lines()
        return lines
