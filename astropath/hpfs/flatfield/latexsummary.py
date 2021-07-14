#imports
from .config import CONST
from ...shared.latexsummary import LatexSummaryWithPlotdir, LatexSummaryForSlideWithPlotdir
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
        lines.append('\\section{Locations of images used}\n')
        l = 'All High Power Fields (HPFs) located on the edges of the tissue were used to find the best overall background thresholds in every image layer. '
        l+= 'Figure~\\ref{fig:rectangle_locations} shows the locations of the tissue edge HPFs in red and the locations of the ``bulk" (non-edge) HPFs in blue.'
        lines.append(l+'\n')
        lines.append('\n')
        lines.append('\\begin{figure}[!ht]\n')
        lines.append('\\centering\n')
        lines.append(f'\\includegraphics[width=0.95\\textwidth]{{{self.plot_dirpath_tex}/{self.slideID}_rectangle_locations}}\n')
        l = f'\\caption{{\\footnotesize Locations of HPFs in slide {self.slideID_tex},'
        l+= ' with tissue edge HPFs shown in red and non-edge HPFs shown in blue.}'
        lines.append(l+'\n')
        lines.append('\\label{fig:rectangle_locations}\n')
        lines.append('\\end{figure}\n\n')
        return lines

    @property
    def thresholds_by_layer(self) :
        lines = []
        lines.append('\\section{Distributions of thresholds found by layer}')
        l = 'Figure~\\ref{fig:thresholds_by_layer} shows the 10th and 90th percentiles in the set of all individual HPF thresholds found in each image layer, '
        l+= 'as well as the final overall chosen thresholds, as a function of image layer, in both counts and counts/ms.'
        lines.append(l+'\n')
        lines.append('\n')
        lines.append('\\begin{figure}[!ht]\n')
        lines.append('\\centering\n')
        lines.append(f'\\includegraphics[width=0.95\\textwidth]{{{self.plot_dirpath_tex}/{self.slideID}_background_thresholds_by_layer}}')
        l = '\\caption{\\footnotesize 10th and 90th percentiles (in red and blue, respectively) of the entire set of individual HPF thresholds found '
        l+= 'in each image layer in raw counts (upper) and in counts/ms (lower). Also shown in black are the overall optimal thresholds chosen.}'
        lines.append(l+'\n')
        lines.append('\\label{fig:thresholds_by_layer}\n')
        lines.append('\\end{figure}\n\n')
        return lines

    @property
    def individual_layer_thresholding_plots(self) :
        all_fns = []
        for fn in self.plot_dirpath.glob('layer_*_background_threshold_plots.png') :
            all_fns.append(fn.name)
        figure_lines = []
        for ifn,fn in enumerate(sorted(all_fns,key=lambda x:int(str(x).split('_')[1]))) :
            figure_lines.append('\\begin{figure}[!ht]\n')
            figure_lines.append('\\centering\n')
            figure_lines.append(f'\\includegraphics[width=0.9\\textwidth]{{{self.plot_dirpath_tex}/{fn}}}\n')
            figure_lines.append(f'\\label{{fig:layer_{ifn+1}_threshold_plots}}\n')
            figure_lines.append('\\end{figure}\n\n')
            if ifn==1 or (ifn>1 and (ifn-1)%3==0) :
                figure_lines.append('\\clearpage\n\n')
        lines = []
        lines.append('\\section{Detailed thresholding results for each image layer}\n')
        l = f'Figures~\\ref{{fig:layer_1_threshold_plots}}-\\ref{{fig:layer_{len(all_fns)}_threshold_plots}} show detailed views of the thresholding results '
        l+= 'in each image layer. The left columns in those figures show histograms of all individual HPF thresholds found for a given image layer, along with '
        l+= 'the means and medians of the distributions. The right columns in those figures show pixel intensity histograms for the same image layer, '
        l+= 'on a log-log scale, summed over all tissue edge HPFs, with the signal and background pixels shown in different colors.'
        lines.append(l+'\n')
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
        lines.append('\\section{Flagged HPF locations}')
        l = f'Figure~\\ref{{fig:flagged_hpf_locations}} shows the locations of every HPF in {self.slideID_tex}, color-coded by their reason for being flagged.'
        lines.append(l+'\n')
        lines.append('\n')
        lines.append('\\begin{figure}[!ht]\n')
        lines.append('\\centering\n')
        lines.append(f'\\includegraphics[width=0.95\\textwidth]{{{self.plot_dirpath_tex}/{self.slideID}_flagged_hpf_locations}}')
        l = f'\\caption{{\\footnotesize locations of all HPFs in {self.slideID_tex}. Those shown in gray did not have any blur or saturation flagged in them. '
        l+= 'Those shown in blue had some region(s) flagged due to blur (either dust or folded tissue). Those shown in gold had some region(s) flagged due to '
        l+= 'saturation in at least one layer group. Those shown in green had some region(s) flagged due to both blur and saturation.}'
        lines.append(l+'\n')
        lines.append('\\label{fig:flagged_hpf_locations}\n')
        lines.append('\\end{figure}\n\n')
        return lines

    @property
    def reproduced_plots(self) :
        all_fns = []
        for fn in self.plot_dirpath.glob('*_masking_plots.png') :
            all_fns.append(fn.name)
        lines = []
        lines.append('\\section{Example masking plots}\n')
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
        l+= f'numerical values pictured corresponding to entries in the ``{labelled_mask_regions_fn_tex} file for the sample.'
        lines.append(l+'\n\n')
        for ifn,fn in enumerate(all_fns) :
            lines.append('\\begin{figure}[!ht]\n')
            lines.append('\\centering\n')
            lines.append(f'\\includegraphics[width=\\textwidth]{{{self.plot_dirpath_tex}/{fn}}}\n')
            img_key_tex = fn.rstrip("_masking_plots.png").replace("_","\\_")
            lines.append(f'\\caption{{\\footnotesize Masking plots for {img_key_tex}}}\n')
            if ifn==0 :
                lines.append('\\label{fig:first_masking_plot}\n')
            elif ifn==len(all_fns)-1 :
                lines.append('\\label{fig:last_masking_plot}\n')
            lines.append('\\end{figure}\n')
            lines.append('\\clearpage\n')
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
        lines.append('\\section{Layers of the mean image}\n')
        figlabel = 'fig:mean_image_layers'
        l = f'Figure~\\ref{{{figlabel}}} shows all individual layers of the computed mean image for {self.slideID_tex}. '
        l+= 'The units in these plots are units of intensity in average counts/ms.'
        lines.append(l+'\n')
        lines.append('\n')
        pattern = f'{self.slideID}-{CONST.MEAN_IMAGE_BIN_FILE_NAME_STEM.rstrip(".bin")}_layer_*.png'
        caption = f'All layers of the mean image computed for {self.slideID_tex}'
        lines+=self.image_layer_grid_plot_tex_lines(pattern,caption,figlabel)
        return lines

    @property
    def std_err_mean_image_plots(self) :
        lines = []
        lines.append('\\section{Layers of the standard error on the mean image}\n')
        figlabel = 'fig:mean_image_std_err_layers'
        l = f'Figure~\\ref{{{figlabel}}} shows the standard error of the {self.slideID_tex} mean image in all individual image layers. '
        l+= 'The units in these plots are also units of intensity in average counts/ms; they represent the uncertainties on the mean image layers shown in '
        l+= 'Figure~\\ref{fig:mean_image_layers}. Any overly bright regions that were not masked out will be especially apparent in these plots.'
        lines.append(l+'\n')
        lines.append('\n')
        pattern = f'{self.slideID}-{CONST.STD_ERR_OF_MEAN_IMAGE_BIN_FILE_NAME_STEM.rstrip(".bin")}_layer_*.png'
        caption = f'The standard error of the mean image for {self.slideID_tex} in all layers'
        lines+=self.image_layer_grid_plot_tex_lines(pattern,caption,figlabel)
        return lines

    @property
    def mask_stack_plots(self) :
        lines = []
        lines.append('\\section{Layers of the mask stack}\n')
        figlabel = 'fig:mask_stack_layers'
        l = f'Figure~\\ref{{{figlabel}}} shows every layer of the mask stack for {self.slideID_tex}. The units in these plots are the number of '
        l+= 'individual images contributing to the mean image at every location. They should be identical unless one or more image layer groups exhibited a '
        l+= 'great deal of saturation that was masked out. Referencing the general number of images stacked to find the mean illumination of each pixel helps '
        l+= 'contextualize the results above.'
        lines.append(l+'\n')
        lines.append('\n')
        pattern = f'{self.slideID}-{CONST.MASK_STACK_BIN_FILE_NAME_STEM.rstrip(".bin")}_layer_*.png'
        caption = f'The stack of all masked images used to compute the {self.slideID_tex} mean image in all layers'
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
        yclip = int(self.__flatfield_image.shape[0]*0.1)
        xclip = int(self.__flatfield_image.shape[1]*0.1)
        flatfield_image_clipped=self.__flatfield_image[yclip:-yclip,xclip:-xclip,:]
        overall_max = np.max(self.__flatfield_image)
        overall_min = np.min(self.__flatfield_image)
        central_max = np.max(flatfield_image_clipped)
        central_min = np.min(flatfield_image_clipped)
        overall_spreads_by_layer = []; overall_stddevs_by_layer = []
        central_spreads_by_layer = []; central_stddevs_by_layer = []
        for li in range(self.__flatfield_image.shape[-1]) :
            sorted_u_layer = np.sort((self.__flatfield_image[:,:,li]).flatten())/np.mean(self.__flatfield_image[:,:,li])
            sorted_c_layer = np.sort((flatfield_image_clipped[:,:,li]).flatten())/np.mean(self.__flatfield_image[:,:,li])
            overall_spreads_by_layer.append(sorted_u_layer[int(0.95*len(sorted_u_layer))]-sorted_u_layer[int(0.05*len(sorted_u_layer))])
            overall_stddevs_by_layer.append(np.std(sorted_u_layer))
            central_spreads_by_layer.append(sorted_c_layer[int(0.95*len(sorted_c_layer))]-sorted_c_layer[int(0.05*len(sorted_c_layer))])
            central_stddevs_by_layer.append(np.std(sorted_c_layer))
        overall_spread = np.mean(np.array(overall_spreads_by_layer))
        overall_stddev = np.mean(np.array(overall_stddevs_by_layer))
        central_spread = np.mean(np.array(central_spreads_by_layer))
        central_stddev = np.mean(np.array(central_stddevs_by_layer))
        lines = []
        lines.append('\\section{Flatfield Correction Factors}\n')
        lines.append('\n')
        l = 'Figure~\\ref{fig:flatflield_pixel_intensities} shows the 5th and 95th percentile, as well as the standard deviation, of the pixel-by-pixel '
        l+= f'correction factors in each layer of the flatfield model'
        if self.__batchID is not None :
            l+=f' created for batch {self.__batchID:02d}'
        l+='. The red lines and green shaded areas show the values calculated over the entire area of the correction image, and the blue lines and yellow '
        l+='shading show the values calculated considering only the central 64\\% ``primary region" of the correction image.'
        lines.append(l+'\n')
        lines.append('\n')
        lines.append('\\begin{figure}[!ht]\n')
        lines.append('\\centering\n')
        l = f'\\includegraphics[width=\\textwidth]{{{self.plot_dirpath_tex}/flatfield'
        if self.__batchID is not None :
            l+=f'_BatchID_{self.__batchID:02d}'
        l+='_pixel_intensities}'
        lines.append(l+'\n')
        l = '\\caption{\\footnotesize 5th and 95th percentile and standard deviation of flatfield correction factors in each image layer. Red lines and green '
        l+= 'shaded areas show statistics calculated using the entire area of each image layer. Blue lines and yellow shading show statistics calculated in the '
        l+= 'central ``primary region" of each image layer.}'
        lines.append(l+'\n')
        lines.append('\\label{fig:flatflield_pixel_intensities}\n')
        lines.append('\\end{figure}\n')
        lines.append('\n')
        l = 'Table~\\ref{tab:average_correction_factors} lists the overall largest and smallest correction factors in any layer of the entire image and its '
        l+= 'primary region. It also denotes the average, over all image layers, of the 5th-95th percentile spread and standard deviation of the correction '
        l+= 'factors in the entire image and its primary region.'
        lines.append(l+'\n')
        lines.append('\n')
        lines.append('\\begin{table}[!htb]\n')
        lines.append('\\centering\n')
        lines.append('\\footnotesize\n')
        lines.append('\\begin{tabular}{c c c}\n')
        lines.append('\\hline\n')
        lines.append('Statistic & Whole image area & Central 64\\% ``primary region" \\\\\n')
        lines.append('\\hline\n')
        lines.append(f'Maximum correction factor & {overall_max:.04f} & {central_max:.04f} \\\\\n')
        lines.append(f'Minimum correction factor & {overall_min:.04f} & {central_min:.04f} \\\\\n')
        lines.append(f'5th-95th pctile spread, avg. over all layers & {overall_spread:.04f} & {central_spread:.04f} \\\\\n')
        lines.append(f'Standard deviation, avg. over all layers & {overall_stddev:.04f} & {central_stddev:.04f} \\\\\n')
        lines.append('\\hline\n')
        lines.append('\\end{tabular}\n')
        l = '\\caption{\\footnotesize Summarizing statistics for the flatfield correction model. The central and rightmost columns list values calculated using '
        l+= ' the entire area of the image and the central 64\\% ``primary region" of the image, respectively. From the upper to lower row the values listed are '
        l+= 'the maximum and minimum correction factors in any image layer, the average over all image layers of the spread from the 5th-95th percentile '
        l+= 'correction factors, and the average over all layers of the standard deviation of the correction factors.}'
        lines.append(l+'\n')
        lines.append('\\label{tab:average_correction_factors}\n')
        lines.append('\\end{table}\n')
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
        lines.append('\\section{Layers of the flatfield correction image}\n')
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
        lines.append('\\section{Layers of the flatfield correction image}\n')
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
        return super().sections
