#imports
from .config import CONST
from ...utilities.misc import cd
import pathlib, subprocess

class LatexSummaryBase :
    """
    Base class for LaTeX summaries that combine a bunch of plot .pngs into a single .pdf document (with other information)
    """

    #################### PUBLIC FUNCTIONS ####################

    def __init__(self,title,filename,output_dir) :
        """
        title = the title of the document
        filename = the name of the .pdf file that will be outputted
        output_dir = the path to the directory that the outputted file should be saved in (if None this will be the current directory)
        """
        self.__title = title
        if not filename.endswith('.pdf') :
            filename+='.pdf'
        self.__pdf_filename = filename
        self.__tex_filename = filename.rstrip('.pdf')+'.tex'
        self.__aux_filename = filename.rstrip('.pdf')+'.aux'
        self.__log_filename = filename.rstrip('.pdf')+'.log'
        self.__output_dir = output_dir if output_dir is not None else pathlib.Path()

    def build_tex_file(self) :
        """
        Build the .tex file in the output directory
        """
        tex_file_lines = []
        for section in self.sections :
            tex_file_lines+=section
            if section!=self.preamble and section!=self.title :
                tex_file_lines.append('\\clearpage\n\n')
        tex_file_lines.append('\\end{document}\n')
        with cd(self.__output_dir) :
            with open(f'{self.__tex_filename}','w') as fp :
                for line in tex_file_lines :
                    fp.write(line)

    def compile(self) :
        """
        Compile the summary.tex file that was created into a .pdf and remove any files that are no longer needed
        If the compilation fails for any reason then remove the aux/log files and put the .tex file in some specified location
        Returns 0 if the compilation was successful and 1 otherwise
        """
        cmd = f'pdflatex {self.__tex_filename}'
        with cd(self.__output_dir) :
            try :
                subprocess.check_call(cmd)
                subprocess.check_call(cmd)
            except Exception :
                to_remove = [self.__output_dir / self.__aux_filename,
                             self.__output_dir / self.__log_filename,
                            ]
                for fp in to_remove :
                    if fp.is_file() :
                        fp.unlink()
                (self.__output_dir / self.__tex_filename).rename((self.failed_compilation_tex_file_path / self.__tex_filename))
                return 1
            if (self.__output_dir / self.__pdf_filename).is_file() :
                for fp in self.filepaths_to_remove_on_success :
                    try :
                        if fp.is_file() :
                            fp.unlink()
                        elif fp.is_dir() :
                            fp.rmdir()
                    except Exception :
                        pass
        return 0

    #################### PROPERTIES ####################

    @property
    def sections(self):
        """
        A list of lists of lines to write to the .tex file. The different sections (except for the preamble and title) 
        will be separated by "clearpage" commands
        """
        return [self.preamble,self.title]

    @property
    def failed_compilation_tex_file_path(self) :
        """
        Where to put the .tex file if it fails to be compiled
        """
        return self.__output_dir

    @property
    def filepaths_to_remove_on_success(self) :
        """
        File or directory paths to remove if the .tex file is successfully compiled into a .pdf
        """
        return [self.__output_dir / self.__aux_filename,
                self.__output_dir / self.__log_filename,
                self.__output_dir / self.__tex_filename,
            ]

    @property
    def preamble(self) :
        lines = []
        lines.append('\\documentclass[letterpaper,11pt]{article}\n')
        lines.append('\\usepackage{graphicx}\n')
        lines.append('\\usepackage[left=10mm,top=10mm,right=10mm,bottom=20mm]{geometry}\n')
        lines.append('\\renewcommand{\\familydefault}{\\sfdefault}\n')
        lines.append('\n')
        return lines

    @property
    def title(self) :
        lines = []
        lines.append(f'\\title{{{self.__title}}}\n')
        lines.append('\\date{\\today}\n')
        lines.append('\\begin{document}\n')
        lines.append('\\maketitle\n')
        lines.append('\n')
        return lines

class LatexSummaryForSlideWithPlotdir(LatexSummaryBase) :
    """
    Class to make a LatexSummary for a single slide with all of its plots in a single directory
    """

    def __init__(self,titlestem,filenamestem,slideID,plot_dirpath,plot_pattern='*') :
        """
        titlestem    = prefix for the title of the document
        filenamestem = suffix for the name of the file
        slideID      = ID of the slide used
        plot_dirpath = path to directory holding the plots that will be used
        plot_pattern = some pattern to use for glob in finding all of the plots to remove if the compilation is successful
                       (default is everything in the directory)
        """
        self.__slideID = slideID
        self.__plot_dirpath = plot_dirpath
        self.__plot_pattern = plot_pattern
        super().__init__(f'{titlestem} for {self.slideID_tex}',
                         f'{self.__slideID}-{filenamestem}',
                         plot_dirpath.parent)

    @property
    def failed_compilation_tex_file_path(self) :
        return self.__plot_dirpath

    @property
    def filepaths_to_remove_on_success(self) :
        to_remove = super().filepaths_to_remove_on_success
        for fp in self.__plot_dirpath.glob(self.__plot_pattern) :
            to_remove.append(fp.resolve())
        to_remove.append(self.__plot_dirpath)
        return to_remove

    @property
    def plot_dirpath(self) :
        return self.__plot_dirpath

    @property
    def plot_dirpath_tex(self) :
        return str(self.__plot_dirpath.as_posix())

    @property
    def slideID(self) :
        return self.__slideID

    @property
    def slideID_tex(self) :
        return self.__slideID.replace("_","\\_")

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
        super().__init__('Image Masking Summary',CONST.MASKING_SUMMARY_PDF_FILENAME,slideID,masking_plot_dirpath,'*_masking_plots.png')

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
        l = f'Figure~\\ref{{fig:mean_image_layers}} shows all individual layers of the computed mean image for {self.slideID_tex}. '
        l+= 'The units in these plots are units of intensity in average counts/ms.'
        lines.append(l+'\n')
        lines.append('\n')
        lines.append('\\begin{figure}[!ht]\n')
        lines.append('\\centering\n')
        for fn in self.plot_dirpath.glob(f'{self.slideID}-{CONST.MEAN_IMAGE_BIN_FILE_NAME_STEM.rstrip(".bin")}_layer_*.png') :
            lines.append(f'\\includegraphics[width=0.175\\textwidth]{{{self.plot_dirpath_tex}/{fn.name}}}\n')
        lines.append(f'\\caption{{\\footnotesize All layers of the mean image computed for {self.slideID_tex}}}\n')
        lines.append('\\label{fig:mean_image_layers}\n')
        lines.append('\\end{figure}\n')
        return lines

    @property
    def std_err_mean_image_plots(self) :
        lines = []
        lines.append('\\section{Layers of the standard error on the mean image}\n')
        l = f'Figure~\\ref{{fig:mean_image_std_err_layers}} shows the standard error of the {self.slideID_tex} mean image in all individual image layers. '
        l+= 'The units in these plots are also units of intensity in average counts/ms; they represent the uncertainties on the mean image layers shown in '
        l+= 'Figure~\\ref{fig:mean_image_layers}. Any overly bright regions that were not masked out will be especially apparent in these plots.'
        lines.append(l+'\n')
        lines.append('\n')
        lines.append('\\begin{figure}[!ht]\n')
        lines.append('\\centering\n')
        for fn in self.plot_dirpath.glob(f'{self.slideID}-{CONST.STD_ERR_OF_MEAN_IMAGE_BIN_FILE_NAME_STEM.rstrip(".bin")}_layer_*.png') :
            lines.append(f'\\includegraphics[width=0.175\\textwidth]{{{self.plot_dirpath_tex}/{fn.name}}}\n')
        lines.append(f'\\caption{{\\footnotesize The standard error of the mean image for {self.slideID_tex} in all layers}}\n')
        lines.append('\\label{fig:mean_image_std_err_layers}\n')
        lines.append('\\end{figure}\n')
        return lines

    @property
    def mask_stack_plots(self) :
        lines = []
        lines.append('\\section{Layers of the mask stack}\n')
        l = f'Figure~\\ref{{fig:mask_stack_layers}} shows every layer of the mask stack for {self.slideID_tex}. The units in these plots are the number of '
        l+= 'individual images contributing to the mean image at every location. They should be identical unless one or more image layer groups exhibited a '
        l+= 'great deal of saturation that was masked out. Referencing the general number of images stacked to find the mean illumination of each pixel helps '
        l+= 'contextualize the results above.'
        lines.append(l+'\n')
        lines.append('\n')
        lines.append('\\begin{figure}[!ht]\n')
        lines.append('\\centering\n')
        for fn in self.plot_dirpath.glob(f'{self.slideID}-{CONST.MASK_STACK_BIN_FILE_NAME_STEM.rstrip(".bin")}_layer_*.png') :
            lines.append(f'\\includegraphics[width=0.175\\textwidth]{{{self.plot_dirpath_tex}/{fn.name}}}\n')
        lines.append(f'\\caption{{\\footnotesize The stack of all masked images used to compute the {self.slideID_tex} mean image in all layers}}\n')
        lines.append('\\label{fig:mask_stack_layers}\n')
        lines.append('\\end{figure}\n')
        return lines
