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
        lines.append('\\usepackage[left=20mm,top=20mm]{geometry}\n')
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

class ThresholdingLatexSummary(LatexSummaryBase) :
    """
    Class to make the background thresholding summary file for a single slide
    """

    def __init__(self,slideID,threshold_plot_dirpath) :
        """
        slideID = the sample slideID
        threshold_plot_dirpath = path to the directory with all of the thresholding summary plots in it
        """
        self.__slideID = slideID
        self.__threshold_plot_dirpath = threshold_plot_dirpath
        super().__init__(f'Background Thresholding Summary for {self.slideID_tex}',
                         CONST.THRESHOLDING_SUMMARY_PDF_FILENAME,
                         self.__threshold_plot_dirpath.parent)

    @property
    def sections(self) :
        return super().sections+[self.rect_locations,self.thresholds_by_layer,self.individual_layer_thresholding_plots]

    @property
    def failed_compilation_tex_file_path(self) :
        return self.__threshold_plot_dirpath

    @property
    def filepaths_to_remove_on_success(self) :
        to_remove = super().filepaths_to_remove_on_success
        for fp in self.__threshold_plot_dirpath.glob('*') :
            to_remove.append(fp.resolve())
        to_remove.append(self.__threshold_plot_dirpath)
        return to_remove

    @property
    def plot_dirpath_tex(self) :
        return str(self.__threshold_plot_dirpath.as_posix())

    @property
    def slideID_tex(self) :
        return self.__slideID.replace("_","\\_")

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
        lines.append(f'\\includegraphics[width=0.95\\textwidth]{{{self.plot_dirpath_tex}/{self.__slideID}_rectangle_locations}}\n')
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
        lines.append(f'\\includegraphics[width=0.95\\textwidth]{{{self.plot_dirpath_tex}/{self.__slideID}_background_thresholds_by_layer}}')
        l = '\\caption{\\footnotesize 10th and 90th percentiles (in red and blue, respectively) of the entire set of individual HPF thresholds found '
        l+= 'in each image layer in raw counts (upper) and in counts/ms (lower). Also shown in black are the overall optimal thresholds chosen.}'
        lines.append(l+'\n')
        lines.append('\\label{fig:thresholds_by_layer}\n')
        lines.append('\\end{figure}\n\n')
        return lines

    @property
    def individual_layer_thresholding_plots(self) :
        all_fns = []
        for fn in self.__threshold_plot_dirpath.glob('layer_*_background_threshold_plots.png') :
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


