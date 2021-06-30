#imports 
from ..utilities.misc import cd
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

class LatexSummaryWithPlotdir(LatexSummaryBase) :
    """
    Class to make a LatexSummary with all of its plots in a single directory
    """

    def __init__(self,title,filename,plot_dirpath,plot_patterns=['*']) :
        """
        plot_dirpath  = path to directory holding the plots that will be used (.pdf file is saved in this directory's parent)
        plot_patterns = list of patterns to use for glob in finding all of the plots to remove if the compilation is successful
                       (default is everything in the directory)
        """
        self.__plot_dirpath = plot_dirpath
        self.__plot_patterns = plot_patterns
        super().__init__(title,filename,plot_dirpath.parent)

    def image_layer_grid_plot_tex_lines(self,pattern,caption,label) :
        """
        return a list of tex lines for an entire grid of plots of individual image layers, sorted by layer number
        image layer plots should have filenames ending in "_layer_[n].png" to get filenames right automatically

        pattern = the pattern to search for to find the individual image layer plots
        caption = the caption of the plot
        label = the label of the plot
        """
        lines = []
        lines.append('\\begin{figure}[!ht]\n')
        lines.append('\\centering\n')
        all_plot_names = []
        for fn in self.__plot_dirpath.glob(pattern) :
            all_plot_names.append(fn.name)
        for pn in sorted(all_plot_names,key=lambda x:int(x.split('_')[-1].split('.')[0])) :
            lines.append(f'\\includegraphics[width=0.175\\textwidth]{{{self.plot_dirpath_tex}/{pn}}}\n')
        lines.append(f'\\caption{{\\footnotesize {caption}}}\n')
        lines.append(f'\\label{{{label}}}\n')
        lines.append('\\end{figure}\n')
        return lines

    @property
    def failed_compilation_tex_file_path(self) :
        #put the .tex file in the plot directory if it couldn't be compiled
        return self.__plot_dirpath

    @property
    def filepaths_to_remove_on_success(self) :
        #remove any of the plots the were used at the end, possibly even the whole plot directory if everything was used
        to_remove = super().filepaths_to_remove_on_success
        for pattern in self.__plot_patterns :
            for fp in self.__plot_dirpath.glob(pattern) :
                to_remove.append(fp.resolve())
        if self.__plot_patterns==['*'] :
            to_remove.append(self.__plot_dirpath)
        return to_remove

    @property
    def plot_dirpath(self) :
        return self.__plot_dirpath

    @property
    def plot_dirpath_tex(self) :
        return str(self.__plot_dirpath.as_posix())


class LatexSummaryForSlideWithPlotdir(LatexSummaryWithPlotdir) :
    """
    Class to make a LatexSummary for a single slide with all of its plots in a single directory
    """

    def __init__(self,titlestem,filenamestem,slideID,plot_dirpath,plot_patterns=['*']) :
        """
        titlestem     = prefix for the title of the document
        filenamestem  = suffix for the name of the file
        slideID       = ID of the slide used
        
        """
        self.__slideID = slideID
        super().__init__(f'{titlestem} for {self.slideID_tex}',
                         f'{self.__slideID}-{filenamestem}',
                         plot_dirpath,plot_patterns)

    @property
    def slideID(self) :
        return self.__slideID

    @property
    def slideID_tex(self) :
        return self.__slideID.replace("_","\\_")
