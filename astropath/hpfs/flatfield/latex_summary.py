#imports
from .utilities import getBatchFlatfieldWorkingDirPath
from .config import CONST
from ...utilities.misc import cd
import pathlib, subprocess

#################### FILE-SCOPE CONSTANTS ####################

VECTRA_3_NAME = 'Vectra 3.0'
VECTRA_POLARIS_NAME = 'Vectra Polaris'

class LatexSummary :
    """
    A class to represent the LaTeX summary file that's built at the end of a batch_flatfield job
    """

    #################### PUBLIC FUNCTIONS ####################

    def __init__(self,rootdir,batchID,nslides,img_dims) :
        """
        rootdir  = path to the Clinical_Specimen directory for all the slides that contributed to this flatfield model
        batchID  = the batchID of this flatfield model
        nslides  = the number of slides used to build the model
        img_dims = the dimensions of the images used to make the model (used to determine which microscope it's for)
        """
        self.rootdir = rootdir
        self.batchID = batchID
        self.workingdir_path = getBatchFlatfieldWorkingDirPath(rootdir,batchID)
        self.nslides = nslides
        if img_dims==(1004,1344,35) :
            self.microscope = VECTRA_3_NAME
        elif img_dims==(1404,1872,43) :
            self.microscope = VECTRA_POLARIS_NAME
        else :
            raise ValueError(f'ERROR: image dimensions {img_dims} not recognized as a microscope!')
        self.tex_file_namestem=f'flatfield_BatchID_{batchID:02d}_summary'

    def buildTemplateFile(self) :
        """
        Function to write out a summary.tex file into the batch_flatfield working directory
        """
        template_file_lines = []
        template_file_lines+=self._getPreamble()
        template_file_lines+=self._getTitle()
        template_file_lines+=self._getPixelIntensitiesSection()
        template_file_lines.append('\\clearpage\n\n')
        template_file_lines+=self._getFlatfieldLayersSection()
        template_file_lines.append('\\clearpage\n\n')
        template_file_lines+=self._getNImagesStackedSection()
        template_file_lines.append('\\clearpage\n\n')
        template_file_lines+=self._getMaskStackLayersSection()
        template_file_lines.append('\\end{document}\n')
        with cd(self.workingdir_path) :
            with open(f'{self.tex_file_namestem}.tex','w') as fp :
                for line in template_file_lines :
                    fp.write(line)

    def compile(self) :
        """
        Function to compile the summary.tex file that was created
        """
        cmd = f'pdflatex {self.tex_file_namestem}.tex'
        with cd(self.workingdir_path) :
            try :
                subprocess.check_call(cmd)
            except Exception :
                return 1
            try :
                subprocess.check_call(cmd)
            except Exception :
                return 1
            if pathlib.Path.is_file(pathlib.Path(f'{self.tex_file_namestem}.pdf')) :
                exts_to_rm = ('.log','.aux','.tex')
                for ext in exts_to_rm :
                    fn = pathlib.Path(f'{self.tex_file_namestem}{ext}')
                    if pathlib.Path.is_file(fn) :
                        fn.unlink()
        return 0

    #################### PRIVATE HELPER FUNCTIONS ####################

    def _getPreamble(self) :
        lines = []
        lines.append('\\documentclass[letterpaper,11pt]{article}\n')
        lines.append('\\usepackage{graphicx}\n')
        lines.append('\\usepackage[left=20mm,top=20mm]{geometry}\n')
        lines.append('\\renewcommand{\\familydefault}{\\sfdefault}\n')
        lines.append('\n')
        return lines

    def _getTitle(self) :
        lines = []
        lines.append(f'\\title{{Flatfield Summary for Batch {self.batchID}}}\n')
        lines.append('\\date{\\today}\n')
        lines.append('\\begin{document}\n')
        lines.append('\\maketitle\n')
        lines.append('\n')
        return lines

    def _getPixelIntensitiesSection(self) :
        lines = []
        lines.append('\\section{Flatfield Pixel Intensities}\n')
        lines.append('\n')
        lines.append('\\begin{figure}[!ht]\n')
        lines.append('\\centering\n')
        imgpath = f'{CONST.POSTRUN_PLOT_DIRECTORY_NAME}/{CONST.PIXEL_INTENSITY_PLOT_NAME}'
        lines.append(f'\\includegraphics[width=0.98\\textwidth]{{{imgpath}}}\n')
        captiontxt = f'5th and 95th percentile and standard deviation of the Batch {self.batchID} {self.microscope} flatfield correction factors'
        captiontxt+= ' measured in each layer computed using the full images (red lines and green shading) and only the central 64\\% of interest'
        captiontxt+= ' (blue lines and orange shading).'
        lines.append(f'\\caption{{\\footnotesize {captiontxt}}}\n')
        lines.append('\\end{figure}\n')
        lines.append('\n')
        return lines

    def _getFlatfieldLayersSection(self) :
        lines = []
        lines.append('\\begin{figure}[!ht]\n')
        lines.append('\\centering\n')
        imgdir = f'{CONST.POSTRUN_PLOT_DIRECTORY_NAME}/{CONST.IMAGE_LAYER_PLOT_DIRECTORY_NAME}'
        if self.microscope==VECTRA_3_NAME :
            lines.append(f'\\includegraphics[width=0.3\\textwidth]{{{imgdir}/flatfield_layer_1}}\n')
            lines.append(f'\\includegraphics[width=0.3\\textwidth]{{{imgdir}/flatfield_layer_5}}\n')
            lines.append(f'\\includegraphics[width=0.3\\textwidth]{{{imgdir}/flatfield_layer_9}}\n')
            lines.append(f'\\includegraphics[width=0.3\\textwidth]{{{imgdir}/flatfield_layer_10}}\n')
            lines.append(f'\\includegraphics[width=0.3\\textwidth]{{{imgdir}/flatfield_layer_14}}\n')
            lines.append(f'\\includegraphics[width=0.3\\textwidth]{{{imgdir}/flatfield_layer_18}}\n')
            lines.append(f'\\includegraphics[width=0.3\\textwidth]{{{imgdir}/flatfield_layer_19}}\n')
            lines.append(f'\\includegraphics[width=0.3\\textwidth]{{{imgdir}/flatfield_layer_22}}\n')
            lines.append(f'\\includegraphics[width=0.3\\textwidth]{{{imgdir}/flatfield_layer_25}}\n')
            lines.append(f'\\includegraphics[width=0.3\\textwidth]{{{imgdir}/flatfield_layer_26}}\n')
            lines.append(f'\\includegraphics[width=0.3\\textwidth]{{{imgdir}/flatfield_layer_29}}\n')
            lines.append(f'\\includegraphics[width=0.3\\textwidth]{{{imgdir}/flatfield_layer_32}}\n')
            lines.append(f'\\includegraphics[width=0.3\\textwidth]{{{imgdir}/flatfield_layer_33}}\n')
            lines.append(f'\\includegraphics[width=0.3\\textwidth]{{{imgdir}/flatfield_layer_34}}\n')
            lines.append(f'\\includegraphics[width=0.3\\textwidth]{{{imgdir}/flatfield_layer_35}}\n')
            captiontxt = 'Layers 1, 5, 9, 10, 14, 18, 29, 22, 25, 26, 29, 32, 33, 34, and 35, from upper left to lower right, respectively,'
            captiontxt+= f' of the Batch {self.batchID} {self.microscope} flatfield image.'
            lines.append(f'\\caption{{\\footnotesize {captiontxt}}}\n')
        elif self.microscope==VECTRA_POLARIS_NAME :
            lines.append(f'\\includegraphics[width=0.32\\textwidth]{{{imgdir}/flatfield_layer_1}}\n')
            lines.append(f'\\includegraphics[width=0.32\\textwidth]{{{imgdir}/flatfield_layer_5}}\n')
            lines.append(f'\\includegraphics[width=0.32\\textwidth]{{{imgdir}/flatfield_layer_9}}\n')
            lines.append(f'\\includegraphics[width=0.32\\textwidth]{{{imgdir}/flatfield_layer_10}}\n')
            lines.append(f'\\includegraphics[width=0.32\\textwidth]{{{imgdir}/flatfield_layer_11}} \\\\\n')
            lines.append(f'\\includegraphics[width=0.32\\textwidth]{{{imgdir}/flatfield_layer_12}}\n')
            lines.append(f'\\includegraphics[width=0.32\\textwidth]{{{imgdir}/flatfield_layer_14}}\n')
            lines.append(f'\\includegraphics[width=0.32\\textwidth]{{{imgdir}/flatfield_layer_17}}\n')
            lines.append(f'\\includegraphics[width=0.32\\textwidth]{{{imgdir}/flatfield_layer_18}}\n')
            lines.append(f'\\includegraphics[width=0.32\\textwidth]{{{imgdir}/flatfield_layer_19}}\n')
            lines.append(f'\\includegraphics[width=0.32\\textwidth]{{{imgdir}/flatfield_layer_20}}\n')
            captiontxt = 'Layers 1, 5, 9, 10, 11, 12, 14, 17, 18, 19, and 20, from upper left to lower right, respectively,'
            captiontxt+= f' of the Batch {self.batchID} {self.microscope} flatfield image.'
            lines.append(f'\\caption{{\\footnotesize {captiontxt}}}\n')
            lines.append('\\end{figure}\n')
            lines.append('\n')
            lines.append('\\begin{figure}[!ht]\n')
            lines.append('\\centering\n')
            lines.append(f'\\includegraphics[width=0.32\\textwidth]{{{imgdir}/flatfield_layer_21}}\n')
            lines.append(f'\\includegraphics[width=0.32\\textwidth]{{{imgdir}/flatfield_layer_25}}\n')
            lines.append(f'\\includegraphics[width=0.32\\textwidth]{{{imgdir}/flatfield_layer_29}}\n')
            lines.append(f'\\includegraphics[width=0.32\\textwidth]{{{imgdir}/flatfield_layer_30}}\n')
            lines.append(f'\\includegraphics[width=0.32\\textwidth]{{{imgdir}/flatfield_layer_33}}\n')
            lines.append(f'\\includegraphics[width=0.32\\textwidth]{{{imgdir}/flatfield_layer_36}}\n')
            lines.append(f'\\includegraphics[width=0.32\\textwidth]{{{imgdir}/flatfield_layer_37}}\n')
            lines.append(f'\\includegraphics[width=0.32\\textwidth]{{{imgdir}/flatfield_layer_40}}\n')
            lines.append(f'\\includegraphics[width=0.32\\textwidth]{{{imgdir}/flatfield_layer_43}}\n')
            captiontxt = 'Layers 21, 25, 29, 30, 33, 36, 37, 40, and 43, from upper left to lower right, respectively,'
            captiontxt+= f' of the Batch {self.batchID} {self.microscope} flatfield image.'
            lines.append(f'\\caption{{\\footnotesize {captiontxt}}}\n')
        lines.append('\\end{figure}\n')
        lines.append('\n')
        return lines

    def _getNImagesStackedSection(self) :
        lines = []
        lines.append('\\section{Numbers of Images Stacked}\n')
        lines.append('\n')
        lines.append('\\begin{figure}[!ht]\n')
        lines.append('\\centering\n')
        imgpath = f'{CONST.POSTRUN_PLOT_DIRECTORY_NAME}/{CONST.N_IMAGES_STACKED_PER_LAYER_PLOT_NAME}'
        lines.append(f'\\includegraphics[width=0.98\\textwidth]{{{imgpath}}}\n')
        nirfp = pathlib.Path(self.workingdir_path / CONST.POSTRUN_PLOT_DIRECTORY_NAME / CONST.N_IMAGES_READ_TEXT_FILE_NAME)
        with open(nirfp,'r') as fp :
            nir = [int(l.rstrip()) for l in fp.readlines() if l.rstrip()!='']
        assert len(nir)==1 
        nir=nir[0]
        captiontxt = f'Number of images selected to be stacked in each layer from the original set of {nir} HPFs from {self.nslides}'
        captiontxt+= f' Batch {self.batchID} {self.microscope} slides.'
        lines.append(f'\\caption{{\\footnotesize {captiontxt}}}\n')
        lines.append('\\end{figure}\n')
        lines.append('\n')
        return lines

    def _getMaskStackLayersSection(self) :
        lines = []
        lines.append('\\section{Layers of the Mask Stack}\n')
        lines.append('\n')
        lines.append('\\begin{figure}[!ht]\n')
        lines.append('\\centering\n')
        imgdir = f'{CONST.POSTRUN_PLOT_DIRECTORY_NAME}/{CONST.IMAGE_LAYER_PLOT_DIRECTORY_NAME}'
        if self.microscope==VECTRA_3_NAME :
            lines.append(f'\\includegraphics[width=0.3\\textwidth]{{{imgdir}/mask_stack_layer_1}}\n')
            lines.append(f'\\includegraphics[width=0.3\\textwidth]{{{imgdir}/mask_stack_layer_5}}\n')
            lines.append(f'\\includegraphics[width=0.3\\textwidth]{{{imgdir}/mask_stack_layer_9}}\n')
            lines.append(f'\\includegraphics[width=0.3\\textwidth]{{{imgdir}/mask_stack_layer_10}}\n')
            lines.append(f'\\includegraphics[width=0.3\\textwidth]{{{imgdir}/mask_stack_layer_14}}\n')
            lines.append(f'\\includegraphics[width=0.3\\textwidth]{{{imgdir}/mask_stack_layer_18}}\n')
            lines.append(f'\\includegraphics[width=0.3\\textwidth]{{{imgdir}/mask_stack_layer_19}}\n')
            lines.append(f'\\includegraphics[width=0.3\\textwidth]{{{imgdir}/mask_stack_layer_22}}\n')
            lines.append(f'\\includegraphics[width=0.3\\textwidth]{{{imgdir}/mask_stack_layer_25}}\n')
            lines.append(f'\\includegraphics[width=0.3\\textwidth]{{{imgdir}/mask_stack_layer_26}}\n')
            lines.append(f'\\includegraphics[width=0.3\\textwidth]{{{imgdir}/mask_stack_layer_29}}\n')
            lines.append(f'\\includegraphics[width=0.3\\textwidth]{{{imgdir}/mask_stack_layer_32}}\n')
            lines.append(f'\\includegraphics[width=0.3\\textwidth]{{{imgdir}/mask_stack_layer_33}}\n')
            lines.append(f'\\includegraphics[width=0.3\\textwidth]{{{imgdir}/mask_stack_layer_34}}\n')
            lines.append(f'\\includegraphics[width=0.3\\textwidth]{{{imgdir}/mask_stack_layer_35}}\n')
            captiontxt = 'Layers 1, 5, 9, 10, 14, 18, 29, 22, 25, 26, 29, 32, 33, 34, and 35, from upper left to lower right, respectively,'
            captiontxt+= f' of the total stack of binary image masks for the Batch {self.batchID} {self.microscope} slides.'
            lines.append(f'\\caption{{\\footnotesize {captiontxt}}}\n')
        elif self.microscope==VECTRA_POLARIS_NAME :
            lines.append(f'\\includegraphics[width=0.32\\textwidth]{{{imgdir}/mask_stack_layer_1}}\n')
            lines.append(f'\\includegraphics[width=0.32\\textwidth]{{{imgdir}/mask_stack_layer_5}}\n')
            lines.append(f'\\includegraphics[width=0.32\\textwidth]{{{imgdir}/mask_stack_layer_9}}\n')
            lines.append(f'\\includegraphics[width=0.32\\textwidth]{{{imgdir}/mask_stack_layer_10}}\n')
            lines.append(f'\\includegraphics[width=0.32\\textwidth]{{{imgdir}/mask_stack_layer_11}} \\\\\n')
            lines.append(f'\\includegraphics[width=0.32\\textwidth]{{{imgdir}/mask_stack_layer_12}}\n')
            lines.append(f'\\includegraphics[width=0.32\\textwidth]{{{imgdir}/mask_stack_layer_14}}\n')
            lines.append(f'\\includegraphics[width=0.32\\textwidth]{{{imgdir}/mask_stack_layer_17}}\n')
            lines.append(f'\\includegraphics[width=0.32\\textwidth]{{{imgdir}/mask_stack_layer_18}}\n')
            lines.append(f'\\includegraphics[width=0.32\\textwidth]{{{imgdir}/mask_stack_layer_19}}\n')
            lines.append(f'\\includegraphics[width=0.32\\textwidth]{{{imgdir}/mask_stack_layer_20}}\n')
            captiontxt = 'Layers 1, 5, 9, 10, 11, 12, 14, 17, 18, 19, and 20, from upper left to lower right, respectively,'
            captiontxt+= f' of the total stack of binary image masks for the Batch {self.batchID} {self.microscope} slides.'
            lines.append(f'\\caption{{\\footnotesize {captiontxt}}}\n')
            lines.append('\\end{figure}\n')
            lines.append('\n')
            lines.append('\\begin{figure}[!ht]\n')
            lines.append('\\centering\n')
            lines.append(f'\\includegraphics[width=0.32\\textwidth]{{{imgdir}/mask_stack_layer_21}}\n')
            lines.append(f'\\includegraphics[width=0.32\\textwidth]{{{imgdir}/mask_stack_layer_25}}\n')
            lines.append(f'\\includegraphics[width=0.32\\textwidth]{{{imgdir}/mask_stack_layer_29}}\n')
            lines.append(f'\\includegraphics[width=0.32\\textwidth]{{{imgdir}/mask_stack_layer_30}}\n')
            lines.append(f'\\includegraphics[width=0.32\\textwidth]{{{imgdir}/mask_stack_layer_33}}\n')
            lines.append(f'\\includegraphics[width=0.32\\textwidth]{{{imgdir}/mask_stack_layer_36}}\n')
            lines.append(f'\\includegraphics[width=0.32\\textwidth]{{{imgdir}/mask_stack_layer_37}}\n')
            lines.append(f'\\includegraphics[width=0.32\\textwidth]{{{imgdir}/mask_stack_layer_40}}\n')
            lines.append(f'\\includegraphics[width=0.32\\textwidth]{{{imgdir}/mask_stack_layer_43}}\n')
            captiontxt = 'Layers 21, 25, 29, 30, 33, 36, 37, 40, and 43, from upper left to lower right, respectively,'
            captiontxt+= f' of the total stack of binary image masks for the Batch {self.batchID} {self.microscope} slides.'
            lines.append(f'\\caption{{\\footnotesize {captiontxt}}}\n')
        lines.append('\\end{figure}\n')
        lines.append('\n')
        return lines
