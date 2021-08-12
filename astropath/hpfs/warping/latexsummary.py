#imports
from ...shared.latexsummary import LatexSummaryWithPlotdir

class FitGroupLatexSummary(LatexSummaryWithPlotdir) :
    """
    Class to make the summary LaTeX/pdf file for a single group of warping fits
    """

    def __init__(self,plot_dirpath,plot_name_stem,fit_group_name) :
        title = f'{fit_group_name} Summary'
        filename = f'{plot_name_stem}_summary.pdf'
        super().__init__(title,filename,plot_dirpath)

    @property
    def sections(self) :
        return [*(super().sections),self.principal_point_plot,self.rad_warp_plots,
                self.fit_iteration_plot,self.warp_field_plots]

    @property
    def principal_point_plot(self) :
        lines = []
        return lines

    @property
    def rad_warp_plots(self) :
        lines = []
        return lines

    @property
    def fit_iteration_plot(self) :
        lines = []
        return lines

    @property
    def warp_field_plots(self) :
        lines = []
        return lines