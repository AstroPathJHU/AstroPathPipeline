#imports
from ...shared.latexsummary import LatexSummaryWithPlotdir

class FitGroupLatexSummary(LatexSummaryWithPlotdir) :
    """
    Class to make the summary LaTeX/pdf file for a single group of warping fits
    """

    def __init__(self,plot_dirpath,plot_name_stem,fit_group_name) :
        self.__fit_group_name = fit_group_name
        self.__plot_name_stem = plot_name_stem
        title = f'{self.__fit_group_name} Summary'
        filename = f'{plot_name_stem}_summary.pdf'
        super().__init__(title,filename,plot_dirpath)

    @property
    def sections(self) :
        return [*(super().sections),self.fit_outcomes,self.average_results]

    @property
    def fit_outcomes(self) :
        lines = []
        lines.append(self.section_start('Fit Outcomes'))
        lines.append('\n')
        pp_figlabel = 'fig:principal_point_plot'
        caption = 'Principal point locations found in each fit, color-coded by fractional reduction in fit cost'
        l = f'Figure~\\ref{{{pp_figlabel}}} shows a scatterplot of all the principal point locations found '
        l+= f'in the {self.__fit_group_name.lower()}, colored by the fractional reduction in the fit cost. '
        l+= 'The mean and weighted mean of these results are also shown with their uncertainties.'
        lines.append(l)
        lines+=self.image_figure_lines(f'{self.plot_dirpath_tex}/{self.__plot_name_stem}_principal_point_plot',
                                       caption,pp_figlabel,widths=0.7)
        rw_figlabel = 'fig:rad_warp_plots'
        caption = 'Upper left: maximum amounts of radial warping found in each fit, with mean and weighted mean. '
        caption+= 'Upper center: fractional reduction in fit cost vs. maximum amount of radial warping. '
        caption+= 'Upper right: principal point locations found, color-coded by the maximum amount of radial warping. '
        caption+= 'Center row: from left to right, fractional reductions in fit cost vs. $k1$, $k2$, and $k3$ '
        caption+= 'radial warping parameters, respectively. '
        caption+= 'Lower plot: dependence of $k1$, $k2$, and $k3$ on one another.'
        l = f'Figure~\\ref{{{rw_figlabel}}} shows several plots with details about the radial warping patterns found '
        l+= f'for the {self.__fit_group_name.lower()}. The leftmost plot in the upper row shows the maximum amounts '
        l+= '(in pixels) of radial warping found in each individual fit; the mean and weighted mean are '
        l+= 'also indicated. The center plot in the upper row shows a scatterplot of the fractional reduction in fit '
        l+= 'cost vs the maximum amount of radial warping. The right plot in the upper row shows the principal point '
        l+= f'locations found, like in Fig.~\\ref{{{pp_figlabel}}}, but this time colored by the maxmimum amount of '
        l+= 'radial warping rather than reduction in fit cost. The center row of plots show scatterplots of the '
        l+= 'fractional reduction in fit cost vs values found for the $k1$, $k2$, and $k3$ parameters from left '
        l+= 'to right, respectively. The lower plot shows a scatterplot of how the $k1$, $k2$, and $k3$ parameters '
        l+= 'depend on one another.'
        lines.append(l)
        lines+=self.image_figure_lines(
                [f'{self.plot_dirpath_tex}/{self.__plot_name_stem}_radial_warp_amount_plots',
                 f'{self.plot_dirpath_tex}/{self.__plot_name_stem}_cost_redux_vs_radial_warp_parameters_plots',
                 f'{self.plot_dirpath_tex}/{self.__plot_name_stem}_all_radial_warp_parameters_plot'],
                caption,rw_figlabel,widths=[0.95,0.95,0.6]
                )
        fi_figlabel = 'fig:fit_iterations_plot'
        caption = 'Fractional reduction in fit cost vs. number of fit iterations'
        l = f'Figure~\\ref{{{fi_figlabel}}} shows a scatterplot of the fractional reduction in fit cost vs the number '
        l+= 'of fit iterations.'
        lines.append(l)
        lines+=self.image_figure_lines(
                f'{self.plot_dirpath_tex}/{self.__plot_name_stem}_cost_redux_vs_fit_iterations_plot',
                caption,fi_figlabel,widths=0.7)
        return lines

    @property
    def average_results(self) :
        lines = []
        lines.append(self.section_start('Average of Results'))
        wfv_figlabel = 'fig:warp_field_variation_plots'
        caption = 'Columns from left to right show total amount of warping, amount of warping in the x direction, '
        caption+= 'and amount of warping in the y direction. Rows from top to bottom show the mean warp field, the '
        caption+= 'standard deviation of all warp fields, the weighted mean warp field, and the standard error on the '
        caption+= 'weighted mean warp field.'
        l = f'Figure~\\ref{{{wfv_figlabel}}} describes the mean and weighted mean of all warp fields found by the '
        l+= 'individual fits, along with their uncertainties. From left to right, the columns show the total amount '
        l+= 'of warp, the amount of warp in the x direction, and the amount of warp in the y direction, respectively. '
        l+= 'From top to bottom, the rows show the mean, standard deviation, weighted mean, and standard error on the '
        l+= 'weighted mean of all warp fields found, respectively. All units are in pixels.'
        lines.append(l)
        lines+=self.image_figure_lines(f'{self.plot_dirpath_tex}/{self.__plot_name_stem}_warp_field_variation_plots',
                                       caption,wfv_figlabel,widths=1.0)
        return lines
