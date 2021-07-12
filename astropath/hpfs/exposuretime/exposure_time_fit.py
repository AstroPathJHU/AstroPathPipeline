#imports
from .overlap_with_exposure_times import OverlapWithExposureTimes
from .utilities import et_fit_logger, FieldLog
from .alignsample import AlignSampleForExposureTime
from ...utilities.tableio import writetable
from ...utilities.misc import cd, MetadataSummary, save_figure_in_dir
from ...utilities.config import CONST as UNIV_CONST
import numpy as np, matplotlib.pyplot as plt
from matplotlib import colors
import pathlib, random, scipy, platform

#helper class to do the fit in one image layer only
class SingleLayerExposureTimeFit :

    #################### PUBLIC FUNCTIONS ####################

    def __init__(self,layer_n,exp_times,med_exp_time,top_plot_dir,slideID,rawfile_top_dir,root_dir,flatfield,min_frac,overlaps,smoothsigma,cutimages) :
        """
        layer_n          = layer number that this fit will run on (indexed from 1)
        exp_times        = dictionary of raw file exposure times, keyed by filename stem
        med_exp_time     = the median exposure time in the entire slideID for this layer
        top_plot_dir     = path to directory in which this fit's subdirectory should be created
        slideID          = name of the slide to fit to ("M21_1" or equivalent)
        rawfile_top_dir  = path to directory containing [slideID] directory with multilayered ".Data.dat" files in it
        root_dir         = path to Clinical_Specimen directory for this slide
        flatfield        = relevant layer of the flatfield file
        min_frac         = some image in the dataset must have at least this fraction of pixels with the maximum offset (prevents too low of a maximum)
        overlaps         = list of overlap #s to run ([-1] if all should be run)
        smoothsigma      = sigma for Gaussian blurring
        cutimages        = True if only central 50% of overlap images should be used
        """
        self.layer = layer_n
        self.med_exp_time = med_exp_time
        self.rawfile_top_dir = rawfile_top_dir
        self.root_dir = root_dir
        self.slideID = slideID
        #set the flatfield
        self.flatfield = flatfield
        #set some other variables that are needed
        self.offsets = []
        self.costs = []
        self.best_fit_offset = None
        self.best_fit_cost = None
        #make sure the fit will actually use some overlaps
        if len(overlaps)<1 :
            et_fit_logger.warn(f'WARNING: layer {self.layer} does not have any aligned overlaps with exposure time differences. This fit will be skipped!')
            self.offset_bounds = [0,100]
            self.exposure_time_overlaps = []
            return
        #make this fit's plot directory name/path
        plotdirname = f'{self.slideID}_layer_{self.layer}_info'
        with cd(top_plot_dir) :
            if not pathlib.Path.is_dir(pathlib.Path(plotdirname)) :
                pathlib.Path.mkdir(pathlib.Path(plotdirname))
        self.plotdirpath = pathlib.Path(f'{top_plot_dir}/{plotdirname}')
        #make an alignsample from the raw files, smoothed and corrected with the flatfield
        et_fit_logger.info(f'Making an AlignSample for just the overlaps with different exposure times in layer {self.layer}....')
        use_GPU = platform.system()!='Darwin'
        a = AlignSampleForExposureTime(self.root_dir,self.rawfile_top_dir,self.slideID,selectoverlaps=overlaps,onlyrectanglesinoverlaps=True,
                                nclip=UNIV_CONST.N_CLIP,useGPU=use_GPU,readlayerfile=False,layer=self.layer,filetype='raw',
                                smoothsigma=smoothsigma,flatfield=self.flatfield)
        #get all the raw file layers and align the overlaps
        a.getDAPI()
        et_fit_logger.info(f'Aligning layer {self.layer} overlaps with corrected/smoothed images....')
        a.align(alreadyalignedstrategy='overwrite')
        #use the alignsample 
        self.offset_bounds = self.__getOffsetBounds(a,min_frac)
        self.exposure_time_overlaps = self.__getExposureTimeOverlaps(a,exp_times,med_exp_time,cutimages)

    def doFit(self,initial_offset,max_iter,gtol,eps,print_every) :
        """
        Run the fit for this layer
        initial_offset = starting point for fits
        max_iter       = maximum number of iterations for each fit to run
        gtol           = gradient projection tolerance for fits
        eps            = step size for approximating Jacobian
        print_every    = how often to print during minimization
        """
        if len(self.exposure_time_overlaps)<1 :
            et_fit_logger.warn(f'WARNING: skipping fit for layer {self.layer} because there are not enough overlaps!')
            return
        msg = f'Starting fit for layer {self.layer} at offset = {initial_offset}; will run for a max of {max_iter} '
        msg+= f'iterations printing every {print_every}'
        et_fit_logger.info(msg)
        self.result = scipy.optimize.minimize(self.__cost,
                                              [initial_offset],
                                              method='L-BFGS-B',
                                              jac='2-point',
                                              bounds=[self.offset_bounds],
                                              options={'disp':True,
                                                       'ftol':1e-20,
                                                       'gtol':gtol,
                                                       'eps':eps,
                                                       'maxiter':max_iter,
                                                       'iprint':print_every,
                                                       'maxls':50,
                                                      }
                                             )
        self.best_fit_offset = self.result.x[0]
        self.best_fit_cost = self.result.fun
        msg = f'Layer {self.layer} fit done! Minimization terminated with exit {self.result.message}. '
        msg+= f'Best-fit offset = {self.best_fit_offset:.4f} (best cost = {self.best_fit_cost:.4f})'
        et_fit_logger.info(msg)

    def writeOutResults(self,n_comparisons_to_save) :
        """
        Save some post-processing plots for this fit
        n_comparisons_to_save = total # of overlap overlay comparisons to write out for each completed fit
        """
        if len(self.exposure_time_overlaps)<1 :
            et_fit_logger.warn(f'WARNING: skipping layer {self.layer} because there are no results to write out!')
            return
        if self.best_fit_offset is None :
            raise RuntimeError('ERROR: best fit offset is None; run fit before calling writeOutResults!')
        self.__plotCostsAndOffsets()
        self.__writeResultsAndPlotCostReductions()
        self.__saveComparisonImages(n_comparisons_to_save)
        
    #################### PRIVATE HELPER FUNCTIONS ####################

    #helper function to calculate the fitting cost
    def __cost(self,pars) :
        offset=pars[0]
        cost=0.; npix=0
        for eto in self.exposure_time_overlaps :
            thiscost, thisnpix = eto.getCostAndNPix(offset)
            cost+=thiscost; npix+=thisnpix
        cost/=npix
        self.offsets.append(offset); self.costs.append(cost)
        return cost

    #helper function to automatically set the bounds on the offset to search
    def __getOffsetBounds(self,alignset,min_frac) :
        et_fit_logger.info(f'Finding upper bound on offsets to test for {self.slideID} layer {self.layer}....')
        offset_upper_bound = 1000.; nrectpix = alignset.rectangles[0].image.shape[0]*alignset.rectangles[0].image.shape[1]
        for r in alignset.rectangles :
            img = r.image
            this_upper_bound = -1
            while np.count_nonzero(img[img<this_upper_bound])<(nrectpix*min_frac) :
                this_upper_bound+=1
            if this_upper_bound<offset_upper_bound :
                et_fit_logger.info(f'Offset upper bound reduced to {this_upper_bound} in rectangle {r.n} for {self.slideID} layer {self.layer}')
                offset_upper_bound = this_upper_bound
        upper_bound_pixel_count = 0
        for r in alignset.rectangles :
            upper_bound_pixel_count+= np.count_nonzero(r.image[r.image<offset_upper_bound])
        pct=100.*(upper_bound_pixel_count/(len(alignset.rectangles)*(nrectpix)))
        msg = f'Offset upper bound for {self.slideID} layer {self.layer} found at {offset_upper_bound}'
        msg+= f' ({upper_bound_pixel_count} total pixels or {pct:.8f}% overall clipped at max)'
        et_fit_logger.info(msg)
        return [0,offset_upper_bound]

    #helper function to return a list of OverlapWithExposureTime objects set up to run on this particular image layer
    def __getExposureTimeOverlaps(self,alignset,exposure_times,med_exp_time,cutimages) :
        #make the exposure time comparison overlap objects
        etolaps = []; relevant_rectangles = {}
        rects_by_n = {}
        for r in alignset.rectangles :
            rects_by_n[r.n] = r
        for io,olap in enumerate(alignset.overlaps) :
            if olap.result.exit!=0 :
                continue
            p1rect = rects_by_n[olap.p1]
            p2rect = rects_by_n[olap.p2]
            p1et = exposure_times[(p1rect.file).rstrip(UNIV_CONST.IM3_EXT)]
            p2et = exposure_times[(p2rect.file).rstrip(UNIV_CONST.IM3_EXT)]
            et_fit_logger.info(f'Finding costs for overlap {olap.n} ({io+1} of {len(alignset.overlaps)} in {self.slideID} layer {self.layer})....')
            etolaps.append(OverlapWithExposureTimes(olap,p1et,p2et,med_exp_time,cutimages,self.offset_bounds))
            if p1rect.n not in relevant_rectangles.keys() :
                relevant_rectangles[p1rect.n]=p1rect
            if p2rect.n not in relevant_rectangles.keys() :
                relevant_rectangles[p2rect.n]=p2rect
        relevant_rectangles = list(relevant_rectangles.values())
        #make the log of the fields used and write it out
        field_logs = []
        for r in relevant_rectangles :
            this_rect_overlaps = [o.n for o in alignset.overlaps if (o.p1==r.n or o.p2==r.n)]
            field_logs.append(FieldLog(r.file,r.n,this_rect_overlaps))
        with cd(self.plotdirpath) :
            writetable(f'fields_used_in_exposure_time_fit_{self.slideID}_layer_{self.layer}.csv',field_logs)
        #make the metadata summary object and write it out
        metadata_summary = MetadataSummary(self.slideID,alignset.Project,alignset.Cohort,alignset.microscopename,
                                           str(min([r.t for r in relevant_rectangles])),str(max([r.t for r in relevant_rectangles])))
        with cd(self.plotdirpath) :
            writetable(f'metadata_summary_exposure_time_{self.slideID}_layer_{self.layer}.csv',[metadata_summary])
        #return the list of exposure time overlaps and the summary of the metadata of the alignmentSet they came from
        et_fit_logger.info(f'Found {len(etolaps)} overlaps that are aligned and have different p1 and p2 exposure times in layer {self.layer}')
        return etolaps

    #helper function to plot cost and offset tested at each fit iteration
    def __plotCostsAndOffsets(self) :
        et_fit_logger.info(f'Plotting costs and offsets for layer {self.layer}....')
        f,ax=plt.subplots(1,2,figsize=(2*6.4,4.6))
        ax[0].plot(list(range(1,len(self.costs)+1)),self.costs,marker='*')
        ax[0].set_xlabel('fit iteration')
        ax[0].set_ylabel('fit cost')
        ax[1].plot(list(range(1,len(self.costs)+1)),self.offsets,marker='*')
        ax[1].set_xlabel('fit iteration')
        ax[1].set_ylabel('offset')
        fn = f'costs_and_offsets_{self.slideID}_layer_{self.layer}.png'
        save_figure_in_dir(plt,fn,self.plotdirpath)

    #helper function to make a plot of each overlap's cost reduction and write out the table of overlap fit results
    def __writeResultsAndPlotCostReductions(self) :
        et_fit_logger.info(f'Writing out fit results for layer {self.layer}....')
        #write out table of overlap fit results
        fitresults = [eto.getFitResult(self.best_fit_offset) for eto in self.exposure_time_overlaps]
        with cd(self.plotdirpath) :
            writetable(f'overlap_fit_results_{self.slideID}_layer_{self.layer}.csv',fitresults)
        #make 1D pre/postfit cost and cost reduction plots
        f,ax=plt.subplots(1,3,figsize=(3*6.4,4.6))
        prefit_costs  = [r.prefit_cost for r in fitresults]
        postfit_costs = [r.postfit_cost for r in fitresults]
        _,bins1,_ = ax[0].hist(prefit_costs,bins=np.linspace(0.,max(max(prefit_costs),max(postfit_costs))+2,80),alpha=0.7,label='prefit costs')
        ax[0].hist(postfit_costs,bins=bins1,alpha=0.7,label='postfit costs')
        ax[0].set_xlabel('overlap cost')
        ax[0].set_ylabel('number of overlaps')
        ax[0].legend(loc='best')
        ax[1].hist([(r.prefit_cost-r.postfit_cost) for r in fitresults],bins=60)
        ax[1].set_xlabel('original cost - post-fit cost')
        ax[1].set_ylabel('number of overlaps')
        ax[2].hist([(r.prefit_cost-r.postfit_cost)/r.prefit_cost for r in fitresults],bins=60)
        ax[2].set_xlabel('(original cost - post-fit cost)/(original cost)')
        ax[2].set_ylabel('number of overlaps')
        fn = f'cost_reduction_plots_1d_{self.slideID}_layer_{self.layer}.png'
        save_figure_in_dir(plt,fn,self.plotdirpath)
        #make 2D pre/postfit cost and cost reduction plots
        f,ax = plt.subplots(2,2,figsize=(2*6.4,2*4.6))
        etdiffs = [r.et_diff for r in fitresults]
        pos = ax[0][0].hist2d(etdiffs,[r.prefit_cost for r in fitresults],bins=80,norm=colors.LogNorm(),cmap='gray')
        ax[0][0].set_title('prefit cost vs. diff. in exposure time')
        f.colorbar(pos[3],ax=ax[0][0])
        pos = ax[0][1].hist2d(etdiffs,[r.postfit_cost for r in fitresults],bins=80,norm=colors.LogNorm(clip=True),cmap='gray')
        ax[0][1].set_title('postfit cost vs. diff. in exposure time')
        f.colorbar(pos[3],ax=ax[0][1])
        pos = ax[1][0].hist2d(etdiffs,[(r.prefit_cost-r.postfit_cost) for r in fitresults],bins=80,norm=colors.LogNorm(),cmap='gray')
        ax[1][0].set_title('cost redux vs. diff. in exposure time')
        f.colorbar(pos[3],ax=ax[1][0])
        ax[1][0].plot([0.98*x for x in ax[1][0].get_xlim()],[0.,0.],linewidth=2)
        pos = ax[1][1].hist2d(etdiffs,[(r.prefit_cost-r.postfit_cost)/r.prefit_cost for r in fitresults],bins=80,norm=colors.LogNorm(),cmap='gray')
        ax[1][1].set_title('frac. cost redux vs. diff. in exposure time')
        f.colorbar(pos[3],ax=ax[1][1])
        ax[1][1].plot([0.98*x for x in ax[1][1].get_xlim()],[0.,0.],linewidth=2)
        fn = f'cost_reduction_plots_2d_{self.slideID}_layer_{self.layer}.png'
        save_figure_in_dir(plt,fn,self.plotdirpath)

    #helper function to write out a set of overlap overlay comparisons
    def __saveComparisonImages(self,n_comparisons_to_save) :
        et_fit_logger.info(f'Saving comparison images for layer {self.layer}....')
        self.exposure_time_overlaps.sort(key=lambda x: abs(x.et_diff))
        n_ends = min(int(n_comparisons_to_save/3),len(self.exposure_time_overlaps))
        n_random = max(0,min(n_comparisons_to_save-(2*n_ends),len(self.exposure_time_overlaps)-2*n_ends))
        #figure out which overlaps will have their raw comparisons saved
        random_indices = None
        raw_olap_ns_for_plots = []
        if n_ends>0 :
            for eto in self.exposure_time_overlaps[:n_ends] :
                raw_olap_ns_for_plots.append(eto.n)
            for eto in self.exposure_time_overlaps[:-(n_ends+1):-1] :
                raw_olap_ns_for_plots.append(eto.n)
        if n_random>0 and (len(self.exposure_time_overlaps)-(2*n_ends)) :
            random_indices = random.sample(range(n_ends,len(self.exposure_time_overlaps)-n_ends),n_random)
            for ri in random_indices :
                raw_olap_ns_for_plots.append(self.exposure_time_overlaps[ri].n)
        if len(raw_olap_ns_for_plots)<1 :
            return
        #make an alignsample for just those overlaps and correct the raw images
        et_fit_logger.info(f'Making an AlignSample for {len(raw_olap_ns_for_plots)} pre/postfit overlay images for layer {self.layer}')
        a = AlignSampleForExposureTime(self.root_dir,self.rawfile_top_dir,self.slideID,selectoverlaps=raw_olap_ns_for_plots,
                                onlyrectanglesinoverlaps=True,nclip=UNIV_CONST.N_CLIP,readlayerfile=False,layer=self.layer,filetype="raw",smoothsigma=None,flatfield=self.flatfield)
        a.getDAPI()
        raw_olap_images = {}
        et_fit_logger.info(f'Aligning overlaps for plots in layer {self.layer}')
        a.align(alreadyalignedstrategy='overwrite')
        for olap in a.overlaps :
            if olap.result.exit!=0 :
                continue
            raw_p1im, raw_p2im = olap.shifted
            raw_olap_images[olap.n] = {'p1im':raw_p1im,'p2im':raw_p2im}
        #add the raw images to the exposure time overlaps
        for eto in [eto for eto in self.exposure_time_overlaps if eto.n in raw_olap_images.keys()] :
            eto.raw_p1im = raw_olap_images[eto.n]['p1im']
            eto.raw_p2im = raw_olap_images[eto.n]['p2im']
        #make the plots
        et_fit_logger.info(f'Saving pre/postfit overlay images for layer {self.layer}')
        with cd(self.plotdirpath) :
            if n_ends>0 :
                for io,eto in enumerate(self.exposure_time_overlaps[:n_ends],start=1) :
                    eto.saveComparisonImages(self.best_fit_offset,f'overlay_comp_least_diff_{io}')
                for io,eto in enumerate(self.exposure_time_overlaps[:-(n_ends+1):-1],start=1) :
                    eto.saveComparisonImages(self.best_fit_offset,f'overlay_comp_most_diff_{io}')
            if random_indices is not None :
                for ri in random_indices :
                    self.exposure_time_overlaps[ri].saveComparisonImages(self.best_fit_offset,f'overlay_comp_random_{ri+1}')
