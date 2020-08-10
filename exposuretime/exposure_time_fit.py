#imports
from .utilities import et_fit_logger, UpdateImage, OverlapWithExposureTimes
from .config import CONST
from ..alignment.alignmentset import AlignmentSetFromXML
from ..flatfield.utilities import smoothImageWorker
from ..utilities.tableio import writetable
from ..utilities.misc import cd
import numpy as np, matplotlib.pyplot as plt
import os, copy, random, scipy

#helper class to do the fit in one image layer only
class SingleLayerExposureTimeFit :

    #################### PUBLIC FUNCTIONS ####################

    def __init__(self,layer_n,exposure_times,max_exp_time,sample,rawfile_top_dir,metadata_top_dir,flatfield,overlaps,smoothsigma,cutimages) :
        """
        layer_n          = layer number that this fit will run on (indexed from 1)
        exposure_times   = dictionary of raw file exposure times, keyed by filename stem
        max_exp_time     = the maximum exposure time in the entire sample for this layer
        sample           = name of the microscope data sample to fit to ("M21_1" or equivalent)
        rawfile_top_dir  = path to directory containing [samplename] directory with multilayered ".Data.dat" files in it
        metadata_top_dir = path to directory containing [samplename]/im3/xml directory
        flatfield        = relevant layer of whole sample flatfield file
        overlaps         = list of sample overlaps to run ([-1] if all should be run)
        smoothsigma      = sigma for Gaussian blurring
        cutimages        = True if only central 50% of overlap images should be used
        """
        self.layer = layer_n
        self.max_exp_time = max_exp_time
        self.rawfile_top_dir = rawfile_top_dir
        self.metadata_top_dir = metadata_top_dir
        self.sample = sample
        self.flatfield = flatfield
        self.exposure_time_overlaps = self.__getExposureTimeOverlaps(exposure_times,max_exp_time,overlaps,smoothsigma,cutimages)
        if len(self.exposure_time_overlaps)<1 :
            et_fit_logger.warn(f'WARNING: layer {self.layer} does not have any aligned overlaps with exposure time differences. This fit will be skipped!')
        self.offsets = []
        self.costs = []
        self.best_fit_offset = None
        self.best_fit_cost = None

    def doFit(self,initial_offset,offset_bounds,max_iter,gtol,eps,print_every) :
        """
        Run the fit for this layer
        initial_offset = starting point for fits
        offset_bounds  = bounds for dark current count offset
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
                                              bounds=[offset_bounds],
                                              options={'disp':True,
                                                       'ftol':1e-20,
                                                       'gtol':gtol,
                                                       'eps':eps,
                                                       'maxiter':max_iter,
                                                       'iprint':print_every,
                                                       'maxls':2*len(self.exposure_time_overlaps),
                                                      }
                                             )
        self.best_fit_offset = self.result.x[0]
        self.best_fit_cost = self.result.fun
        msg = f'Layer {self.layer} fit done! Minimization terminated with exit {self.result.message}. '
        msg+= f'Best-fit offset = {self.best_fit_offset:.4f} (best cost = {self.best_fit_cost:.4f})'
        et_fit_logger.info(msg)

    def writeOutResults(self,top_plot_dir,n_comparisons_to_save) :
        """
        Save some post-processing plots for this fit
        top_plot_dir          = path to directory in which this fit's subdirectory should be created
        n_comparisons_to_save = total # of overlap overlay comparisons to write out for each completed fit
        """
        if len(self.exposure_time_overlaps)<1 :
            et_fit_logger.warn(f'WARNING: skipping layer {self.layer} because there are no results to write out!')
            return
        if self.best_fit_offset is None :
            raise RuntimeError('ERROR: best fit offset is None; run fit before calling writeOutResults!')
        #make this fit's plot directory name/path
        plotdirname = f'layer_{self.layer}_plots'
        with cd(top_plot_dir) :
            if not os.path.isdir(plotdirname) :
                os.mkdir(plotdirname)
        self.plotdirpath = os.path.join(top_plot_dir,plotdirname)
        self.__plotCostsAndOffsets()
        self.__writeResultsAndPlotCostReductions()
        self.__saveComparisonImages(n_comparisons_to_save)

        
    #################### PRIVATE HELPER FUNCTIONS ####################

    #helper function to calculate the fitting cost
    def __cost(self,pars) :
        offset=pars[0]
        cost=0.; npix=0.
        for eto in self.exposure_time_overlaps :
            thiscost, thisnpix = eto.getCostAndNPix(offset)
            cost+=thiscost; npix+=thisnpix
        cost/=npix
        self.offsets.append(offset); self.costs.append(cost)
        return cost

    #helper function to return a list of OverlapWithExposureTime objects set up to run on this particular image layer
    def __getExposureTimeOverlaps(self,exposure_times,max_exp_time,overlaps,smoothsigma,cutimages) :
        #first get the list of overlaps that have different p1 and p2 exposure times
        et_fit_logger.info(f'Finding overlaps with different p1 and p2 exposure times in layer {self.layer}....')
        overlaps_with_et_diff = self.__getOverlapsWithExposureTimeDifferences(exposure_times)
        overlaps = overlaps_with_et_diff if overlaps==[-1] else [n for n in overlaps if n in overlaps_with_et_diff]
        if len(overlaps)<1 :
            return []
        #make an alignmentset from the raw files
        et_fit_logger.info(f'Making an AlignmentSet for just the overlaps with different exposure times in layer {self.layer}....')
        a = AlignmentSetFromXML(self.metadata_top_dir,self.rawfile_top_dir,self.sample,selectoverlaps=overlaps,onlyrectanglesinoverlaps=True,
                                nclip=CONST.N_CLIP,readlayerfile=False,layer=self.layer)
        #get all the raw file layers
        a.getDAPI(filetype='raw')
        #correct the rectangle images with the flatfield file and applying some smoothing
        et_fit_logger.info(f'Correcting rectangle images for layer {self.layer}....')
        update_images = []
        for ri,r in enumerate(a.rectangles) :
            rfkey=r.file.rstrip('.im3')
            image = np.rint((r.image*a.meanimage.flatfield)/self.flatfield).astype(np.uint16)
            image = smoothImageWorker(image,smoothsigma)
            update_images.append(UpdateImage(rfkey,image,ri))
        #update and align with the smoothed images
        et_fit_logger.info(f'Updating and aligning layer {self.layer} overlaps with corrected/smoothed images....')
        a.updateRectangleImages(update_images,usewarpedimages=False,correct_with_meanimage=False)
        a.align(alreadyalignedstrategy='overwrite')
        #make the exposure time comparison overlap objects
        etolaps = []
        for io,olap in enumerate(a.overlaps) :
            if olap.result.exit!=0 :
                continue
            p1et = exposure_times[(([r for r in a.rectangles if r.n==olap.p1])[0].file).rstrip(CONST.IM3_EXT)]
            p2et = exposure_times[(([r for r in a.rectangles if r.n==olap.p2])[0].file).rstrip(CONST.IM3_EXT)]
            etolaps.append(OverlapWithExposureTimes(olap,p1et,p2et,max_exp_time,cutimages))
        #return the whole list
        et_fit_logger.info(f'Found {len(etolaps)} overlaps that are aligned and have different p1 and p2 exposure times')
        return etolaps

    #helper function to return a list of overlap ns for overlaps where the p1 and p2 image exposure times are different
    def __getOverlapsWithExposureTimeDifferences(self,exp_times) :
        a = AlignmentSetFromXML(self.metadata_top_dir,self.rawfile_top_dir,self.sample,nclip=CONST.N_CLIP,readlayerfile=False,layer=self.layer)
        olaps_with_et_diffs = []
        for olap in a.overlaps :
            p1et = exp_times[(([r for r in a.rectangles if r.n==olap.p1])[0].file).rstrip('.im3')]
            p2et = exp_times[(([r for r in a.rectangles if r.n==olap.p2])[0].file).rstrip('.im3')]
            if p2et!=p1et :
                olaps_with_et_diffs.append(olap.n)
        return olaps_with_et_diffs

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
        with cd(self.plotdirpath) :
            plt.savefig('costs_and_offsets.png')
        plt.close()

    #helper function to make a plot of each overlap's cost reduction and write out the table of overlap fit results
    def __writeResultsAndPlotCostReductions(self) :
        et_fit_logger.info(f'Writing out fit results for layer {self.layer}....')
        fitresults = []; cost_reduxes = []; frac_cost_reduxes = []
        for eto in self.exposure_time_overlaps :
            fitresult = eto.getFitResult(self.best_fit_offset)
            fitresults.append(fitresult)
            cost_reduxes.append(fitresult.prefit_cost-fitresult.postfit_cost)
            frac_cost_reduxes.append((fitresult.prefit_cost-fitresult.postfit_cost)/(fitresult.prefit_cost))
        f,ax=plt.subplots(1,2,figsize=(2*6.4,4.6))
        ax[0].hist(cost_reduxes,bins=60)
        ax[0].set_xlabel('original cost - post-fit cost')
        ax[0].set_ylabel('number of overlaps')
        ax[1].hist(frac_cost_reduxes,bins=60)
        ax[1].set_xlabel('(original cost - post-fit cost)/(original cost)')
        ax[1].set_ylabel('number of overlaps')
        with cd(self.plotdirpath) :
            plt.savefig(f'cost_reductions.png')
        plt.close()
        with cd(self.plotdirpath) :
            writetable('overlap_fit_results.csv',fitresults)

    #helper function to write out a set of overlap overlay comparisons
    def __saveComparisonImages(self,n_comparisons_to_save) :
        et_fit_logger.info(f'Saving comparison images for layer {self.layer}....')
        self.exposure_time_overlaps.sort(key=lambda x: abs(x.et_diff))
        n_ends = min(int(n_comparisons_to_save/3),len(self.exposure_time_overlaps))
        n_random = max(0,min(n_comparisons_to_save-(2*n_ends),len(self.exposure_time_overlaps)-2*n_ends))
        #figure out which overlaps will have their raw comparisons saved
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
        #make an alignmentset for just those overlaps and correct the raw images
        et_fit_logger.info(f'Making an AlignmentSet for {len(raw_olap_ns_for_plots)} pre/postfit overlay images for layer {self.layer}')
        a = AlignmentSetFromXML(self.metadata_top_dir,self.rawfile_top_dir,self.sample,selectoverlaps=raw_olap_ns_for_plots,
                                onlyrectanglesinoverlaps=True,nclip=CONST.N_CLIP,readlayerfile=False,layer=self.layer)
        et_fit_logger.info(f'Correcting images for plots in layer {self.layer}')
        a.getDAPI(filetype='raw')
        raw_update_images = []
        for ri,r in enumerate(a.rectangles) :
            rfkey=r.file.rstrip('.im3')
            image = np.rint((r.image*a.meanimage.flatfield)/self.flatfield).astype(np.uint16)
            raw_update_images.append(UpdateImage(rfkey,copy.deepcopy(image),ri))
        raw_olap_p1_images = {}; raw_olap_p2_images = {}
        et_fit_logger.info(f'Updating rectangle images and aligning overlaps for plots in layer {self.layer}')
        a.updateRectangleImages(raw_update_images,usewarpedimages=False,correct_with_meanimage=False)
        a.align(alreadyalignedstrategy='overwrite')
        for olap in a.overlaps :
            if olap.result.exit!=0 :
                continue
            raw_p1im, raw_p2im = olap.shifted
            raw_olap_p1_images[olap.n] = raw_p1im; raw_olap_p2_images[olap.n] = raw_p2im
        #add the raw images to the exposure time overlaps
        for eto in [eto for eto in self.exposure_time_overlaps if (eto.n in raw_olap_ns_for_plots) and (eto.n in raw_olap_p1_images.keys())] :
            eto.raw_p1_im = raw_olap_p1_images[eto.n]
            eto.raw_p2_im = raw_olap_p2_images[eto.n]
        #make the plots
        et_fit_logger.info(f'Saving pre/postfit overlay images for layer {self.layer}')
        with cd(self.plotdirpath) :
            if n_ends>0 :
                for io,eto in enumerate(self.exposure_time_overlaps[:n_ends],start=1) :
                    eto.saveComparisonImages(self.best_fit_offset,f'overlay_comparison_least_different_{io}')
                for io,eto in enumerate(self.exposure_time_overlaps[:-(n_ends+1):-1],start=1) :
                    eto.saveComparisonImages(self.best_fit_offset,f'overlay_comparison_most_different_{io}')
            if n_random>0 and (len(self.exposure_time_overlaps)-(2*n_ends)) :
                random_indices = random.sample(range(n_ends,len(self.exposure_time_overlaps)-n_ends),n_random)
                for ri in random_indices :
                    self.exposure_time_overlaps[ri].saveComparisonImages(self.best_fit_offset,f'overlay_comparison_random_{ri+1}')
