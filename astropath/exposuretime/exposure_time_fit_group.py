#imports
from .exposure_time_fit import SingleLayerExposureTimeFit
from .utilities import et_fit_logger, getFirstLayerInGroup, getOverlapsWithExposureTimeDifferences
from ..utilities.img_file_io import getRawAsHWL, getImageHWLFromXMLFile, getExposureTimesByLayer, LayerOffset
from ..utilities.tableio import writetable
from ..utilities.misc import cd, cropAndOverwriteImage
from ..utilities.config import CONST as UNIV_CONST
import numpy as np, matplotlib.pyplot as plt, multiprocessing as mp
import os, glob

#helper function to create a fit for a single layer
#can be run in parallel if given a return dictionary
def getExposureTimeFitWorker(layer_n,exposure_times,med_exp_time,top_plotdir,slideID,rawfile_top_dir,root_dir,flatfield,min_frac,overlaps,smoothsigma,cutimages,return_dict=None) :
    fit = SingleLayerExposureTimeFit(layer_n,exposure_times,med_exp_time,top_plotdir,slideID,rawfile_top_dir,root_dir,flatfield,min_frac,overlaps,smoothsigma,cutimages)
    if return_dict is not None :
        return_dict[layer_n] = fit
    else :
        return fit

#main class for fitting to find the optimal offset
class ExposureTimeOffsetFitGroup :

    #################### PUBLIC FUNCTIONS ####################

    def __init__(self,slide_ID,rawfile_top_dir,root_dir,workingdir_name,layers,n_threads) :
        """
        slide_ID         = name of the microscope slide to use ("M21_1" or equivalent)
        rawfile_top_dir  = path to directory containing [slideID] directory with multilayered ".Data.dat" files in it
        root_dir         = path to Clinical_Specimen directory for this slide
        working_dir_name = path to some local directory to store files produced by the WarpFitter
        layers           = list of layer numbers to find offsets for
        n_threads        = max number of processes to open at once for stuff that's parallelized
        """
        self.slideID = slide_ID
        self.rawfile_top_dir = rawfile_top_dir
        self.root_dir = root_dir
        self.workingdir_name = workingdir_name
        self.img_dims = getImageHWLFromXMLFile(self.rawfile_top_dir,self.slideID)
        self.layers = self.__getLayers(layers)
        self.n_threads = n_threads

    def runFits(self,ff_filepath,overlaps,smoothsigma,wholeimages,initial_offset,min_frac,max_iter,gtol,eps,print_every,n_comparisons_to_save,allow_edges) :
        """
        Run all of the fits
        ff_filepath           = path to flatfield file to use in correcting raw image illumination
        overlaps              = list of overlap numbers to consider (should really only use this for testing)
        smoothsigma           = sigma for Gaussian blurring to apply to images
        wholeimages           = True if the whole image (nor just the central regions) should be considered
        initial_offset        = starting point for fits
        min_frac              = some image in the dataset must have at least this fraction of pixels with the maximum offset (prevents too low of a maximum)
        max_iter              = maximum number of iterations for each fit to run
        gtol                  = gradient projection tolerance for fits
        eps                   = step size for approximating Jacobian
        print_every           = how often to print during minimization
        n_comparisons_to_save = total # of overlap overlay comparisons to write out for each completed fit
        allow_edges           = True if overlaps containing rectangles on the edges of the tissue should be considered
        """
        cutimages = (not wholeimages)
        #first get all of the raw image exposure times, and the median exposure times in each layer
        all_exposure_times, med_ets_by_layer = self.__getExposureTimes()
        #next get the flatfield to use
        self.flatfield = self.__getFlatfield(ff_filepath)
        #lastly, find the list of overlaps with different exposure times for each layer group
        overlaps_to_use_by_layer_group = self.__getOverlapsToUseByLayerGroup(all_exposure_times,overlaps,allow_edges)
        #prep, run, and save the output of all the fits
        offsets = []
        if self.n_threads > 1 :
            layer_batches = []; layer_batches.append([])
            for ln in self.layers :
                if len(layer_batches[-1])>=self.n_threads :
                    layer_batches.append([])
                layer_batches[-1].append(ln)
            for bi,layer_batch in enumerate(layer_batches,start=1) :
                li_start = (bi-1)*self.n_threads
                batch_fits = self.__getBatchFits(layer_batch,li_start,all_exposure_times,med_ets_by_layer,min_frac,overlaps_to_use_by_layer_group,
                                                 smoothsigma,cutimages)
                et_fit_logger.info(f'Done preparing fits in batch {bi} (of {len(layer_batches)}).')
                et_fit_logger.info('Running fits....')
                for fit in batch_fits :
                    fit.doFit(initial_offset,max_iter,gtol,eps,print_every)
                et_fit_logger.info(f'Done running fits in batch {bi} (of {len(layer_batches)}).')
                et_fit_logger.info('Writing output....')
                procs = []
                for fit in batch_fits :
                    p = mp.Process(target=fit.writeOutResults,
                                   args=(n_comparisons_to_save,)
                                  )
                    procs.append(p)
                    p.start()
                for proc in procs :
                    proc.join()
                for fit in batch_fits :
                    if fit.best_fit_offset is not None :
                        offsets.append(LayerOffset(fit.layer,len(fit.exposure_time_overlaps),fit.best_fit_offset,fit.best_fit_cost))
        else :
            for li,ln in enumerate(self.layers) :
                et_fit_logger.info(f'Setting up fit for layer {ln} ({li+1} of {len(self.layers)})....')
                this_layer_all_exposure_times = {}
                for rfs in all_exposure_times.keys() :
                    this_layer_all_exposure_times[rfs] = all_exposure_times[rfs][li]
                this_layer_overlaps = overlaps_to_use_by_layer_group[getFirstLayerInGroup(ln,self.img_dims[-1])]
                fit = getExposureTimeFitWorker(ln,this_layer_all_exposure_times,med_ets_by_layer[li],
                                               self.workingdir_name,self.slideID,self.rawfile_top_dir,self.root_dir,self.flatfield[:,:,ln-1],
                                               min_frac,this_layer_overlaps,smoothsigma,cutimages)
                et_fit_logger.info(f'Running fit for layer {ln} ({li+1} of {len(self.layers)})....')
                fit.doFit(initial_offset,max_iter,gtol,eps,print_every)
                et_fit_logger.info(f'Writing output for layer {ln} ({li+1} of {len(self.layers)})....')
                fit.writeOutResults(n_comparisons_to_save)
                if fit.best_fit_offset is not None :
                    offsets.append(LayerOffset(fit.layer,len(fit.exposure_time_overlaps),fit.best_fit_offset,fit.best_fit_cost))
        #write out all the results
        with cd(self.workingdir_name) :
            all_results_fn = f'{self.slideID}_layers_{self.layers[0]}-{self.layers[-1]}_'
            all_results_fn+= f'{UNIV_CONST.LAYER_ET_OFFSET_FILE_NAME_STEM}_{os.path.basename(os.path.normpath(self.workingdir_name))}.csv'
            writetable(all_results_fn,offsets)
        #save the plot of the offsets by layer
        plt.plot([o.layer_n for o in offsets],[o.offset for o in offsets],marker='*')
        plt.xlabel('image layer')
        plt.ylabel('best-fit offset')
        with cd(self.workingdir_name) :
            fn = f'{self.slideID}_best_fit_offsets_by_layer.png'
            plt.savefig(fn)
            plt.close()
            cropAndOverwriteImage(fn)
        et_fit_logger.info('All fits finished.')

    #################### PRIVATE HELPER FUNCTIONS ####################

    #helper function to get the list of overlaps to use in each layer group (those with different exposure times)
    def __getOverlapsToUseByLayerGroup(self,all_exp_times,overlaps,allow_edges) :
        et_fit_logger.info('Finding overlaps with different exposure times in each layer group....')
        layer_group_ns = set()
        for layer in self.layers :
            layer_group_ns.add(getFirstLayerInGroup(layer,self.img_dims[-1]))
        overlap_dict = {}
        if self.n_threads>1 :
            manager=mp.Manager()
            rdict = manager.dict()
            procs = []
            for layer_n in layer_group_ns :
                this_layer_all_exp_times = {}
                for rfs in all_exp_times.keys() :
                    this_layer_all_exp_times[rfs] = all_exp_times[rfs][self.layers.index(layer_n)]
                p=mp.Process(target=getOverlapsWithExposureTimeDifferences,
                             args=(self.rawfile_top_dir,self.root_dir,self.slideID,this_layer_all_exp_times,layer_n,overlaps,allow_edges,rdict)
                             )
                procs.append(p)
                p.start()
                if len(procs)>=self.n_threads :
                    for proc in procs :
                        proc.join()
                        procs = []
            for proc in procs :
                proc.join()
            for layer_n in layer_group_ns :
                overlap_dict[layer_n] = rdict[layer_n]
        else :
            for layer_n in layer_group_ns :
                this_layer_all_exp_times = {}
                for rfs in all_exp_times.keys() :
                    this_layer_all_exp_times[rfs] = all_exp_times[rfs][self.layers.index(layer_n)]
                overlap_dict[layer_n] = getOverlapsWithExposureTimeDifferences(self.rawfile_top_dir,self.root_dir,self.slideID,
                                                                               this_layer_all_exp_times,layer_n,overlaps,allow_edges)
        return overlap_dict

    #helper function to get the flatfield from the given arguments
    def __getFlatfield(self,flatfield_filepath) :
        if flatfield_filepath is None :
            et_fit_logger.warn('WARNING: No flatfield file path specified; corrections will not be applied!')
            return np.ones(self.img_dims,dtype=UNIV_CONST.FLATFIELD_IMAGE_DTYPE)
        else :
            return getRawAsHWL(flatfield_filepath,*(self.img_dims),UNIV_CONST.FLATFIELD_IMAGE_DTYPE)

    #helper function to return the list of layer numbers to run from the given arguments
    def __getLayers(self,layers) :
        if len(layers)==1 and layers[0]==-1 :
            return list(range(1,self.img_dims[-1]+1))
        else :
            for l in layers :
                if not l in range(1,self.img_dims[-1]+1) :
                    raise ValueError(f'ERROR: requested layers {layers} but images in {self.slideID} have {self.img_dims[-1]} layers!')
            return layers

    #helper function to get the dictionary of all the image exposure times keyed by the stem of the file name and the list of median times by layer
    def __getExposureTimes(self) :
        et_fit_logger.info('Getting all image exposure times....')
        with cd(os.path.join(self.rawfile_top_dir,self.slideID)) :
            all_rfps = [os.path.join(self.rawfile_top_dir,self.slideID,fn) for fn in glob.glob(f'*{UNIV_CONST.RAW_EXT}')]
        #get the dictionary of exposure times keyed by raw file stem
        exp_times = {}
        for rfp in all_rfps :
            rfs = os.path.basename(rfp).rstrip(UNIV_CONST.RAW_EXT)
            exp_times[rfs] = []
            all_layer_exposure_times = getExposureTimesByLayer(rfp,self.rawfile_top_dir)
            for li,ln in enumerate(self.layers) :
                exp_times[rfs].append(all_layer_exposure_times[ln-1])
        #make the list of median exposure times
        med_exp_times = []
        for li in range(len(self.layers)) :
            med_exp_times.append(np.median(np.array([exp_times[rfs][li] for rfs in exp_times.keys()])))
        #plot the exposure times with their medians
        for li,ln in enumerate(self.layers) :
            this_layer_ets = [exp_times[rfs][li] for rfs in exp_times.keys()]
            f, ax = plt.subplots()
            ax.hist(this_layer_ets,label='exposure times')
            ax.plot([med_exp_times[li],med_exp_times[li]],[0.8*y for y in ax.get_ylim()],color='k',linewidth=2,label='median')
            ax.set_title(f'{self.slideID} layer {ln} exposure times')
            ax.set_xlabel('exposure time (ms)')
            ax.set_ylabel('HPF count')
            ax.legend(loc='best')
            with cd(self.workingdir_name) :
                fn = f'exposure_times_{self.slideID}_layer_{ln}.png'
                plt.savefig(fn)
                plt.close()
                cropAndOverwriteImage(fn)
        return exp_times, med_exp_times

    #helper function to set up and return a list of single-layer fit objects
    def __getBatchFits(self,layer_batch,li_start,all_exposure_times,med_ets_by_layer,min_frac,overlaps_by_layer_group,smoothsigma,cutimages) :
        batch_fits = []
        manager = mp.Manager()
        return_dict = manager.dict()
        procs = []
        for li,ln in enumerate(layer_batch,start=li_start) :
            et_fit_logger.info(f'Setting up fit for layer {ln} ({li+1-li_start} of {len(layer_batch)} in this batch)....')
            this_layer_all_exposure_times = {}
            for rfs in all_exposure_times.keys() :
                this_layer_all_exposure_times[rfs] = all_exposure_times[rfs][li]
            this_layer_overlaps = overlaps_by_layer_group[getFirstLayerInGroup(ln,self.img_dims[-1])]
            p = mp.Process(target=getExposureTimeFitWorker, 
                           args=(ln,this_layer_all_exposure_times,med_ets_by_layer[li],
                                 self.workingdir_name,self.slideID,self.rawfile_top_dir,self.root_dir,self.flatfield[:,:,ln-1],
                                 min_frac,this_layer_overlaps,smoothsigma,cutimages,return_dict)
                          )
            procs.append(p)
            p.start()
        for proc in procs:
            proc.join()
            procs = []
        for ln in layer_batch :
            batch_fits.append(return_dict[ln])
        return batch_fits

