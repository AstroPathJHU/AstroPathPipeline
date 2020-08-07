#imports
from .exposure_time_fit import SingleLayerExposureTimeFit
from .utilities import et_fit_logger, LayerOffset
from .config import CONST
from ..utilities.img_file_io import getRawAsHWL, getImageHWLFromXMLFile, getExposureTimesByLayer
from ..utilities.tableio import writetable
from ..utilities.misc import cd
import numpy as np, matplotlib.pyplot as plt, multiprocessing as mp
import os, glob

#helper function to create a fit for a single layer
#can be run in parallel if given a return dictionary
def getExposureTimeFitWorker(layer_n,exposure_times,max_exp_time,sample,rawfile_top_dir,metadata_top_dir,flatfield,overlaps,smoothsigma,cutimages,return_dict=None) :
    fit = SingleLayerExposureTimeFit(layer_n,exposure_times,max_exp_time,sample,rawfile_top_dir,metadata_top_dir,flatfield,overlaps,smoothsigma,cutimages)
    if return_dict is not None :
        return_dict[layer_n] = fit
    else :
        return fit

#main class for fitting to find the optimal offset
class ExposureTimeOffsetFitGroup :

    #################### PUBLIC FUNCTIONS ####################

    def __init__(self,sample_name,rawfile_top_dir,metadata_top_dir,workingdir_name,layers,n_threads) :
        """
        sample_name      = name of the microscope data sample to fit to ("M21_1" or equivalent)
        rawfile_top_dir  = path to directory containing [samplename] directory with multilayered ".Data.dat" files in it
        metadata_top_dir = path to directory containing [samplename]/im3/xml directory
        working_dir_name = path to some local directory to store files produced by the WarpFitter
        layers           = list of layer numbers to find offsets for
        n_threads        = max number of processes to open at once for stuff that's parallelized
        """
        self.sample = sample_name
        self.rawfile_top_dir = rawfile_top_dir
        self.metadata_top_dir = metadata_top_dir
        self.workingdir_name = workingdir_name
        self.layers = self.__getLayers(layers)
        self.n_threads = n_threads

    def prepFits(self,) :
        """
        Load all of the raw file layers and their exposure times, correct the images with the flatfield and smooth them, 
        align the overlaps, and prep the fits
        
        """
        

    def runFits(self,flatfield_filepath,overlaps,smoothsigma,cutimages,initial_offset,offset_bounds,max_iter,gtol,eps,print_every,n_comparisons_to_save) :
        """
        Run all of the fits
        flatfield_filepath    = path to flatfield file to use in correcting raw image illumination
        overlaps              = list of overlap numbers to consider (should really only use this for testing)
        smoothsigma           = sigma for Gaussian blurring to apply to images
        cutimages             = True if only the central regions of the images should be considered
        initial_offset        = starting point for fits
        offset_bounds         = bounds for dark current count offset
        max_iter              = maximum number of iterations for each fit to run
        gtol                  = gradient projection tolerance for fits
        eps                   = step size for approximating Jacobian
        print_every           = how often to print during minimization
        n_comparisons_to_save = total # of overlap overlay comparisons to write out for each completed fit
        """
        #first get all of the raw image exposure times, and the maximum exposure times in each layer
        all_exposure_times, max_exp_times_by_layer = self.__getExposureTimes()
        #next get the flatfield to use
        self.flatfield = self.__getFlatfield(flatfield_filepath)
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
                batch_fits = self.__getBatchFits(layer_batch,li_start,all_exposure_times,max_exp_times_by_layer,overlaps,smoothsigma,cutimages)
                et_fit_logger.info(f'Done preparing fits in batch {bi} (of {len(layer_batches)}).')
                et_fit_logger.info('Running fits....')
                for fit in batch_fits :
                    fit.doFit(initial_offset,offset_bounds,max_iter,gtol,eps,print_every)
                et_fit_logger.info(f'Done running fits in batch {bi} (of {len(layer_batches)}).')
                et_fit_logger.info('Writing output....')
                procs = []
                for fit in batch_fits :
                    p = mp.Process(target=fit.writeOutResults,
                                   args=(self.workingdir_name,n_comparisons_to_save)
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
                fit = getExposureTimeFitWorker(ln,this_layer_all_exposure_times,max_exp_times_by_layer[li],
                                               self.sample,self.rawfile_top_dir,self.metadata_top_dir,self.flatfield[:,:,ln-1],
                                               overlaps,smoothsigma,cutimages)
                et_fit_logger.info(f'Running fit for layer {ln} ({li+1} of {len(self.layers)})....')
                fit.doFit(initial_offset,offset_bounds,max_iter,gtol,eps,print_every)
                et_fit_logger.info(f'Writing output for layer {ln} ({li+1} of {len(self.layers)})....')
                fit.writeOutResults(self.workingdir_name,n_comparisons_to_save)
                if fit.best_fit_offset is not None :
                    offsets.append(LayerOffset(fit.layer,len(fit.exposure_time_overlaps),fit.best_fit_offset,fit.best_fit_cost))
        #write out all the results
        with cd(self.workingdir_name) :
            writetable(f'{self.sample}_best_fit_offsets_first_layer_{self.layers[0]}.csv',offsets)
        #save the plot of the offsets by layer
        plt.plot([o.layer_n for o in offsets],[o.offset for o in offsets],marker='*')
        plt.xlabel('image layer')
        plt.ylabel('best-fit offset')
        with cd(self.workingdir_name) :
            plt.savefig(f'{self.sample}_best_fit_offsets_by_layer.png')
        plt.close()
        et_fit_logger.info('All fits finished.')

    #################### PRIVATE HELPER FUNCTIONS ####################

    #helper function to get the flatfield from the given arguments
    def __getFlatfield(self,flatfield_filepath) :
        img_dims = getImageHWLFromXMLFile(self.metadata_top_dir,self.sample)
        if flatfield_filepath is None :
            et_fit_logger.warn('WARNING: No flatfield file path specified; corrections will not be applied!')
            return np.ones(img_dims,dtype=CONST.FLATFIELD_DTYPE)
        else :
            return getRawAsHWL(flatfield_filepath,*(img_dims),CONST.FLATFIELD_DTYPE)

    #helper function to return the list of layer numbers to run from the given arguments
    def __getLayers(self,layers) :
        _,_,nlayers = getImageHWLFromXMLFile(self.metadata_top_dir,self.sample)
        if len(layers)==1 and layers[0]==-1 :
            return list(range(1,nlayers+1))
        else :
            for l in layers :
                if not l in range(1,nlayers+1) :
                    raise ValueError(f'ERROR: requested layers {layers} but images in {self.sample} have {nlayers} layers!')
            return layers

    #helper function to get the dictionary of all the image exposure times keyed by the stem of the file name and the list of maximum times by layer
    def __getExposureTimes(self) :
        et_fit_logger.info('Getting all image exposure times....')
        _,_,nlayers = getImageHWLFromXMLFile(self.metadata_top_dir,self.sample)
        with cd(os.path.join(self.rawfile_top_dir,self.sample)) :
            all_rfps = [os.path.join(self.rawfile_top_dir,self.sample,fn) for fn in glob.glob(f'*{CONST.RAW_EXT}')]
        exp_times = {}; max_exp_times = [0 for ln in self.layers]
        for rfp in all_rfps :
            rfs = os.path.basename(rfp).rstrip(CONST.RAW_EXT)
            exp_times[rfs] = []
            all_layer_exposure_times = getExposureTimesByLayer(rfp,nlayers,self.metadata_top_dir)
            for li,ln in enumerate(self.layers) :
                exp_times[rfs].append(all_layer_exposure_times[ln-1])
                if all_layer_exposure_times[ln-1] > max_exp_times[li] :
                    max_exp_times[li] = all_layer_exposure_times[ln-1]
        for li,ln in enumerate(self.layers) :
            this_layer_ets = [exp_times[rfs][li] for rfs in exp_times.keys()]
            plt.hist(this_layer_ets)
            plt.title(f'{self.sample} layer {ln} exposure times')
            plt.xlabel('exposure time (ms)')
            plt.ylabel('HPF count')
            with cd(self.workingdir_name) :
                plt.savefig(f'exposure_times_{self.sample}_layer_{ln}.png')
            plt.close()
        return exp_times, max_exp_times

    #helper function to set up and return a list of single-layer fit objects
    def __getBatchFits(self,layer_batch,li_start,all_exposure_times,max_exp_times_by_layer,overlaps,smoothsigma,cutimages) :
        batch_fits = []
        manager = mp.Manager()
        return_dict = manager.dict()
        procs = []
        for li,ln in enumerate(layer_batch,start=li_start) :
            et_fit_logger.info(f'Setting up fit for layer {ln} ({li+1-li_start} of {len(layer_batch)} in this batch)....')
            this_layer_all_exposure_times = {}
            for rfs in all_exposure_times.keys() :
                this_layer_all_exposure_times[rfs] = all_exposure_times[rfs][li]
            p = mp.Process(target=getExposureTimeFitWorker, 
                           args=(ln,this_layer_all_exposure_times,max_exp_times_by_layer[li],
                                 self.sample,self.rawfile_top_dir,self.metadata_top_dir,self.flatfield[:,:,ln-1],
                                 overlaps,smoothsigma,cutimages,return_dict)
                          )
            procs.append(p)
            p.start()
        for proc in procs:
            proc.join()
            procs = []
        for ln in layer_batch :
            batch_fits.append(return_dict[ln])
        return batch_fits

