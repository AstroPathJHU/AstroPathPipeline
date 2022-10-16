#imports
import pathlib, methodtools, random
import numpy as np
from ...utilities.config import CONST as UNIV_CONST
from ...utilities.miscfileio import cd
from ...utilities.tableio import readtable, writetable
from ...shared.samplemetadata import MetadataSummary
from ...shared.argumentparser import FileTypeArgumentParser, WorkingDirArgumentParser
from ...shared.image_masking.config import CONST as MASK_CONST
from ...shared.image_masking.utilities import LabelledMaskRegion
from ...shared.image_masking.image_mask import return_new_mask_labelled_regions, save_plots_for_image
from ...shared.overlap import Overlap
from ...shared.sample import WorkflowSample, ParallelSample
from ...shared.sample import ReadCorrectedRectanglesOverlapsIm3MultiLayerFromXML, MaskSampleBase, XMLLayoutReaderTissue
from .config import CONST
from .utilities import get_background_thresholds_and_pixel_hists_for_rectangle_image
from .utilities import RectangleThresholdTableEntry, FieldLog, ThresholdTableEntry
from .latexsummary import ThresholdingLatexSummary, MaskingLatexSummary
from .plotting import plot_tissue_edge_rectangle_locations, plot_image_layer_thresholds_with_histograms
from .plotting import plot_background_thresholds_by_layer, plot_flagged_HPF_locations
from .imagestack import MeanImage

class MeanImageSampleBase(ReadCorrectedRectanglesOverlapsIm3MultiLayerFromXML, XMLLayoutReaderTissue, MaskSampleBase, 
                          ParallelSample, FileTypeArgumentParser, WorkingDirArgumentParser) :
    """
    Base class to use in running the basic MeanImage methods (i.e. not to be used bare in workflows)
    Used as the starting point for actual MeanImageSamples as well as other types of samples that use MeanImage stuff
    """

    #################### PUBLIC FUNCTIONS ####################

    def __init__(self,*args,workingdir=pathlib.Path(UNIV_CONST.MEANIMAGE_DIRNAME),skip_masking=False,**kwargs) :
        #initialize the parent classes
        super().__init__(*args,**kwargs)
        self.__workingdirpath = workingdir
        #if the workingdir arg was just a name, set it to a directory with that name in the default location
        if self.__workingdirpath.name==str(self.__workingdirpath) :
            self.__workingdirpath = self.im3folder/workingdir
        self.__workingdirpath.mkdir(parents=True,exist_ok=True)
        #set some other variables
        self.__skip_masking = skip_masking
        self.field_logs = []
        self.__image_masking_dirpath = None

    def create_or_find_image_masks(self) :
        """
        Sets some variables as to where to find the image masks for the sample
        If they can't be found they will be created
        """
        #set the image masking directory path 
        #by default this is just the default maskfolder in root/slideID/im3/meanimage/image_masking
        #(or the corresponding value from a possibly overwritten maskroot)
        #but if a workingdirectory was given, and the maskroot hasn't been modified,
        #then change it to workingdirectory/image_masking instead
        if ( (self.__workingdirpath.name!=UNIV_CONST.MEANIMAGE_DIRNAME or self.__workingdirpath.parent!=self.im3folder) 
            and self.maskroot==self.root ) :
            self.maskfolder=self.__workingdirpath/CONST.IMAGE_MASKING_SUBDIR_NAME
        self.__image_masking_dirpath = self.maskfolder if not self.__skip_masking else None
        if (self.__image_masking_dirpath is not None) and (not self.__image_masking_dirpath.is_dir()) :
            self.__image_masking_dirpath.mkdir(parents=True)
        self.__use_precomputed_masks = False
        if self.__image_masking_dirpath.is_dir() :
            self.__use_precomputed_masks = self.__dir_has_precomputed_masks(self.__image_masking_dirpath)
        if not self.__use_precomputed_masks :
            self.__create_sample_image_masks()
        else :
            self.logger.debug(f'Will use already-created image masks in {self.__image_masking_dirpath}')

    #################### CLASS VARIABLES + PROPERTIES ####################

    overlaptype = Overlap
    nclip = UNIV_CONST.N_CLIP
    
    @property
    def workingdirpath(self) :
        return self.__workingdirpath
    @property
    def skip_masking(self) :
        return self.__skip_masking
    @property
    def image_masking_dirpath(self) :
        return self.__image_masking_dirpath
    @methodtools.lru_cache()
    @property
    def exposure_time_histograms_and_bins_by_layer_group(self) :
        all_exp_times = {}
        for lgn in self.layer_groups.keys() :
            all_exp_times[lgn] = []
        for r in self.rectangles :
            for lgn,lgb in self.layer_groups.items() :
                all_exp_times[lgn].append(r.allexposuretimes[lgb[0]-1])    
        exp_time_hists_and_bins = {}
        for lgn in self.layer_groups.keys() :
            newhist,newbins = np.histogram(all_exp_times[lgn],bins=60)
            exp_time_hists_and_bins[lgn] = (newhist,newbins)
        return exp_time_hists_and_bins

    @property
    def rectangleextrakwargs(self):
      return {
        **super().rectangleextrakwargs,
        "_DEBUG": False, #need to load images multiple times
      }

    #################### CLASS METHODS ####################

    @classmethod
    def makeargumentparser(cls, **kwargs):
        p = super().makeargumentparser(**kwargs)
        p.add_argument('--skip-masking', action='store_true',
                   help='''Add this flag to entirely skip masking out the background regions of the images 
                           as they get added [use this argument to completely skip the background thresholding 
                           and masking]''')
        return p
    @classmethod
    def initkwargsfromargumentparser(cls, parsed_args_dict):
        #if a working directory was given, and maskroot was not, but the working directory also has parent directories
        #such that the maskroot could be redefined, then redefine the mask root in the parsed arguments
        wd = parsed_args_dict['workingdir']
        if (wd is not None) and (parsed_args_dict['maskroot']==parsed_args_dict['root']) :
            if ( wd.name==UNIV_CONST.MEANIMAGE_DIRNAME and wd.parent.name==UNIV_CONST.IM3_DIR_NAME and 
                                                            wd.parent.parent.name==parsed_args_dict['SlideID'] ) :
                parsed_args_dict['maskroot']=wd
        return {
            **super().initkwargsfromargumentparser(parsed_args_dict),
            'skip_masking': parsed_args_dict.pop('skip_masking'),
        }
    @classmethod
    def defaultunits(cls) :
        return "fast"

    #################### PRIVATE HELPER FUNCTIONS ####################

    def __dir_has_precomputed_masks(self,dirpath) :
        """
        Return True if the given directory has a complete set of mask files needed to run for the slide
        """
        if (dirpath/CONST.LABELLED_MASK_REGIONS_CSV_FILENAME).is_file() :
            lmrs_as_read = readtable(dirpath/CONST.LABELLED_MASK_REGIONS_CSV_FILENAME,LabelledMaskRegion)
            masked_rect_keys = set([lmr.image_key for lmr in lmrs_as_read])
        else :
            masked_rect_keys = set([])
        for r in self.rectangles :
            mfp = dirpath/f'{r.file.rstrip(UNIV_CONST.IM3_EXT)}_{CONST.TISSUE_MASK_FILE_NAME_STEM}'
            if not mfp.is_file() :
                return False
            if r.file.rstrip(UNIV_CONST.IM3_EXT) in masked_rect_keys :
                mfp = dirpath/f'{r.file.rstrip(UNIV_CONST.IM3_EXT)}_{CONST.BLUR_AND_SATURATION_MASK_FILE_NAME_STEM}'
                if not mfp.is_file() :
                    return False
        return True

    def __create_sample_image_masks(self) :
        """
        Find the optimal background thresholds for the sample and use them to create masks for every image
        """
        self.logger.debug(f'Will create masks for all images in {self.SlideID}')
        #start by finding the background thresholds
        background_thresholds = self.__get_background_thresholds()
        #and then create masks for every rectangle's image
        labelled_mask_regions = []
        if (self.njobs is not None) and (self.njobs>1) :
            proc_results = {}
            with self.pool() as pool :
                for ri,r in enumerate(self.rectangles) :
                    msg = f'Creating masks for {r.file.rstrip(UNIV_CONST.IM3_EXT)} '
                    msg+= f'({ri+1} of {len(self.rectangles)})....'
                    self.logger.debug(msg)
                    with r.using_corrected_im3() as im :
                        r_key = (r.n,r.file)
                        ets = self.med_ets if self.et_offset_file is not None else r.allexposuretimes
                        proc_results[r_key] = pool.apply_async(return_new_mask_labelled_regions,
                                                               (im,self.layer_groups,self.brightest_layers,
                                                                r.file.rstrip(UNIV_CONST.IM3_EXT),
                                                                background_thresholds,ets,
                                                                self.__image_masking_dirpath))
                for (rn,rfile),res in proc_results.items() :
                    try :
                        new_lmrs = res.get()
                        labelled_mask_regions+=new_lmrs
                    except Exception as e :
                        warnmsg = f'WARNING: getting image mask for rectangle {rn} ({rfile.rstrip(UNIV_CONST.IM3_EXT)})'
                        warnmsg+= f' failed with the error "{e}" and this rectangle WILL BE SKIPPED when stacking '
                        warnmsg+= 'images in the meanimage!'
                        self.logger.warning(warnmsg)
        #do the same as above except serially
        else :
            for ri,r in enumerate(self.rectangles) :
                msg = f'Creating masks for {r.file.rstrip(UNIV_CONST.IM3_EXT)} ({ri+1} of {len(self.rectangles)})....'
                self.logger.debug(msg)
                try :
                    with r.using_corrected_im3() as im :
                        ets = self.med_ets if self.et_offset_file is not None else r.allexposuretimes
                        new_lmrs=return_new_mask_labelled_regions(im,self.layer_groups,self.brightest_layers,
                                                                  r.file.rstrip(UNIV_CONST.IM3_EXT),
                                                                  background_thresholds,ets,
                                                                  self.__image_masking_dirpath)
                        labelled_mask_regions+=new_lmrs
                except Exception as e :
                    warnmsg = f'WARNING: getting image mask for rectangle {r.n} ({r.file.rstrip(UNIV_CONST.IM3_EXT)})'
                    warnmsg+= f' failed with the error "{e}" and this rectangle WILL BE SKIPPED when stacking images '
                    warnmsg+= 'in the meanimage!'
                    self.logger.warning(warnmsg) 
        #if anything in the sample was masked
        if len(labelled_mask_regions)>0 :
            #write out the table of labelled mask regions
            self.logger.debug('Writing out labelled mask regions')
            with cd(self.__image_masking_dirpath) :
                writetable(f'{CONST.LABELLED_MASK_REGIONS_CSV_FILENAME}',labelled_mask_regions)
            #save some masking plots for images with the largest numbers of masked pixels
            self.__save_set_of_masking_plots(labelled_mask_regions,background_thresholds)
        #make and save the plot of the flagged HPF locations
        plot_flagged_HPF_locations(self.SlideID,self.rectangles,labelled_mask_regions,self.__image_masking_dirpath)
        #write out the latex summary containing all of the image masking plots that were made
        latex_summary = MaskingLatexSummary(self.SlideID,self.__image_masking_dirpath)
        latex_summary.build_tex_file()
        check = latex_summary.compile()
        if check!=0 :
            warnmsg = 'WARNING: failed while compiling thresholding summary LaTeX file into a PDF. '
            warnmsg+= f'tex file will be in {latex_summary.failed_compilation_tex_file_path}'
            self.logger.warning(warnmsg)

    def __get_background_thresholds(self) :
        """
        Return the background thresholds for each image layer, either reading them from an existing file 
        or calculating them from the rectangles on the edges of the tissue
        """
        #first check the working directory for the background threshold file
        threshold_file_path = self.__workingdirpath/f'{self.SlideID}-{CONST.BACKGROUND_THRESHOLD_CSV_FILE_NAME_STEM}'
        #if it's not in the working directory, check in the slide's meanimage directory (if it exists)
        if not threshold_file_path.is_file() :
            other_tfp = self.root/self.SlideID/UNIV_CONST.IM3_DIR_NAME/UNIV_CONST.MEANIMAGE_DIRNAME 
            other_tfp = other_tfp/f'{self.SlideID}-{CONST.BACKGROUND_THRESHOLD_CSV_FILE_NAME_STEM}'
            if other_tfp.is_file() :
                threshold_file_path = other_tfp
        #read the values from the files or find them from the tissue edge rectangles
        if threshold_file_path.is_file() :
            return self.__get_background_thresholds_from_file(threshold_file_path)
        else :
            return self.__find_background_thresholds_from_tissue_edge_images()

    def __get_background_thresholds_from_file(self,threshold_file_path) :
        """
        Return the list of background thresholds found in a given file
        """
        self.logger.info(f'Reading background thresholds for {self.SlideID} from file {threshold_file_path}')
        background_thresholds_read = readtable(threshold_file_path,ThresholdTableEntry)
        background_thresholds_to_return = []
        for li in range(self.nlayersim3) :
            layer_counts_threshold = [t.counts_threshold for t in background_thresholds_read if t.layer_n==li+1]
            if len(layer_counts_threshold)>1 :
                errmsg = f'ERROR: conflicting background thresholds for layer {li+1} listed in {threshold_file_path}'
                raise ValueError(errmsg)
            elif len(layer_counts_threshold)==0 :
                raise ValueError(f'ERROR: no background threshold for layer {li+1} listed in {threshold_file_path}')
            else :
                background_thresholds_to_return.append(layer_counts_threshold[0])
        return background_thresholds_to_return

    def __find_background_thresholds_from_tissue_edge_images(self) :
        """
        Find, write out, and return the list of optimal background thresholds found from the set of images 
        located on the edges of the tissue
        Also makes some plots and datatables in the process
        """
        self.logger.debug(f'Finding background thresholds for {self.SlideID} using images on the edges of the tissue')
        thresholding_plot_dir = self.__workingdirpath/CONST.THRESHOLDING_SUMMARY_PDF_FILENAME.replace('.pdf','_plots')
        #get the list of rectangles that are on the edge of the tissue, plot their locations, 
        #and save a summary of their metadata
        msg = f'Found {len(self.tissue_edge_rects)} images on the edge of the tissue from a set of '
        msg+= f'{len(self.rectangles)} total images'
        self.logger.debug(msg)
        self.logger.debug('Plotting rectangle locations and saving tissue edge HPF MetadataSummary')
        plot_tissue_edge_rectangle_locations(self.rectangles,self.tissue_edge_rects,self.root,self.SlideID,
                                             thresholding_plot_dir)
        edge_rect_ts = [r.t for r in self.tissue_edge_rects]
        mds = MetadataSummary(self.SlideID,self.Project,self.Cohort,self.microscopename,
                              str(min(edge_rect_ts)),str(max(edge_rect_ts)))
        writetable(self.__workingdirpath/f'{self.SlideID}-{CONST.METADATA_SUMMARY_THRESHOLDING_IMAGES_CSV_FILENAME}',
                  [mds])
        #find the optimal thresholds for each tissue edge image, write them out, 
        #and make plots of the thresholds found in each layer
        image_bgts_by_layer,image_hists_by_layer=self.__get_background_thresholds_and_pixel_hists_for_edge_rectangles()
        #choose the best thresholds based on those results and make some plots of the distributions
        self.logger.debug('Finding best thresholds based on those found for individual images')
        chosen_thresholds = []
        for li in range(self.nlayersim3) :
            valid_layer_thresholds = image_bgts_by_layer[:,li][image_bgts_by_layer[:,li]!=0]
            if len(valid_layer_thresholds)<1 :
                errmsg = f"ERROR: not enough image background thresholds were found in layer {li+1} for "
                errmsg+= f"{self.SlideID} and so this slide can't be used"
                raise RuntimeError(errmsg)
            if not self.et_offset_file is None :
                chosen_thresholds.append(ThresholdTableEntry(li+1,int(np.median(valid_layer_thresholds)),
                                                             np.median(valid_layer_thresholds)/self.med_ets[li]))
            else :
                chosen_thresholds.append(ThresholdTableEntry(li+1,int(np.median(valid_layer_thresholds)),-1.))
        writetable(self.__workingdirpath/f'{self.SlideID}-{CONST.BACKGROUND_THRESHOLD_CSV_FILE_NAME_STEM}',
                   chosen_thresholds)
        self.logger.debug('Saving final thresholding plots')
        thresholding_datatable_fp = self.__workingdirpath/f'{self.SlideID}-{CONST.THRESHOLDING_DATA_TABLE_CSV_FILENAME}'
        if thresholding_datatable_fp.is_file() :
            plot_background_thresholds_by_layer(thresholding_datatable_fp,chosen_thresholds,thresholding_plot_dir)
        plot_image_layer_thresholds_with_histograms(image_bgts_by_layer,chosen_thresholds,image_hists_by_layer,
                                                    thresholding_plot_dir)
        #collect plots in a .pdf file
        latex_summary = ThresholdingLatexSummary(self.SlideID,thresholding_plot_dir)
        latex_summary.build_tex_file()
        check = latex_summary.compile()
        if check!=0 :
            warnmsg = 'WARNING: failed while compiling thresholding summary LaTeX file into a PDF. '
            warnmsg+= f'tex file will be in {latex_summary.failed_compilation_tex_file_path}'
            self.logger.warning(warnmsg)
        #return the list of background thresholds in counts
        return [ct.counts_threshold for ct in sorted(chosen_thresholds,key = lambda x:x.layer_n)]

    def __get_background_thresholds_and_pixel_hists_for_edge_rectangles(self) :
        """
        Return arrays of optimal background thresholds found and pixel histograms in every layer for every 
        tissue edge rectangle image
        Will spawn a pool of multiprocessing processes if n_threads is greater than 1
        """
        #start up the lists that will be returned (and the list of datatable entries to write out)
        image_background_thresholds_by_layer = np.zeros((len(self.tissue_edge_rects),self.nlayersim3),dtype=np.uint16)
        tissue_edge_layer_hists = np.zeros((np.iinfo(np.uint16).max+1,self.nlayersim3),dtype=np.uint64)
        rectangle_data_table_entries = []
        #run the thresholding/histogram function in multiple parallel processes
        if (self.njobs is not None) and (self.njobs>1) :
            proc_results = {}; current_image_i = 0
            with self.pool() as pool :
                for ri,r in enumerate(self.tissue_edge_rects) :
                    msg = f'Finding background thresholds for {r.file.rstrip(UNIV_CONST.IM3_EXT)} '
                    msg+= f'({ri+1} of {len(self.tissue_edge_rects)})....'
                    self.logger.debug(msg)
                    with r.using_corrected_im3() as im :
                        r_key = (r.n,r.file)
                        proc_results[r_key] = pool.apply_async(get_background_thresholds_and_pixel_hists_for_rectangle_image,(im,))
                for (rn,rfile),res in proc_results.items() :
                    try :
                        thresholds, hists = res.get()
                        for li,t in enumerate(thresholds) :
                            if self.et_offset_file is not None :
                                rectangle_data_table_entries.append(RectangleThresholdTableEntry(rn,li+1,int(t),
                                                                                                 t/self.med_ets[li]))
                            else :
                                rectangle_data_table_entries.append(RectangleThresholdTableEntry(rn,li+1,int(t),
                                                                                            t/r.allexposuretimes[li]))
                        image_background_thresholds_by_layer[current_image_i,:] = thresholds
                        tissue_edge_layer_hists+=hists
                        self.field_logs.append(FieldLog(self.SlideID,
                                                        rfile.replace(UNIV_CONST.IM3_EXT,UNIV_CONST.RAW_EXT),
                                                        'edge','thresholding'))
                        current_image_i+=1
                    except Exception as e :
                        warnmsg = f'WARNING: finding thresholds for rectangle {rn} ({rfile.rstrip(UNIV_CONST.IM3_EXT)})'
                        warnmsg+= f' failed with the error "{e}" and this rectangle WILL BE SKIPPED when finding '
                        warnmsg+= 'thresholds for the overall slide!'
                        self.logger.warning(warnmsg)
        #run the thresholding/histogram function serially in this current single process
        else :
            current_image_i=0
            for ri,r in enumerate(self.tissue_edge_rects) :
                msg = f'Finding background thresholds for {r.file.rstrip(UNIV_CONST.IM3_EXT)} '
                msg+= f'({ri+1} of {len(self.tissue_edge_rects)})....'
                self.logger.debug(msg)
                try :
                    with r.using_corrected_im3() as im :
                        thresholds, hists = get_background_thresholds_and_pixel_hists_for_rectangle_image(im)
                    for li,t in enumerate(thresholds) :
                        if self.et_offset_file is not None :
                            rectangle_data_table_entries.append(RectangleThresholdTableEntry(r.n,li+1,int(t),
                                                                                             t/self.med_ets[li]))
                        else :
                            rectangle_data_table_entries.append(RectangleThresholdTableEntry(r.n,li+1,int(t),
                                                                                             t/r.allexposuretimes[li]))
                    image_background_thresholds_by_layer[current_image_i,:] = thresholds
                    tissue_edge_layer_hists+=hists
                    self.field_logs.append(FieldLog(self.SlideID,
                                                    r.file.replace(UNIV_CONST.IM3_EXT,UNIV_CONST.RAW_EXT),
                                                    'edge','thresholding'))
                    current_image_i+=1
                except Exception as e :
                    warnmsg = f'WARNING: finding thresholds for rectangle {r.n} ({r.file.rstrip(UNIV_CONST.IM3_EXT)})'
                    warnmsg+= f' failed with error {e} and this rectangle WILL BE SKIPPED when finding thresholds '
                    warnmsg+= 'for the overall slide!'
                    self.logger.warning(warnmsg)
        #write out the data table of all the individual rectangle layer thresholds
        if len(rectangle_data_table_entries)>0 :
            self.logger.debug('Writing out individual image threshold datatable')
            with cd(self.__workingdirpath) :
                writetable(f'{self.SlideID}-{CONST.THRESHOLDING_DATA_TABLE_CSV_FILENAME}',rectangle_data_table_entries)
        return image_background_thresholds_by_layer, tissue_edge_layer_hists

    def __save_set_of_masking_plots(self,labelled_mask_regions,background_thresholds) :
        """
        Figure out which images had the largest numbers of pixels masked due to blur and saturation and recompute 
        their masks to save plots of the process

        labelled_mask_regions = the list of LabelledMaskRegion objects computed for the sample
        background_thresholds = the list of background thresholds in counts by image layer
        """
        #find the images that had the most pixels masked out (up to 10 each for blur and saturation)
        self.logger.debug('Finding images that had the largest numbers of pixels masked due to blur or saturation')
        blur_lmrs = [lmr for lmr in labelled_mask_regions if lmr.reason_flagged==MASK_CONST.BLUR_FLAG_STRING]
        regions_by_n_blurred_pixels = sorted(blur_lmrs,key=lambda x: x.n_pixels,reverse=True)
        sat_lmrs = [lmr for lmr in labelled_mask_regions if lmr.reason_flagged==MASK_CONST.SATURATION_FLAG_STRING]
        regions_by_n_saturated_pixels = sorted(sat_lmrs,key=lambda x: x.n_pixels,reverse=True)
        top_blur_keys = set()
        for lmr in regions_by_n_blurred_pixels :
            top_blur_keys.add(lmr.image_key)
            if len(top_blur_keys)>=10 :
                break
        top_saturation_keys = set()
        for lmr in regions_by_n_saturated_pixels :
            top_saturation_keys.add(lmr.image_key)
            if len(top_saturation_keys)>=10 :
                break
        #if fewer than twenty total plots are being made, add up to ten random rectangles to plot
        random_keys = set()
        if len((top_blur_keys | top_saturation_keys)) < 20 :
            random_rects = random.sample(self.rectangles,
                                         min(10,20-len((top_blur_keys | top_saturation_keys)),len(self.rectangles)))
            random_keys = set([r.file.rstrip(UNIV_CONST.IM3_EXT) for r in random_rects])
        #recompute the masks for those images and write out the masking plots for them
        keys_to_plot = (top_blur_keys | top_saturation_keys | random_keys)
        rects_to_plot = [r for r in self.rectangles if r.file.rstrip(UNIV_CONST.IM3_EXT) in keys_to_plot]
        if (self.njobs is not None) and (self.njobs>1) :
            proc_results = {}
            with self.pool() as pool :
                for ri,r in enumerate(rects_to_plot) :
                    msg = f'Recreating masks for {r.file.rstrip(UNIV_CONST.IM3_EXT)} and saving masking plots '
                    msg+= f'({ri+1} of {len(rects_to_plot)})....'
                    self.logger.debug(msg)
                    with r.using_corrected_im3() as im :
                        r_key = (r.n,r.file)
                        ets = self.med_ets if self.et_offset_file is not None else r.allexposuretimes
                        proc_results[r_key] = pool.apply_async(save_plots_for_image,
                                                               (im,self.layer_groups,self.brightest_layers,
                                                                r.file.rstrip(UNIV_CONST.IM3_EXT),
                                                                background_thresholds,ets,r.allexposuretimes,
                                                                self.exposure_time_histograms_and_bins_by_layer_group,
                                                                self.__image_masking_dirpath))
                for (rn,rfile),res in proc_results.items() :
                    try :
                        res.get()
                    except Exception as e :
                        warnmsg = f'WARNING: saving masking plots for rectangle {rn} '
                        warnmsg+= f'({rfile.rstrip(UNIV_CONST.IM3_EXT)}) failed with the error "{e}"'
                        self.logger.warning(warnmsg)
        #do the same as above except serially
        else :
            for ri,r in enumerate(rects_to_plot) :
                msg = f'Recreating masks for {r.file.rstrip(UNIV_CONST.IM3_EXT)} and saving masking plots '
                msg+= f'({ri+1} of {len(rects_to_plot)})....'
                self.logger.debug(msg)
                try :
                    with r.using_corrected_im3() as im :
                        save_plots_for_image(im,self.layer_groups,self.brightest_layers,
                                             r.file.rstrip(UNIV_CONST.IM3_EXT),background_thresholds,
                                             self.med_ets if self.et_offset_file is not None else r.allexposuretimes,
                                             r.allexposuretimes,self.exposure_time_histograms_and_bins_by_layer_group,
                                             self.__image_masking_dirpath)
                except Exception as e :
                    warnmsg = f'WARNING: saving masking plots for rectangle {r.n} '
                    warnmsg+= f'({r.file.rstrip(UNIV_CONST.IM3_EXT)}) failed with the error "{e}"'
                    self.logger.warning(warnmsg) 
                    raise e

class MeanImageSample(MeanImageSampleBase,WorkflowSample) :
    """
    Main class to handle creating the meanimage for a slide
    """

    #################### PUBLIC FUNCTIONS ####################

    def __init__(self,*args,**kwargs) :
        #initialize the parent classes
        super().__init__(*args,**kwargs)
        #start up the meanimage
        self.__meanimage = MeanImage((self.fheight,self.fwidth,self.nlayersim3),self.logger)

    def inputfiles(self,**kwargs) :
        return [*super().inputfiles(**kwargs),
                *(r.im3file for r in self.rectangles),
               ]

    def run(self) :
        """
        Main "run" function to be looped when entire cohorts are run
        """
        if not self.skip_masking :
            self.create_or_find_image_masks()
        #make the mean image from all of the tissue bulk rectangles
        n_threads = self.njobs if self.njobs is not None else 4
        new_field_logs = self.__meanimage.stack_rectangle_images(self,self.tissue_bulk_rects,self.med_ets,
                                                                 self.image_masking_dirpath,n_threads)
        for fl in new_field_logs :
            fl.slide = self.SlideID
            self.field_logs.append(fl)
        bulk_rect_ts = [r.t for r in self.tissue_bulk_rects]
        if len(bulk_rect_ts)>0 :
            mds = MetadataSummary(self.SlideID,self.Project,self.Cohort,self.microscopename,
                                  str(min(bulk_rect_ts)),str(max(bulk_rect_ts)))
            writetable(self.workingdirpath/f'{self.SlideID}-{CONST.METADATA_SUMMARY_STACKED_IMAGES_CSV_FILENAME}',[mds])
        #create and write out the final mask stack, mean image, and std. error of the mean image
        self.__meanimage.make_mean_image()
        self.__meanimage.write_output(self.SlideID,self.workingdirpath)
        #write out the field log
        if len(self.field_logs)>0 :
            with cd(self.workingdirpath) :
                writetable(CONST.FIELDS_USED_CSV_FILENAME,self.field_logs)

    #################### CLASS VARIABLES + PROPERTIES ####################

    @property
    def workflowkwargs(self) :
        return{**super().workflowkwargs,'skip_masking':self.skip_masking,'workingdir':self.workingdirpath}

    #################### CLASS METHODS ####################

    @classmethod
    def getoutputfiles(cls,SlideID,root,skip_masking,workingdir=None,**otherworkflowkwargs) :
        outputfiles = []
        if workingdir is None:
            meanimagedir = root/SlideID/UNIV_CONST.IM3_DIR_NAME/UNIV_CONST.MEANIMAGE_DIRNAME
        else:
            meanimagedir = workingdir
        outputfiles.append(meanimagedir/f'{SlideID}-{CONST.MEAN_IMAGE_BIN_FILE_NAME_STEM}')
        outputfiles.append(meanimagedir/f'{SlideID}-{CONST.SUM_IMAGES_SQUARED_BIN_FILE_NAME_STEM}')
        outputfiles.append(meanimagedir/f'{SlideID}-{CONST.STD_ERR_OF_MEAN_IMAGE_BIN_FILE_NAME_STEM}')
        #the files below might not actually exist in the case that no images were stacked
        #outputfiles.append(meanimagedir/CONST.FIELDS_USED_CSV_FILENAME)
        #outputfiles.append(meanimagedir/f'{SlideID}-{CONST.METADATA_SUMMARY_STACKED_IMAGES_CSV_FILENAME}')
        #if not skip_masking :
        #    outputfiles.append(meanimagedir/f'{SlideID}-{CONST.MASK_STACK_BIN_FILE_NAME_STEM}')
        return outputfiles
    @classmethod
    def logmodule(cls) : 
        return "meanimage"
    @classmethod
    def workflowdependencyclasses(cls, **kwargs):
        return super().workflowdependencyclasses(**kwargs)

#################### FILE-SCOPE FUNCTIONS ####################

def main(args=None) :
    MeanImageSample.runfromargumentparser(args)

if __name__=='__main__' :
    main()
