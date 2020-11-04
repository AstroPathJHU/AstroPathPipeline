#imports
from .flatfield_slide import FlatfieldSlide 
from .mean_image import MeanImage
from .utilities import flatfield_logger, FlatFieldError, chunkListOfFilepaths, readImagesMT, slideNameFromFilepath, FieldLog
from .utilities import getSlideMeanImageFilepath, getSlideMaskStackFilepath
from .config import CONST
from ..alignment.alignmentset import AlignmentSetFromXML
from ..utilities.img_file_io import getSlideMedianExposureTimesByLayer, LayerOffset
from ..utilities.tableio import readtable, writetable
from ..utilities.misc import cd, MetadataSummary
import os, random, methodtools

#main class
class FlatfieldProducer :
    """
    Main class used in producing the flatfield correction image
    """

    #################### PROPERTIES ####################

    @property
    def exposure_time_correction_offsets(self) :
        return self._et_correction_offsets #the list of offsets to use for exposure time correction in this run

    #################### CLASS CONSTANTS ####################
    
    IMAGE_STACK_MDS_FN_STEM    = 'metadata_summary_stacked_images' #partial filename for the metadata summary file for the stacked images
    FIELDS_USED_STEM           = 'fields_used'                     #partial filename for the field log file to write out

    #################### PUBLIC FUNCTIONS ####################
    
    def __init__(self,slides,all_slide_rawfile_paths_to_run,workingdir_name,skip_et_correction=False,skip_masking=False,logger=None) :
        """
        slides                         = list of FlatfieldSlideInfo objects for this run
        all_slide_rawfile_paths_to_run = list of paths to raw files to stack for all slides that will be run (may be None in batch_flatfield mode)
        workingdir_name                = name of the directory to save everything in
        skip_et_correction             = if True, image flux will NOT be corrected for exposure time differences in each layer
        skip_masking                   = if True, image layers won't be masked before being added to the stack
        logger                         = a RunLogger object whose context is entered, if None the default log will be used
        """
        self.all_slide_rawfile_paths_to_run = all_slide_rawfile_paths_to_run
        #make a dictionary to hold all of the separate slides we'll be considering (keyed by name)
        self.flatfield_slide_dict = {}
        for s in slides :
            self.flatfield_slide_dict[s.name]=FlatfieldSlide(s)
        img_dims = None
        for ff_slide in self.flatfield_slide_dict.values() :
            if img_dims is None :
                img_dims=ff_slide.img_dims
            elif img_dims!=ff_slide.img_dims :
                raise FlatFieldError('ERROR: slides do not all share the same dimensions!')
        #Start up a new mean image to use for making the actual flatfield
        self.mean_image = MeanImage(img_dims,workingdir_name,skip_et_correction,skip_masking)
        #Set up the exposure time correction offsets by layer
        self._et_correction_offsets = []
        for li in range(img_dims[-1]) :
            self._et_correction_offsets.append(None)
        self._metadata_summaries = []
        self._field_logs = []
        #Set up the logger
        self._logger = logger

    def makeFlatfieldFromSlideMeanImages(self) :
        """
        Function to make a flatfield using all the slides' individual mean_image and mask_stack files that already exist
        """
        #first read in all of the individual slides' information
        for sn,slide in sorted(self.flatfield_slide_dict.items()) :
            #get and add this slide's mean image and mask stack
            mifp = getSlideMeanImageFilepath(slide)
            msfp = getSlideMaskStackFilepath(slide)
            self.__writeLog(f'Reading and adding mean image for slide {sn}','info',sn,slide.root_dir)
            self.mean_image.addSlideMeanImageAndMaskStack(mifp,msfp)
            #aggregate the slide's metadata as well
            mds = readtable(os.path.join(os.path.dirname(mifp),f'{self.IMAGE_STACK_MDS_FN_STEM}_{CONST.AUTOMATIC_MEANIMAGE_DIRNAME}.csv'),MetadataSummary)
            self._metadata_summaries+=mds
            fl = readtable(os.path.join(os.path.dirname(mifp),f'{self.FIELDS_USED_STEM}_{CONST.AUTOMATIC_MEANIMAGE_DIRNAME}.csv'))
            self._field_logs+=fl
        #make the meanimage
        self.makeMeanImage()
        #make the flatfield
        self.makeFlatField()

    def readInExposureTimeCorrectionOffsets(self,et_correction_file) :
        """
        Function to read in the offset factors for exposure time corrections from the given directory
        et_correction_file = path to file containing records of LayerOffset objects specifying an offset to use for each layer
        """
        #read in the file and get the offsets by layer
        self.__writeLog(f'Copying exposure time offsets from file {et_correction_file}','info')
        if self._et_correction_offsets[0] is not None :
            raise FlatFieldError('ERROR: calling readInExposureTimeCorrectionOffsets with an offset list already set!')
        layer_offsets_from_file = readtable(et_correction_file,LayerOffset)
        for ln in range(1,len(self._et_correction_offsets)+1) :
            this_layer_offset = [lo.offset for lo in layer_offsets_from_file if lo.layer_n==ln]
            if len(this_layer_offset)==1 :
                self._et_correction_offsets[ln-1]=this_layer_offset[0]
            elif len(this_layer_offset)==0 :
                msg = f'WARNING: LayerOffset file {et_correction_file} does not have an entry for layer {ln}'
                msg+=  ', so that offset will be set to zero!'
                self.__writeLog(msg,'warning')
                self._et_correction_offsets[ln-1]=0.
            else :
                raise FlatFieldError(f'ERROR: more than one entry found in LayerOffset file {et_correction_file} for layer {ln}!')

    def readInBackgroundThresholds(self,threshold_file_dir) :
        """
        Function to read in previously-determined background thresholds for each slide
        threshold_file_dir = directory holding [slidename]_[CONST.THRESHOLD_TEXT_FILE_NAME_STEM] files to read thresholds 
                             from instead of finding them from the images themselves
        """
        #read each slide's list of background thresholds by layer
        for sn,slide in sorted(self.flatfield_slide_dict.items()) :
            threshold_file_name = f'{sn}_{CONST.THRESHOLD_TEXT_FILE_NAME_STEM}'
            threshold_file_path = os.path.join(threshold_file_dir,threshold_file_name)
            self.__writeLog(f'Copying background thresholds from file {threshold_file_path} for slide {sn}...','info',sn,slide.root_dir)
            slide.readInBackgroundThresholds(threshold_file_path,self._logger)

    def findBackgroundThresholds(self,all_slide_rawfile_paths,n_threads) :
        """
        Function to determine, using HPFs that image edges of the tissue in each slide, what thresholds to use for masking out background
        in each layer of each slide
        all_slide_rawfile_paths = list of every rawfile path for every slide that will be run
        n_threads               = max number of threads/processes to open at once
        """
        #make each slide's list of background thresholds by layer
        for sn,slide in sorted(self.flatfield_slide_dict.items()) :
            threshold_file_name = f'{sn}_{CONST.THRESHOLD_TEXT_FILE_NAME_STEM}'
            self.__writeLog(f'Finding background thresholds from tissue edges for slide {sn}','info',sn,slide.root_dir)
            new_field_logs = slide.findBackgroundThresholds([rfp for rfp in all_slide_rawfile_paths if slideNameFromFilepath(rfp)==sn],
                                                           n_threads,
                                                           self.exposure_time_correction_offsets,
                                                           os.path.join(self.mean_image.workingdir_path,CONST.THRESHOLDING_PLOT_DIR_NAME),
                                                           threshold_file_name,
                                                           self._logger
                                                        )
            self._field_logs+=new_field_logs

    def stackImages(self,n_threads,selected_pixel_cut,n_masking_images_per_slide,allow_edge_HPFs) :
        """
        Function to mask out background and stack portions of images up
        n_threads                  = max number of threads/processes to open at once
        selected_pixel_cut         = fraction (0->1) of how many pixels must be selected as signal for an image to be stacked
        n_masking_images_per_slide = how many example masking image plots to save for each slide (exactly which are saved will be decided randomly)
        allow_edge_HPFs            = 'True' if HPFs on the edge of the tissue should NOT be removed before stacking slide images
        """
        #do one slide at a time
        for sn,slide in sorted(self.flatfield_slide_dict.items()) :
            self.__writeLog(f'Stacking raw images from slide {sn}','info',sn,slide.root_dir)
            #get all the filepaths in this slide
            this_slide_fps_to_run = [fp for fp in self.all_slide_rawfile_paths_to_run if slideNameFromFilepath(fp)==sn]
            #If they're being neglected, get the filepaths corresponding to HPFs on the edge of the tissue
            this_slide_edge_HPF_filepaths = slide.findTissueEdgeFilepaths(this_slide_fps_to_run)
            if not allow_edge_HPFs :
                self.__writeLog(f'Neglecting {len(this_slide_edge_HPF_filepaths)} files on the edge of the tissue','info',sn,slide.root_dir)
                this_slide_fps_to_run = [fp for fp in this_slide_fps_to_run if fp not in this_slide_edge_HPF_filepaths]
            #If this slide doesn't have any images to stack, warn the user and continue
            if len(this_slide_fps_to_run)<1 :
                self.__writeLog(f'WARNING: slide {sn} does not have any images to be stacked!','warningglobal',sn,slide.root_dir)
                continue
            #otherwise add the metadata summary for this slide to the producer's list
            a = AlignmentSetFromXML(slide.root_dir,os.path.dirname(os.path.dirname(this_slide_fps_to_run[0])),sn,nclip=CONST.N_CLIP,readlayerfile=False,layer=1)
            this_slide_rect_fn_stems = [os.path.basename(os.path.normpath(fp)).split('.')[0] for fp in this_slide_fps_to_run]
            rect_ts = [r.t for r in a.rectangles if r.file.replace(CONST.IM3_EXT,'') in this_slide_rect_fn_stems]
            self._metadata_summaries.append(MetadataSummary(sn,a.Project,a.Cohort,a.microscopename,str(min(rect_ts)),str(max(rect_ts))))
            #choose which of them will have their masking images saved
            if len(this_slide_fps_to_run)<n_masking_images_per_slide :
                msg=f'WARNING: Requested to save {n_masking_images_per_slide} masking images for each slide,'
                msg+=f' but {sn} will only stack {len(this_slide_fps_to_run)} total files! (Masking plots will be saved for all of them.)'
                self.__writeLog(msg,'warning',sn,slide.root_dir)
            this_slide_indices_for_masking_plots = list(range(len(this_slide_fps_to_run)))
            random.shuffle(this_slide_indices_for_masking_plots)
            this_slide_indices_for_masking_plots=this_slide_indices_for_masking_plots[:n_masking_images_per_slide]
            #get the median exposure times by layer if the images should be normalized
            if (not self.mean_image.skip_et_correction) :
                try :
                    med_exp_times_by_layer = getSlideMedianExposureTimesByLayer(slide.rawfile_top_dir,sn)
                except FileNotFoundError :
                    med_exp_times_by_layer = getSlideMedianExposureTimesByLayer(slide.root_dir,sn)
            else :
                med_exp_times_by_layer = None
            #break the list of this slide's filepaths into chunks to run in parallel
            fileread_chunks = chunkListOfFilepaths(this_slide_fps_to_run,slide.img_dims,slide.root_dir,n_threads)
            #for each chunk, get the image arrays from the multithreaded function and then add them to to stack
            for fr_chunk in fileread_chunks :
                if len(fr_chunk)<1 :
                    continue
                new_field_logs = [FieldLog(sn,fr.rawfile_path,'edge' if fr.rawfile_path in this_slide_edge_HPF_filepaths else 'bulk','stacking') for fr in fr_chunk]
                new_img_arrays = readImagesMT(fr_chunk,
                                              med_exposure_times_by_layer=med_exp_times_by_layer,
                                              et_corr_offsets_by_layer=self.exposure_time_correction_offsets)
                this_chunk_masking_plot_indices=[fr_chunk.index(fr) for fr in fr_chunk 
                                                 if this_slide_fps_to_run.index(fr.rawfile_path) in this_slide_indices_for_masking_plots]
                fields_stacked_in_layers = self.mean_image.addGroupOfImages(new_img_arrays,slide,selected_pixel_cut,med_exp_times_by_layer,
                                                                            this_chunk_masking_plot_indices,self._logger)
                for fi in range(len(new_field_logs)) :
                    new_field_logs[fi].stacked_in_layers = ','.join([str(ln) for ln in fields_stacked_in_layers[fi]])
                self._field_logs+=new_field_logs
        #write out the list of metadata summaries
        with cd(self.mean_image.workingdir_path) :
            writetable(f'{self.IMAGE_STACK_MDS_FN_STEM}_{os.path.basename(os.path.normpath(self.mean_image.workingdir_path))}.csv',self._metadata_summaries)

    def makeMeanImage(self) :
        """
        Take the mean of the stacked images
        """
        self.__writeLog('Creating mean image','info')
        self.mean_image.makeMeanImage(self._logger)

    def makeFlatField(self,batchID=None) :
        """
        Smooth the mean image and make the flatfield image by dividing each layer by its mean pixel value
        """
        self.__writeLog('Creating flatfield image','info')
        self.mean_image.makeFlatFieldImage(self._logger)

    def applyFlatField(self,flatfield_file_path) :
        """
        Take the mean of the stacked images, smooth it, and make the corrected mean image by dividing the mean image by the existing flatfield
        """
        self.__writeLog(f'Applying flatfield at {flatfield_file_path} to mean image','info')
        self.mean_image.makeCorrectedMeanImage(flatfield_file_path)

    def writeFileLog(self,filename) :
        """
        Write out a text file of all the filenames that were added
        filename = name of the file to write to
        """
        self.__writeLog('Writing filepath text file','imageinfo')
        if not os.path.isdir(self.mean_image.workingdir_path) :
            os.mkdir(self.mean_image.workingdir_path)
        with cd(self.mean_image.workingdir_path) :
            with open(filename,'w') as fp :
                for sn,slide in sorted(self.flatfield_slide_dict.items()) :
                    for path in [fp for fp in self.all_slide_rawfile_paths_to_run if slideNameFromFilepath(fp)==sn] :
                        fp.write(f'{path}\n')

    def writeOutInfo(self,batchID=None) :
        """
        Save layer-by-layer images, some plots, and the log of fields used
        batchID = the integer batch ID to append to the outputted file names
        """
        #figure out what to append to the filenames
        batch_or_slide_ID = None
        if batchID is not None :
            batch_or_slide_ID=batchID
        elif len(self.flatfield_slide_dict)==1 :
            batch_or_slide_ID = (list(self.flatfield_slide_dict.values())[0]).name
        #save the images
        self.__writeLog('Saving layer-by-layer images','imageinfo')
        self.mean_image.saveImages(batch_or_slide_ID)
        #make some visualizations of the images
        self.__writeLog('Saving plots','imageinfo')
        self.mean_image.savePlots()
        with cd(self.mean_image.workingdir_path) :
            writetable(f'{self.FIELDS_USED_STEM}_{os.path.basename(os.path.normpath(self.mean_image.workingdir_path))}.csv',self._field_logs)

    #################### PRIVATE HELPER FUNCTIONS ####################

    #helper function to return the slideID and root_dir for the only slide in this run
    #if there is more than one slide it returns None, None
    @methodtools.lru_cache()
    def __getSingleSlideIDAndRootDir(self) :
        if len(self.flatfield_slide_dict)==1 :
            slide = list(self.flatfield_slide_dict.values())[0]
            sid = slide.name
            rd = slide.root_dir
        else :
            sid = None; rd = None
        return sid, rd

    #helper function to handle logging a line
    def __writeLog(self,msg,level,sid=None,rd=None) :
        if self._logger is not None :
            if sid is None and rd is None :
                sid,rd = self.__getSingleSlideIDAndRootDir()
            if level=='info' :
                self._logger.info(msg,sid,rd)
            elif level=='imageinfo' :
                self._logger.imageinfo(msg,sid,rd)
            elif level=='error' :
                self._logger.error(msg,sid,rd)
            elif level=='warningglobal' :
                self._logger.warningglobal(msg,sid,rd)
            elif level=='warning' :
                self._logger.warning(msg,sid,rd)
            elif level=='debug' :
                self._logger.debug(msg,sid,rd)
        else :
            if level=='info' :
                flatfield_logger.info(msg)
            elif level=='imageinfo' :
                flatfield_logger.info(msg)
            elif level=='error' :
                flatfield_logger.error(msg)
            elif level=='warningglobal' :
                flatfield_logger.warn(msg)
            elif level=='warning' :
                flatfield_logger.warn(msg)
            elif level=='debug' :
                flatfield_logger.debug(msg)
