#imports
from .flatfield_sample import FlatfieldSample 
from .mean_image import MeanImage
from .utilities import flatfield_logger, chunkListOfFilepaths, readImagesMT, sampleNameFromFilepath
from .config import CONST
from ..utilities.misc import cd
import os, random

#main class
class FlatfieldProducer :
    """
    Main class used in producing the flatfield correction image
    """

    #################### CLASS CONSTANTS ####################
    
    THRESHOLDING_PLOT_DIR_NAME = 'thresholding_info' #name of the directory where the thresholding information will be stored

    #################### INIT FUNCTION ####################
    
    def __init__(self,img_dims,sample_names,all_sample_rawfile_paths_to_run,dbload_top_dir,workingdir_name,skip_masking) :
        """
        img_dims                        = dimensions of images in files in order as (height, width, # of layers) 
        sample_names                    = list of names of samples that will be considered in this run
        all_sample_rawfile_paths_to_run = list of paths to raw files to stack for all samples that will be run
        dbload_top_dir                  = path to the directory holding all of the [samplename]/dbload directories
        workingdir_name                 = name of the directory to save everything in
        skip_masking                    = if True, image layers won't be masked before being added to the stack
        """
        self.all_sample_rawfile_paths_to_run = all_sample_rawfile_paths_to_run
        #make a dictionary to hold all of the separate samples we'll be considering (keyed by name)
        self.flatfield_sample_dict = {}
        for sn in sample_names :
            self.flatfield_sample_dict[sn]=FlatfieldSample(sn,img_dims,os.path.join(dbload_top_dir,sn,'dbload'))
        #Start up a new mean image to use for making the actual flatfield
        self.mean_image = MeanImage(img_dims[0],img_dims[1],img_dims[2],workingdir_name,skip_masking)

    #################### PUBLIC FUNCTIONS ####################

    def readInBackgroundThresholds(self,threshold_file_dir) :
        """
        Function to read in previously-determined background thresholds for each sample
        threshold_file_dir       = directory holding [samplename]_[CONST.THRESHOLD_TEXT_FILE_NAME_STEM] files to read thresholds 
                                   from instead of finding them from the images themselves
        """
        #read each sample's list of background thresholds by layer
        for sn,samp in sorted(self.flatfield_sample_dict.items()) :
            threshold_file_name = f'{sn}_{CONST.THRESHOLD_TEXT_FILE_NAME_STEM}'
            threshold_file_path = os.path.join(threshold_file_dir,threshold_file_name)
            flatfield_logger.info(f'Copying background thresholds from file {threshold_file_path} for sample {sn}...')
            samp.readInBackgroundThresholds(threshold_file_path)

    def findBackgroundThresholds(self,all_sample_rawfile_paths,n_threads) :
        """
        Function to determine, using HPFs that image edges of the tissue in each slide, what thresholds to use for masking out background
        in each layer of each sample
        all_sample_rawfile_paths = list of every rawfile path for every sample that will be run
        n_threads                = max number of threads/processes to open at once
        """
        #make each sample's list of background thresholds by layer
        for sn,samp in sorted(self.flatfield_sample_dict.items()) :
            threshold_file_name = f'{sn}_{CONST.THRESHOLD_TEXT_FILE_NAME_STEM}'
            flatfield_logger.info(f'Finding background thresholds from tissue edges for sample {sn}...')
            samp.findBackgroundThresholds([rfp for rfp in all_sample_rawfile_paths if sampleNameFromFilepath(rfp)==sn],
                                          n_threads,
                                          os.path.join(self.mean_image.workingdir_name,self.THRESHOLDING_PLOT_DIR_NAME),
                                          threshold_file_name,
                                          )

    def stackImages(self,n_threads,selected_pixel_cut,n_masking_images_per_sample,allow_edge_HPFs) :
        """
        Function to mask out background and stack portions of images up
        n_threads                   = max number of threads/processes to open at once
        selected_pixel_cut          = fraction (0->1) of how many pixels must be selected as signal for an image to be stacked
        n_masking_images_per_sample = how many example masking image plots to save for each sample (exactly which are saved will be decided randomly)
        allow_edge_HPFs             = 'True' if HPFs on the edge of the tissue should NOT be removed before stacking sample images
        """
        #do one sample at a time
        for sn,samp in sorted(self.flatfield_sample_dict.items()) :
            flatfield_logger.info(f'Stacking raw images from sample {sn}...')
            #get all the filepaths in this sample
            this_samp_fps_to_run = [fp for fp in self.all_sample_rawfile_paths_to_run if sampleNameFromFilepath(fp)==sn]
            #If they're being neglected, get the filepaths corresponding to HPFs on the edge of the tissue
            if not allow_edge_HPFs :
                this_samp_edge_HPF_filepaths = samp.findTissueEdgeFilepaths(this_samp_fps_to_run)
                flatfield_logger.info(f'Neglecting {len(this_samp_edge_HPF_filepaths)} files on the edge of the tissue')
                this_samp_fps_to_run = [fp for fp in this_samp_fps_to_run if fp not in this_samp_edge_HPF_filepaths]
            #choose which of them will have their masking images saved
            if len(this_samp_fps_to_run)<n_masking_images_per_sample :
                msg=f'WARNING: Requested to save {n_masking_images_per_sample} masking images for each sample,'
                msg+=f' but {sn} will only stack {len(this_samp_fps_to_run)} total files! (Masking plots will be saved for all of them.)'
                flatfield_logger.warn(msg)
            this_samp_indices_for_masking_plots = list(range(len(this_samp_fps_to_run)))
            random.shuffle(this_samp_indices_for_masking_plots)
            this_samp_indices_for_masking_plots=this_samp_indices_for_masking_plots[:n_masking_images_per_sample]
            #break the list of this sample's filepaths into chunks to run in parallel
            filepath_chunks = chunkListOfFilepaths(this_samp_fps_to_run,self.mean_image.dims,n_threads)
            #for each chunk, get the image arrays from the multithreaded function and then add them to to stack
            for fp_chunk in filepath_chunks :
                if len(fp_chunk)<1 :
                    continue
                new_img_arrays = readImagesMT(fp_chunk)
                this_chunk_masking_plot_indices=[fp_chunk.index(fp) for fp in fp_chunk 
                                                 if this_samp_fps_to_run.index(fp[0]) in this_samp_indices_for_masking_plots]
                self.mean_image.addGroupOfImages(new_img_arrays,samp,selected_pixel_cut,this_chunk_masking_plot_indices)

    def makeFlatField(self) :
        """
        Take the mean of the stacked images, smooth it and make the flatfield image by dividing each layer by its mean pixel value
        """
        flatfield_logger.info('Getting/smoothing mean image and making flatfield....')
        self.mean_image.makeFlatFieldImage()

    def applyFlatField(self,flatfield_file_path) :
        """
        Take the mean of the stacked images, smooth it, and make the corrected mean image by dividing the mean image by the existing flatfield
        """
        flatfield_logger.info(f'Applying flatfield at {flatfield_file_path} to mean image....')
        self.mean_image.makeCorrectedMeanImage(flatfield_file_path)

    def writeFileLog(self,filename) :
        """
        Write out a text file of all the filenames that were added
        filename = name of the file to write to
        """
        flatfield_logger.info('Writing filepath text file....')
        with cd(self.mean_image.workingdir_name) :
            with open(filename,'w') as fp :
                for sn,samp in sorted(self.flatfield_sample_dict.items()) :
                    for path in [fp for fp in self.all_sample_rawfile_paths_to_run if sampleNameFromFilepath(fp)==sn] :
                        fp.write(f'{path}\n')

    def writeOutInfo(self) :
        """
        Save layer-by-layer images, some plots, and the list of rawfile paths
        """
        #save the images
        flatfield_logger.info('Saving layer-by-layer images....')
        self.mean_image.saveImages()
        #make some visualizations of the images
        flatfield_logger.info('Saving plots....')
        self.mean_image.savePlots()
