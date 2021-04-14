#imports
from .utilities import flatfield_logger, FlatFieldError, chunkListOfFilepaths, getImageLayerHistsMT, findLayerThresholds, FieldLog
from .plotting import plotFlaggedHPFLocations
from ..image_masking.config import CONST as MASKING_CONST
from ...slides.align.alignsample import AlignSampleFromXML
from ...utilities import units
from ...utilities.img_file_io import getImageHWLFromXMLFile, getSlideMedianExposureTimesByLayer, getExposureTimeHistogramsByLayerGroupForSlide
from ...utilities.tableio import writetable
from ...utilities.misc import cd, MetadataSummary, getAlignSampleTissueEdgeRectNs, cropAndOverwriteImage
from ...utilities.config import CONST as UNIV_CONST
import numpy as np, matplotlib.pyplot as plt, matplotlib.image as mpimg, multiprocessing as mp
import pathlib, glob

units.setup('fast')

#main class definition
class FlatfieldSlide() :
    """
    Main class for organizing properties of a particular slide as they pertain to flatfielding
    """

    #################### PROPERTIES ####################

    @property
    def name(self):
        return self._name # the name of the slide
    @property
    def rawfile_top_dir(self):
        return self._rawfile_top_dir # location of raw files for all slides
    @property
    def root_dir(self):
        return self._root_dir # location of Clinical_Specimen directory for all slides
    @property
    def img_dims(self):
        return self._img_dims # dimensions of the images in the slide
    @property
    def exp_time_hists(self):
        return self._exp_time_hists # histograms of the exposure times by layer group
    @property
    def med_exp_times_by_layer(self):
        return self._med_exp_times_by_layer # list of median exposure times by layer
    @property
    def background_thresholds_for_masking(self):
        return self._background_thresholds_for_masking # the list of background thresholds by layer

    #################### CLASS CONSTANTS ####################

    RECTANGLE_LOCATION_PLOT_STEM  = 'rectangle_locations'           #stem for the name of the rectangle location reference plot
    THRESHOLD_PLOT_DIR_STEM       = 'thresholding_plots'            #stem for the name of the thresholding plot dir for this slide
    TISSUE_EDGE_MDS_STEM          = 'metadata_summary_tissue_edges' #stem for the metadata summary file from the tissue edge files only

    #################### PUBLIC FUNCTIONS ####################

    def __init__(self,slide) :
        """
        slide = FlatfieldSlideInfo object for this slide
        """
        self._name = slide.name
        self._rawfile_top_dir = slide.rawfile_top_dir
        self._root_dir = slide.root_dir
        try :
            self._img_dims = getImageHWLFromXMLFile(slide.rawfile_top_dir,slide.name)
        except FileNotFoundError :
            self._img_dims = getImageHWLFromXMLFile(slide.root_dir,slide.name)
        self._exp_time_hists = getExposureTimeHistogramsByLayerGroupForSlide(self._rawfile_top_dir,self._name)
        self._med_exp_times_by_layer = getSlideMedianExposureTimesByLayer(self._rawfile_top_dir,self._name)
        self._background_thresholds_for_masking = None

    def readInBackgroundThresholds(self,threshold_file_path) :
        """
        Function to read in background threshold values from the output file of a previous run with this slide
        threshold_file_path = path to threshold value file
        """
        if self._background_thresholds_for_masking is not None :
            raise FlatFieldError('ERROR: calling readInBackgroundThresholds with non-empty thresholds list')
        self._background_thresholds_for_masking=[]
        with open(threshold_file_path,'r') as tfp :
            all_lines = [l.rstrip() for l in tfp.readlines()]
            for line in all_lines :
                try :
                    self._background_thresholds_for_masking.append(int(line))
                except ValueError :
                    pass
        if not len(self._background_thresholds_for_masking)==self._img_dims[-1] :
            raise FlatFieldError(f'ERROR: number of background thresholds read from {threshold_file_path} is not equal to the number of image layers!')

    def findBackgroundThresholds(self,rawfile_paths,n_threads,et_correction_offsets,top_plotdir_path,threshold_file_name,logger=None) :
        """
        Function to determine this slide's background pixel flux thresholds per layer
        rawfile_paths         = a list of the rawfile paths to consider for this slide's background threshold calculations
        n_threads             = max number of threads/processes to open at once
        et_correction_offsets = list of offsets by layer to use for exposure time correction
        top_plotdir_path      = path to the directory in which to save plots from the thresholding process
        threshold_file_name   = name of file to save background thresholds in, one line per layer
        logger                = a RunLogger object whose context is entered, if None the default log will be used
        """
        #make sure the plot directory exists
        if not pathlib.Path.is_dir(pathlib.Path(top_plotdir_path)) :
            with cd((pathlib.Path(top_plotdir_path)).name) :
                pathlib.Path.mkdir(pathlib.Path(pathlib.path(top_plotdir_path).name))
        this_slide_threshold_plotdir_name = f'{self._name}_{self.THRESHOLD_PLOT_DIR_STEM}'
        plotdir_path = pathlib.Path(f'{top_plotdir_path}/{this_slide_threshold_plotdir_name}')
        if not pathlib.Path.is_dir(plotdir_path) :
            with cd(top_plotdir_path) :
                pathlib.Path.mkdir(pathlib.Path(this_slide_threshold_plotdir_name))
        #first find the filepaths corresponding to the edges of the tissue in the slides
        self.__writeLogInfo(logger,f'Finding tissue edge HPFs for slide {self._name}')
        tissue_edge_filepaths = self.findTissueEdgeFilepaths(rawfile_paths,plotdir_path)
        #chunk them together to be read in parallel
        tissue_edge_fr_chunks = chunkListOfFilepaths(tissue_edge_filepaths,self._img_dims,self._root_dir,n_threads)
        #make histograms of all the tissue edge rectangle pixel fluxes per layer
        self.__writeLogInfo(logger,f'Getting raw tissue edge images to determine thresholds for slide {self._name}')
        nbins=np.iinfo(np.uint16).max+1
        all_image_thresholds_by_layer = np.empty((self._img_dims[-1],len(tissue_edge_filepaths)),dtype=np.uint16)
        all_tissue_edge_layer_hists = np.zeros((nbins,self._img_dims[-1]),dtype=np.int64)
        manager = mp.Manager()
        field_logs = []
        for fr_chunk in tissue_edge_fr_chunks :
            if len(fr_chunk)<1 :
                continue
            #get the smoothed image layer histograms for this chunk 
            new_img_layer_hists = getImageLayerHistsMT(fr_chunk,
                                                       smooth_sigma=MASKING_CONST.TISSUE_MASK_SMOOTHING_SIGMA,
                                                       med_exposure_times_by_layer=self._med_exp_times_by_layer if et_correction_offsets[0]!=-1. else None,
                                                       et_corr_offsets_by_layer=et_correction_offsets)
            #add the new histograms to the total layer histograms
            for new_img_layer_hist in new_img_layer_hists :
                all_tissue_edge_layer_hists+=new_img_layer_hist
            #find each image's optimal layer thresholds (in parallel)
            return_dict = manager.dict()
            procs=[]
            for ci,fr in enumerate(fr_chunk) :
                self.__writeLogImageInfo(logger,f'determining layer thresholds for file {fr.rawfile_path} {fr.sequence_print}')
                field_logs.append(FieldLog(self._name,fr.rawfile_path,'edge','thresholding'))
                ii=int((fr.sequence_print.split())[0][1:])
                p = mp.Process(target=findLayerThresholds,
                               args=(new_img_layer_hists[ci],
                                     ii,
                                     return_dict
                                     )
                               )
                procs.append(p)
                p.start()
            for proc in procs:
                proc.join()   
            #add all of these images' layer thresholds to the total list
            for ii,layer_thresholds in return_dict.items() :
                for li in range(len(layer_thresholds)) :
                    all_image_thresholds_by_layer[li,ii-1]=layer_thresholds[li]
        #when all the images are done, find the optimal thresholds for each layer
        low_percentile_by_layer=[]; high_percentile_by_layer=[]
        self._background_thresholds_for_masking=[]
        for li in range(self._img_dims[-1]) :
            this_layer_thresholds=all_image_thresholds_by_layer[li,:]
            this_layer_thresholds=this_layer_thresholds[this_layer_thresholds!=0]
            this_layer_thresholds=np.sort(this_layer_thresholds)
            med = int(round(np.median(this_layer_thresholds))) if len(this_layer_thresholds)>0 else 0.
            mean = int(round(np.mean(this_layer_thresholds))) if len(this_layer_thresholds)>0 else 0.
            low_percentile_by_layer.append(this_layer_thresholds[int(0.1*len(this_layer_thresholds))] if len(this_layer_thresholds)>0 else 0.)
            high_percentile_by_layer.append(this_layer_thresholds[int(0.9*len(this_layer_thresholds))] if len(this_layer_thresholds)>0 else 0.)
            self._background_thresholds_for_masking.append(med)
            self.__writeLogInfo(logger,f'threshold for layer {li+1} found at {self._background_thresholds_for_masking[li]}')
            if len(this_layer_thresholds)>0 :
                with cd(plotdir_path) :
                    f,(ax1,ax2,ax3) = plt.subplots(1,3,figsize=(3*6.4,4.6))
                    max_threshold_found = np.max(this_layer_thresholds)
                    ax1.hist(this_layer_thresholds,max_threshold_found+11,(0,max_threshold_found+11))            
                    ax1.plot([mean,mean],[0.8*y for y in ax1.get_ylim()],linewidth=2,color='m',label=f'mean={mean}')
                    ax1.plot([med,med],[0.8*y for y in ax1.get_ylim()],linewidth=2,color='r',label=f'median={med}')
                    ax1.set_title(f'optimal thresholds for images in layer {li+1}')
                    ax1.set_xlabel('pixel flux (counts)')
                    ax1.set_ylabel('n images')
                    ax1.legend(loc='best')
                    ax2.bar(list(range(mean+1)),all_tissue_edge_layer_hists[:mean+1,li],width=1.0,label='background')
                    right_bin = len(all_tissue_edge_layer_hists[:,li])-1
                    while all_tissue_edge_layer_hists[right_bin,li]==0 :
                        right_bin-=1
                    ax2.bar(list(range(mean+1,right_bin+1)),all_tissue_edge_layer_hists[mean+1:right_bin+1,li],width=1.0,label='signal')
                    ax2.set_yscale('log')
                    ax2.set_title('pixel histogram (summed over all images)')
                    ax2.set_xlabel('pixel flux (counts)')
                    ax2.set_ylabel('n image pixels')
                    ax2.legend(loc='best')
                    ax3.bar(list(range(mean+1)),all_tissue_edge_layer_hists[:mean+1,li],width=1.0,label='background')
                    right_plot_limit = min(max_threshold_found,int(1.5*mean))+100
                    ax3.bar(list(range(mean+1,right_plot_limit)),all_tissue_edge_layer_hists[mean+1:right_plot_limit,li],width=1.0,label='signal')            
                    ax3.plot([mean,mean],[0.8*y for y in ax3.get_ylim()],linewidth=2,color='m',label=f'mean={mean}')
                    ax3.plot([med,med],[0.8*y for y in ax3.get_ylim()],linewidth=2,color='r',label=f'median={med}')
                    ax3.set_title('partial pixel histogram')
                    ax3.set_xlabel('pixel flux (counts)')
                    ax3.set_ylabel('n image pixels')
                    ax3.legend(loc='best')
                    fn = f'{self._name}_layer_{li+1}_background_threshold_plots.png'
                    plt.savefig(fn)
                    plt.close()
                    cropAndOverwriteImage(fn)
        #make a little plot of the threshold min/max and final values by layer
        with cd(plotdir_path) :
            xvals=list(range(1,self._img_dims[-1]+1))
            plt.plot(xvals,low_percentile_by_layer,marker='v',color='r',linewidth=2,label='10th %ile thresholds')
            plt.plot(xvals,high_percentile_by_layer,marker='^',color='b',linewidth=2,label='90th %ile thresholds')
            plt.plot(xvals,self._background_thresholds_for_masking,marker='o',color='k',linewidth=2,label='optimal (median) thresholds')
            plt.title('Thresholds chosen from tissue edge HPFs by image layer')
            plt.xlabel('image layer')
            plt.ylabel('pixel flux (counts)')
            plt.legend(loc='best')
            fn = f'{self._name}_background_thresholds_by_layer.png'
            plt.savefig(fn)
            plt.close()
            cropAndOverwriteImage(fn)
        #save the threshold values to a text file
        with cd(top_plotdir_path) :
            with open(f'{self._name}_{UNIV_CONST.BACKGROUND_THRESHOLD_TEXT_FILE_NAME_STEM}','w') as tfp :
                for bgv in self._background_thresholds_for_masking :
                    tfp.write(f'{bgv}\n')
        #return the field logs
        return field_logs

    def findTissueEdgeFilepaths(self,rawfile_paths,plotdir_path=None) :
        """
        Return a list of filepaths corresponding to HPFs that are on the edge of the tissue
        rawfile_paths    = The list of filepaths that will be searched for those on the edge of the tissue
        plotdir_path     = Add a valid directory to this argument to save a plot of where the edge HPFs are next to the reference qptiff
        """
        #make an AlignSample to use in getting the islands
        rawfile_top_dir = ((pathlib.Path(rawfile_paths[0])).parent).parent
        a = AlignSampleFromXML(self._root_dir,rawfile_top_dir,self._name,nclip=UNIV_CONST.N_CLIP,readlayerfile=False,layer=1)
        edge_rect_ns = getAlignSampleTissueEdgeRectNs(a)
        #use this to return the list of tissue edge filepaths
        edge_rect_filenames = [r.file.split('.')[0] for r in a.rectangles if r.n in edge_rect_ns] 
        #use these to make the plot of the rectangle locations
        edge_rect_xs = [r.x for r in a.rectangles if r.n in edge_rect_ns]
        edge_rect_ys = [r.y for r in a.rectangles if r.n in edge_rect_ns] 
        #use this to find the minimum and maximum collection time of the edge rectangle images
        edge_rect_ts = [r.t for r in a.rectangles if r.n in edge_rect_ns] 
        #save the metadata summary file for the thresholding file group
        ms = MetadataSummary(self._name,a.Project,a.Cohort,a.microscopename,str(min(edge_rect_ts)),str(max(edge_rect_ts)))
        if plotdir_path is not None :
            with cd(plotdir_path) :
                writetable(f'{self.TISSUE_EDGE_MDS_STEM}_{self._name}.csv',[ms])
        #make and save the plot of the edge field locations next to the qptiff for reference
        bulk_rect_xs = [r.x for r in a.rectangles if r.file.split('.')[0] not in edge_rect_filenames]
        bulk_rect_ys = [r.y for r in a.rectangles if r.file.split('.')[0] not in edge_rect_filenames]
        if plotdir_path is not None :
            with cd(plotdir_path) :
                has_qptiff = pathlib.Path.is_file(pathlib.Path(f'{self._root_dir}/{self._name}/dbload/{self._name}_qptiff.jpg'))
                if has_qptiff :
                    f,(ax1,ax2) = plt.subplots(1,2,figsize=(25.6,9.6))
                else :
                    f,ax1 = plt.subplots(figsize=(12.8,9.6))
                ax1.scatter(edge_rect_xs,edge_rect_ys,marker='o',color='r',label='edges')
                ax1.scatter(bulk_rect_xs,bulk_rect_ys,marker='o',color='b',label='bulk')
                ax1.invert_yaxis()
                ax1.set_title(f'{self._name} rectangles, ({len(edge_rect_xs)} edge and {len(bulk_rect_xs)} bulk) :',fontsize=18)
                ax1.legend(loc='best',fontsize=18)
                ax1.set_xlabel('x position',fontsize=18)
                ax1.set_ylabel('y position',fontsize=18)
                if has_qptiff :
                    ax2.imshow(mpimg.imread(pathlib.Path(f'{self._root_dir}/{self._name}/dbload/{self._name}_qptiff.jpg')))
                    ax2.set_title('reference qptiff',fontsize=18)
                fn = f'{self._name}_{self.RECTANGLE_LOCATION_PLOT_STEM}.png'
                plt.savefig(fn)
                plt.close()
                cropAndOverwriteImage(fn)
        #return the list of the filepaths whose rectangles are on the edge of the tissue
        return [rfp for rfp in rawfile_paths if ((pathlib.Path(rfp)).name).split('.')[0] in edge_rect_filenames]

    def plotLabelledMaskRegions(self,labelled_mask_regions,rfps_added,masking_plot_dirpath) :
    	"""
		Plot the labelled mask region locations for this slide
		labelled_mask_regions = list of labelled mask regions that were flagged for some reason for this slide
		rfps_added            = list of rawfile paths that were read and stacked from this slide
		masking_plot_dirpath  = path to where the labelled mask region plot should be saved
    	"""
    	#first get the list of all the rawfile paths for this slide
    	with cd(pathlib.Path(f'{self._rawfile_top_dir}/{self._name}')) :
    		all_rfps = [pathlib.Path(f'{self._rawfile_top_dir}/{self._name}/{fn}') for fn in glob.glob(f'*{UNIV_CONST.RAW_EXT}')]
    	#run the plotting function for this slide
    	plotFlaggedHPFLocations(self._name,all_rfps,rfps_added,labelled_mask_regions,masking_plot_dirpath)

    #################### PRIVATE HELPER FUNCTIONS ####################

    #helper function to write an info message to a given log (if None, the default logger is used)
    def __writeLogInfo(self,logger,msg) :
        if logger is not None :
            logger.info(msg,self._name,self._root_dir)
        else :
            flatfield_logger.info(msg)
    #helper function to write an imageinfo message to a given log (if None, the default logger is used)
    def __writeLogImageInfo(self,logger,msg) :
        if logger is not None :
            logger.imageinfo(msg,self._name,self._root_dir)
        else :
            flatfield_logger.info(msg)
