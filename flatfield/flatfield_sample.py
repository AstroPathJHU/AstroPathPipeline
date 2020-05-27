#imports
from .config import *
from .utilities import chunkListOfFilepaths, readImagesMT
from ..prepdb.overlap import rectangleoverlaplist_fromcsvs
from ..utilities import units
from ..utilities.misc import cd
import numpy as np, matplotlib.pyplot as plt, matplotlib.image as mpimg, multiprocessing as mp
import os, cv2, scipy.stats

units.setup('fast')

#main class definition
class FlatfieldSample() :

    def __init__(self,name,img_dims) :
        """
        name          = name of the sample
        img_dims      = dimensions of images in files in order as (height, width, # of layers) 
        """
        self.name = name
        self.dims = img_dims
        self.background_thresholds_for_masking = None

    def readInBackgroundThresholds(self,threshold_file_path) :
        """
        Function to read in background threshold values from the output file of a previous run with this sample
        threshold_file_path = path to threshold value file
        """
        if self.background_thresholds_for_masking is not None :
            raise FlatFieldError('ERROR: calling readInBackgroundThresholds with non-empty thresholds list')
        self.background_thresholds_for_masking=[]
        with open(threshold_file_path,'r') as tfp :
            all_lines = [l.rstrip() for l in tfp.readlines()]
            for line in all_lines :
                try :
                    self.background_thresholds_for_masking.append(int(line))
                except ValueError :
                    pass
        if not len(self.background_thresholds_for_masking)==self.dims[-1] :
            raise FlatFieldError(f'ERROR: number of background thresholds read from {threshold_file_path} is not equal to the number of image layers!')

    def findBackgroundThresholds(self,rawfile_paths,dbload_dir,n_threads,plotdir_path,threshold_file_name) :
        """
        Function to determine this sample's background pixel flux thresholds per layer
        rawfile_paths       = a list of the rawfile paths to consider for this sample's background threshold calculations
        dbload_dir          = this sample's dbload directory 
        n_threads           = max number of threads/processes to open at once
        plotdir_path        = path to the directory in which to save plots from the thresholding process
        threshold_file_name = name of file to save background thresholds in, one line per layer
        """
        #make sure the plot directory exists
        if not os.path.isdir(plotdir_path) :
            with cd(os.path.join(*[pp for pp in plotdir_path.split(os.sep)[:-1]])) :
                os.mkdir(plotdir_path.split(os.sep)[-1])
        #first find the filepaths corresponding to the edges of the tissue in the samples
        flatfield_logger.info(f'Finding tissue edge HPFs for sample {self.name}...')
        tissue_edge_filepaths = self.__findTissueEdgeFilepaths(rawfile_paths,dbload_dir,plotdir_path)
        #chunk them together to be read in parallel
        tissue_edge_fp_chunks = chunkListOfFilepaths(tissue_edge_filepaths,self.dims,n_threads)
        #make an array of all the tissue edge rectangle pixel fluxes per layer
        flatfield_logger.info(f'Getting raw tissue edge images to determine thresholds for sample {self.name}...')
        all_tissue_edge_image_arrays = np.ndarray((self.dims[0],self.dims[1],self.dims[2],len(tissue_edge_filepaths)),dtype=np.uint16)
        starting_image_i=0
        for fp_chunk in tissue_edge_fp_chunks :
            if len(fp_chunk)<1 :
                continue
            #read the raw images from this chunk
            new_img_arrays = readImagesMT(fp_chunk)
            for chunk_image_i,img_array in enumerate(new_img_arrays) :
                #copy each image to the total array
                np.copyto(all_tissue_edge_image_arrays[:,:,:,starting_image_i+chunk_image_i],img_array)
            starting_image_i+=len(new_img_arrays)
        #transpose the array to put the layers at the end of all of the images to just be a list of pixel values per layer
        all_tissue_edge_images_per_layer = np.transpose(all_tissue_edge_image_arrays,(0,1,3,2))
        #in parallel, find the thresholds per layer
        manager = mp.Manager()
        return_dict = manager.dict()
        procs = []
        for li in range(self.dims[2]) :
            flatfield_logger.info(f'  determining threshold for layer {li+1}....')
            p = mp.Process(target=findLayerBackgroundThreshold, args=(all_tissue_edge_images_per_layer[:,:,:,li],li,self.name,plotdir_path,return_dict))
            procs.append(p)
            p.start()
            if len(procs)>=n_threads :
                for proc in procs :
                    proc.join()
                procs=[]
        for proc in procs:
            proc.join()
        #when all the layers are done, assign them to this sample's list
        lower_bounds_by_layer = []; upper_bounds_by_layer = []
        self.background_thresholds_for_masking=[]
        for li in range(self.dims[2]) :
            lower_bounds_by_layer.append(return_dict[li]['lower_bound'])
            upper_bounds_by_layer.append(return_dict[li]['upper_bound'])
            self.background_thresholds_for_masking.append(return_dict[li]['final_threshold'])
            msg =f'  threshold for layer {li+1} found at {self.background_thresholds_for_masking[li]} '
            msg+=f'(searched between {lower_bounds_by_layer[li]} and {upper_bounds_by_layer[li]})'
            flatfield_logger.info(msg)
        #make a little plot of the threshold bounds and final values by layer, and save a text file of those values
        with cd(plotdir_path) :
            xvals=list(range(1,self.dims[2]+1))
            plt.plot(xvals,lower_bounds_by_layer,marker='v',color='r',linewidth=2,label='lower bounds')
            plt.plot(xvals,upper_bounds_by_layer,marker='^',color='b',linewidth=2,label='upper bounds')
            plt.plot(xvals,self.background_thresholds_for_masking,marker='o',color='k',linewidth=2,label='optimal thresholds')
            plt.title('Thresholds chosen from tissue edge HPFs by image layer')
            plt.xlabel('image layer')
            plt.ylabel('pixel flux')
            plt.legend(loc='best')
            plt.savefig(f'{self.name}_background_thresholds_by_layer.png')
            plt.close()
            #save the threshold values to a text file
            with open(THRESHOLD_TEXT_FILE_NAME_STEM,'w') as tfp :
                for bgv in self.background_thresholds_for_masking :
                    tfp.write(f'{bgv}{os.linesep}')

    #################### PRIVATE HELPER FUNCTIONS ####################

    #helper function to return the subset of the filepath list corresponding to HPFs on the edge of tissue
    def __findTissueEdgeFilepaths(self,rawfile_paths,dbload_dir,plotdir_path) :
        #make the RectangleOverlapList 
        rol = rectangleoverlaplist_fromcsvs(dbload_dir)
        #get the list of sets of rectangle IDs by island
        samp_islands = rol.islands()
        edge_rect_filenames = [] #use this to return the list of tissue edge filepaths
        edge_rect_xs = []; edge_rect_ys = [] #use these to make the plot of the rectangle locations
        #for each island
        for ii,island in enumerate(samp_islands,start=1) :
            island_rects = [r for r in rol.rectangles if r.n in island]
            #get the width and height of the rectangles
            rw, rh = island_rects[0].w, island_rects[0].h
            #get the x/y positions of the rectangles in the island
            x_poss = sorted(list(set([r.x for r in island_rects])))
            y_poss = sorted(list(set([r.y for r in island_rects])))
            #make a list of the ns of the rectangles on the edge of this island
            island_edge_rect_ns = []
            #iterate over them first from top to bottom to add the vertical edges
            for row_y in y_poss :
                row_rects = [r for r in island_rects if r.y==row_y]
                row_x_poss = sorted([r.x for r in row_rects])
                #add the rectangles of the ends
                island_edge_rect_ns+=[r.n for r in row_rects if r.x in (row_x_poss[0],row_x_poss[-1])]
                #add any rectangles that have a gaps between them and the previous
                for irxp in range(1,len(row_x_poss)) :
                    if abs(row_x_poss[irxp]-row_x_poss[irxp-1])>rw :
                        island_edge_rect_ns+=[r.n for r in row_rects if r.x in (row_x_poss[irxp-1],row_x_poss[irxp])]
            #iterate over them again from left to right to add the horizontal edges
            for col_x in x_poss :
                col_rects = [r for r in island_rects if r.x==col_x]
                col_y_poss = sorted([r.y for r in col_rects])
                #add the rectangles of the ends
                island_edge_rect_ns+=[r.n for r in col_rects if r.y in (col_y_poss[0],col_y_poss[-1])]
                #add any rectangles that have a gaps between them and the previous
                for icyp in range(1,len(col_y_poss)) :
                    if abs(col_y_poss[icyp]-col_y_poss[icyp-1])>rh :
                        island_edge_rect_ns+=[r.n for r in col_rects if r.y in (col_y_poss[icyp-1],col_y_poss[icyp])]
            #add this island's edge rectangles' filenames and x/y positions to the total lists
            add_rects = [r for r in island_rects if r.n in list(set(island_edge_rect_ns))]
            edge_rect_filenames+=[r.file.split('.')[0] for r in add_rects]
            edge_rect_xs+=[r.x for r in add_rects]
            edge_rect_ys+=[r.y for r in add_rects]
        #make and save the plot of the edge field locations next to the qptiff for reference
        bulk_rect_xs = [r.x for r in rol.rectangles if r.file.split('.')[0] not in edge_rect_filenames]
        bulk_rect_ys = [r.y for r in rol.rectangles if r.file.split('.')[0] not in edge_rect_filenames]
        with cd(plotdir_path) :
            f,(ax1,ax2) = plt.subplots(1,2,figsize=(25.6,9.6))
            ax1.scatter(edge_rect_xs,edge_rect_ys,marker='o',color='r',label='edges')
            ax1.scatter(bulk_rect_xs,bulk_rect_ys,marker='o',color='b',label='bulk')
            ax1.invert_yaxis()
            ax1.set_title(f'{self.name} rectangles, ({len(edge_rect_xs)} edge and {len(bulk_rect_xs)} bulk) :',fontsize=18)
            ax1.legend(loc='best',fontsize=18)
            ax1.set_xlabel('x position',fontsize=18)
            ax1.set_ylabel('y position',fontsize=18)
            ax2.imshow(mpimg.imread(os.path.join(dbload_dir,f'{self.name}_qptiff.jpg')))
            ax2.set_title('reference qptiff',fontsize=18)
            plt.savefig(f'{RECTANGLE_LOCATION_PLOT_STEM}_{self.name}.png')
            plt.close()
        #return the list of the filepaths whose rectangles are on the edge of the tissue
        return [rfp for rfp in rawfile_paths if rfp.split(os.sep)[-1].split('.')[0] in edge_rect_filenames]

#################### FILE-SCOPE HELPER FUNCTIONS ####################

#helper function to determine a background threshold flux given a single layer's total pixel array
#designed to be run in parallel
def findLayerBackgroundThreshold(images_array,layer_i,sample_name,plotdir_path,return_dict) :
    #first smooth all the images in the given list
    images_array = cv2.GaussianBlur(images_array,(0,0),GENTLE_GAUSSIAN_SMOOTHING_SIGMA,borderType=cv2.BORDER_REPLICATE)
    #sort this layer's list of pixel fluxes
    layerpix = np.sort(images_array.flatten())
    #iterate calculating and applying the Otsu threshold values, keeping track of the lowest threshold
    #that results in a sufficiently large background pixel distribution kurtosis, and the threshold at which
    #the background pixel distribution skew goes negative
    next_it_pixels = layerpix
    threshold=100000; last_large_kurtosis_threshold=100000; iterations=0; skew = 1000.; grayscale_max_value=np.iinfo(np.uint8).max
    while skew>0.0 :
        #cut the smoothed images at the value for the current iteration
        this_it_pixels = next_it_pixels
        #convert the 16bit integer pixels to 8bit integer grayscale values, rescaling the maximum if necessary
        this_it_rs = (1.*grayscale_max_value/max(np.max(this_it_pixels),grayscale_max_value))
        for_threshold = (this_it_rs*this_it_pixels).astype('uint8')
        #find the test threshold from Otsu's algorithm for this iteration and rescale it back to the 16bit range
        test_threshold_rs,_ = cv2.threshold(for_threshold,0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        test_threshold = 1.*test_threshold_rs/this_it_rs
        #calculate the skew and kurtosis of the pixels that would be background at this threshold
        bg_pixels = layerpix[:np.where(layerpix<=test_threshold)[0][-1]+1]
        skew = scipy.stats.skew(bg_pixels)
        kurtosis = scipy.stats.kurtosis(bg_pixels)
        #record this iteration if the skew is positive and the kurotsis is large enough
        if skew>0.0 and kurtosis>UPPER_THRESHOLD_KURTOSIS_CUT :
            last_large_kurtosis_threshold = test_threshold
        #set the new threshold and the next iteration's pixels
        threshold = test_threshold
        next_it_pixels = bg_pixels
        iterations+=1
        ##print some info
        #msg = f'  layer {layer_i+1} it. {iterations+1} '
        #msg+=f'skewness = {skew:.3f}, '
        #msg+=f'kurtosis = {kurtosis:.3f}, '
        #msg+=f'abs(skew)*abs(kurtosis) = {abs(skew)*abs(kurtosis):.3f}, '
        #msg+=f'test thresh.={test_threshold:.1f}:'
        #flatfield_logger.info(msg)
    #within the two threshold limits given by the lowest threshold with large kurtosis and the threshold where the skew flips sign, 
    #exhaustively find the values between which the product of the absolute values of the skew and kurtosis of the background pixels changed the most
    test_thresholds = list(range(int(threshold),max(int(last_large_kurtosis_threshold)+1,int(threshold)+MIN_POINTS_TO_SEARCH)))
    skews = []; kurtoses = []
    test_thresh_indices = [(np.where(layerpix<=t))[0][-1]+1 for t in test_thresholds]
    for ti in test_thresh_indices :
        skews.append(scipy.stats.skew(layerpix[:ti]))
        kurtoses.append(scipy.stats.kurtosis(layerpix[:ti]))
    kurtosis_diffs = [kurtoses[i+1]-kurtoses[i] for i in range(len(kurtoses)-1)]
    kurtosis_diffs_no_negative_skew = [] 
    for i in range(len(kurtosis_diffs)) :
        if skews[i]>0 :
            kurtosis_diffs_no_negative_skew.append(kurtosis_diffs[i])
        else :
            kurtosis_diffs_no_negative_skew.append(-10.)
    final_threshold = test_thresholds[kurtosis_diffs_no_negative_skew.index(max(kurtosis_diffs_no_negative_skew))+1]
    #make and save plots
    figname=f'{sample_name}_layer_{layer_i+1}_background_threshold_plots.png'
    with cd(plotdir_path) :
        f,(ax1,ax2,ax3)=plt.subplots(1,3,figsize=(19.2,4.6))
        ax1.plot(test_thresholds,skews,marker='v',color='r',label='bg pixel skew')
        ax1.plot(test_thresholds,kurtoses,marker='^',color='b',label='bg pixel kurtoses')
        ax1.plot([final_threshold,final_threshold],list(ax1.get_ylim()),color='k',linewidth=2,label='chosen threshold')
        ax1.set_title('background pixel skew and kurtosis at candidate thresholds')
        ax1.set_xlabel('candidate threshold')
        ax1.set_ylabel('background pixel skew and kurtosis')
        ax1.legend(loc='best')
        ax2.plot(test_thresholds[1:],kurtosis_diffs,marker='o',linewidth=2)
        ax2.plot([final_threshold,final_threshold],list(ax2.get_ylim()),color='k',linewidth=2)
        ax2.set_title('slope of kurtoses of background pixels')
        ax2.set_xlabel('candidate threshold')
        ax2.set_ylabel('slope of kurtosis curve')
        ax3.hist(layerpix[:np.where(layerpix<=final_threshold)[0][-1]+1])
        ax3.set_title('histogram of final background pixels')
        plt.savefig(figname)
        plt.close()
    #set the values in the return dict
    return_dict[layer_i] = {'lower_bound':threshold,
                            'upper_bound':max(last_large_kurtosis_threshold,int(threshold)+MIN_POINTS_TO_SEARCH),
                            'final_threshold':final_threshold
                            }
