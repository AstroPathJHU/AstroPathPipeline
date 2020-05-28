#imports
from .config import *
from .utilities import chunkListOfFilepaths, readImagesMT
from ..prepdb.overlap import rectangleoverlaplist_fromcsvs
from ..utilities import units
from ..utilities.misc import cd
import numpy as np, matplotlib.pyplot as plt, matplotlib.image as mpimg, multiprocessing as mp
import os, cv2, scipy.stats
import time

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
        #make histograms of all the tissue edge rectangle pixel fluxes per layer
        flatfield_logger.info(f'Getting raw tissue edge images to determine thresholds for sample {self.name}...')
        nbins=np.iinfo(np.uint16).max+1
        all_tissue_edge_image_pixel_hists = np.zeros((nbins,self.dims[-1]),dtype=np.int64)
        for fp_chunk in tissue_edge_fp_chunks :
            if len(fp_chunk)<1 :
                continue
            #read the smoothed raw images from this chunk 
            new_smoothed_img_arrays = readImagesMT(fp_chunk,smoothed=True)
            for smoothed_img_array in new_smoothed_img_arrays :
                #add each image's pixel values to the total layer histogram array
                for li in range(self.dims[-1]) :
                    this_layer_new_hist,_ = np.histogram(smoothed_img_array[:,:,li],nbins,(0,nbins))
                    all_tissue_edge_image_pixel_hists[:,li]+=this_layer_new_hist
        #in parallel, find the thresholds per layer
        manager = mp.Manager()
        return_dict = manager.dict()
        procs = []
        for li in range(self.dims[-1]) :
            flatfield_logger.info(f'  determining threshold for layer {li+1}....')
            p = mp.Process(target=findLayerBackgroundThreshold,
                           args=(all_tissue_edge_image_pixel_hists[:,li],
                                 li,
                                 self.name,
                                 plotdir_path,
                                 return_dict
                                 )
                           )
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
        for li in range(self.dims[-1]) :
            lower_bounds_by_layer.append(return_dict[li]['lower_bound'])
            upper_bounds_by_layer.append(return_dict[li]['upper_bound'])
            self.background_thresholds_for_masking.append(return_dict[li]['final_threshold'])
            msg =f'  threshold for layer {li+1} found at {self.background_thresholds_for_masking[li]} '
            msg+=f'(searched between {lower_bounds_by_layer[li]} and {upper_bounds_by_layer[li]})'
            flatfield_logger.info(msg)
        #make a little plot of the threshold bounds and final values by layer, and save a text file of those values
        with cd(plotdir_path) :
            xvals=list(range(1,self.dims[-1]+1))
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

#helper function to determine the Otsu threshold given a histogram of pixel values 
#algorithm from python code at https://docs.opencv.org/3.4/d7/d4d/tutorial_py_thresholding.html
#reimplemented here for speed and to increase resolution to 16bit
def getOtsuThreshold(pixel_hist) :
    # normalize histogram
    hist_norm = pixel_hist/pixel_hist.sum()
    # get cumulative distribution function
    Q = hist_norm.cumsum()
    # find the upper limit of the histogram
    max_val = len(pixel_hist)
    # set up the loop to determine the threshold
    bins = np.arange(max_val); fn_min = np.inf; thresh = -1
    # loop over all possible values to find where the function is minimized
    for i in range(1,max_val):
        p1,p2 = np.hsplit(hist_norm,[i]) # probabilities
        q1,q2 = Q[i],Q[max_val-1]-Q[i] # cum sum of classes
        if q1 < 1.e-6 or q2 < 1.e-6:
            continue
        b1,b2 = np.hsplit(bins,[i]) # weights
        # finding means and variances
        m1,m2 = np.sum(p1*b1)/q1, np.sum(p2*b2)/q2
        v1,v2 = np.sum(((b1-m1)**2)*p1)/q1,np.sum(((b2-m2)**2)*p2)/q2
        # calculates the minimization function
        fn = v1*q1 + v2*q2
        if fn < fn_min:
            fn_min = fn
            thresh = i
    # return the threshold
    return thresh

#helper function to calculate the nth moment of a histogram
#used in finding the skewness and kurtosis
def moment(hist,n,standardized=False) :
    norm = 1.*hist.sum()
    print(f'norm = {norm}')
    mean = 0.
    for k,p in enumerate(hist) :
        mean+=p*k
    mean/=norm
    var  = 0.
    moment = 0.
    for k,p in enumerate(hist) :
        var+=p*((k-mean)**2)
        moment+=p*((k-mean)**n)
    var/=norm
    moment/=norm
    if standardized :
        retnorm = (var**(n/2.))
        print(f'retnorm = {retnorm}')
        retval = moment/retnorm
    else :
        retval = moment
    return retval

#helper function to determine a background threshold flux given the histogram of a single layer's pixel fluxes
#designed to be run in parallel
def findLayerBackgroundThreshold(layerpix,layer_i,sample_name,plotdir_path,return_dict) :
    #iterate calculating and applying the Otsu threshold values, keeping track of the lowest threshold
    #that results in a sufficiently large background pixel distribution kurtosis, and the threshold at which
    #the background pixel distribution skew goes negative
    next_it_pixels = layerpix; lowest_threshold=100000; last_large_kurtosis_threshold=100000; iterations=0; skew = 1000.
    while skew>0.0 :
        #get the threshold from OpenCV's Otsu thresholding procedure
        test_threshold = getOtsuThreshold(next_it_pixels)
        #calculate the skew and kurtosis of the pixels that would be background at this threshold
        bg_pixels = layerpix[:test_threshold]
        skew = moment(bg_pixels,3,True)
        kurtosis = moment(bg_pixels,4,True)-3
        #record this iteration if the skew is positive and the kurotsis is large enough
        if skew>0.0 and kurtosis>UPPER_THRESHOLD_KURTOSIS_CUT :
            last_large_kurtosis_threshold = test_threshold
        #set the new threshold and the next iteration's pixels
        lowest_threshold = test_threshold
        next_it_pixels = bg_pixels
        iterations+=1
        ##print some info
        #msg = f'  layer {layer_i+1} it. {iterations+1} '
        #msg+=f'skewness = {skew:.3f}, '
        #msg+=f'kurtosis = {kurtosis:.3f}, '
        #msg+=f'test thresh.={test_threshold:.1f}:'
        #flatfield_logger.info(msg)
    #within the two threshold limits given by the lowest threshold with large kurtosis and the threshold where the skew flips sign, 
    #exhaustively find the values between which the kurtosis of the background pixels changed the most
    upper_bound = max(last_large_kurtosis_threshold+1,lowest_threshold+MIN_POINTS_TO_SEARCH)
    test_thresholds = list(range(lowest_threshold,upper_bound))
    skews = []; kurtoses = []
    for tt in test_thresholds :
        test_hist = layerpix[:tt]
        skews.append(moment(test_hist,3,True))
        kurtoses.append(moment(test_hist,4,True)-3)
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
        ax3.bar(list(range(final_threshold)),layerpix[:final_threshold],width=1.0)
        ax3.set_title('histogram of final background pixels')
        plt.savefig(figname)
        plt.close()
    #set the values in the return dict
    return_dict[layer_i] = {'lower_bound':lowest_threshold,
                            'upper_bound':upper_bound,
                            'final_threshold':final_threshold
                            }
