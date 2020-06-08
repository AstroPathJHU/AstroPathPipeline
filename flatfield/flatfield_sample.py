#imports
from .config import *
from .utilities import chunkListOfFilepaths, getImageLayerHistsMT
from ..prepdb.overlap import rectangleoverlaplist_fromcsvs
from ..utilities import units
import matplotlib.pyplot as plt, matplotlib.image as mpimg, multiprocessing as mp
import math, scipy.stats

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
        all_image_thresholds_by_layer = np.empty((self.dims[-1],len(tissue_edge_filepaths)),dtype=np.uint16)
        all_tissue_edge_layer_hists = np.zeros((nbins,self.dims[-1]),dtype=np.int64)
        manager = mp.Manager()
        for fp_chunk in tissue_edge_fp_chunks :
            if len(fp_chunk)<1 :
                continue
            #get the smoothed image layer histograms for this chunk 
            new_smoothed_img_layer_hists = getImageLayerHistsMT(fp_chunk,smoothed=True)
            #add the new histograms to the total layer histograms
            for new_smoothed_img_layer_hist in new_smoothed_img_layer_hists :
                all_tissue_edge_layer_hists+=new_smoothed_img_layer_hist
            #find each image's optimal layer thresholds (in parallel)
            return_dict = manager.dict()
            procs=[]
            for ci,fpt in enumerate(fp_chunk) :
                flatfield_logger.info(f'  determining layer thresholds for file {fpt[0]} {fpt[1]}')
                ii=int((fpt[1].split())[0][1:])
                p = mp.Process(target=findLayerThresholds,
                               args=(new_smoothed_img_layer_hists[ci],
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
        self.background_thresholds_for_masking=[]
        for li in range(self.dims[-1]) :
            this_layer_thresholds=all_image_thresholds_by_layer[li,:]
            this_layer_thresholds=this_layer_thresholds[this_layer_thresholds!=0]
            this_layer_thresholds=np.sort(this_layer_thresholds)
            med = int(round(np.median(this_layer_thresholds)))
            mean = int(round(np.mean(this_layer_thresholds)))
            mode,_ = scipy.stats.mode(this_layer_thresholds)
            mode=int(round(mode[0]))
            low_percentile_by_layer.append(this_layer_thresholds[int(round(0.1*len(this_layer_thresholds)))])
            high_percentile_by_layer.append(this_layer_thresholds[int(round(0.9*len(this_layer_thresholds)))])
            self.background_thresholds_for_masking.append(med)
            flatfield_logger.info(f'  threshold for layer {li+1} found at {self.background_thresholds_for_masking[li]}')
            with cd(plotdir_path) :
                f,(ax1,ax2) = plt.subplots(1,2,figsize=(12.8,4.6))
                ax1.hist(this_layer_thresholds,np.max(this_layer_thresholds)+1,(0,np.max(this_layer_thresholds)+1))
                ax1.plot([mode,mode],[0.8*y for y in ax1.get_ylim()],linewidth=2,color='c',label=f'mode={mode}')
                ax1.plot([mean,mean],[0.8*y for y in ax1.get_ylim()],linewidth=2,color='m',label=f'mean={mean}')
                ax1.plot([med,med],[0.8*y for y in ax1.get_ylim()],linewidth=2,color='r',label=f'median={med}')
                ax1.set_title(f'optimal thresholds for images in layer {li+1}')
                ax1.set_xlabel('pixel flux')
                ax1.set_ylabel('n images')
                ax1.legend(loc='best')
                ax2.bar(list(range(med+1)),all_tissue_edge_layer_hists[:med+1,li],width=1.0)
                ax2.set_title('histogram of background pixels (summed over all images)')
                ax2.set_xlabel('pixel flux')
                ax2.set_ylabel('n image pixels')
                plt.savefig(f'{self.name}_layer_{li+1}_background_threshold_plots.png')
                plt.close()
        #make a little plot of the threshold min/max and final values by layer, and save a text file of those values
        with cd(plotdir_path) :
            xvals=list(range(1,self.dims[-1]+1))
            plt.plot(xvals,low_percentile_by_layer,marker='v',color='r',linewidth=2,label='10th %ile thresholds')
            plt.plot(xvals,high_percentile_by_layer,marker='^',color='b',linewidth=2,label='90th %ile thresholds')
            plt.plot(xvals,self.background_thresholds_for_masking,marker='o',color='k',linewidth=2,label='optimal thresholds')
            plt.title('Thresholds chosen from tissue edge HPFs by image layer')
            plt.xlabel('image layer')
            plt.ylabel('pixel flux')
            plt.legend(loc='best')
            plt.savefig(f'{self.name}_background_thresholds_by_layer.png')
            plt.close()
            #save the threshold values to a text file
            with open(f'{self.name}_{THRESHOLD_TEXT_FILE_NAME_STEM}','w') as tfp :
                for bgv in self.background_thresholds_for_masking :
                    tfp.write(f'{bgv}\n')

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
            plt.savefig(f'{self.name}_{RECTANGLE_LOCATION_PLOT_STEM}.png')
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
def moment(hist,n,standardized=True) :
    norm = 1.*hist.sum()
    #if there are no entries the moments are undefined
    if norm==0. :
        return float('NaN')
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
    #if the moment is zero, then the histogram was just one bin and the moment is undefined
    if moment==0 :
        return float('NaN')
    if standardized :
        return moment/(var**(n/2.))
    else :
        return moment

# a helper function to take a list of layer histograms and return the list of optimal thresholds
#designed to be run in parallel
def findLayerThresholds(layer_hists,i,rdict) :
    best_thresholds = []
    #for each layer
    for li in range(layer_hists.shape[-1]) :
        hist=layer_hists[:,li]
        #iterate calculating and applying the Otsu threshold values
        next_it_pixels = hist; skew = 1000.
        test_thresholds=[]; test_weighted_skew_slopes=[]
        while not math.isnan(skew) :
            #get the threshold from OpenCV's Otsu thresholding procedure
            test_threshold = getOtsuThreshold(next_it_pixels)
            #calculate the skew and kurtosis of the pixels that would be background at this threshold
            bg_pixels = hist[:test_threshold+1]
            skew = moment(bg_pixels,3)
            if not math.isnan(skew) :
                test_thresholds.append(test_threshold)
                skewslope=(moment(hist[:test_threshold+2],3) - moment(hist[:test_threshold],3))/2.
                test_weighted_skew_slopes.append(skewslope/skew if not math.isnan(skewslope) else 0)
            #set the next iteration's pixels
            next_it_pixels = bg_pixels
        #add the best threshold to the list
        if len(test_thresholds)<1 :
            best_thresholds.append(0)
        else :
            best_thresholds.append(test_thresholds[test_weighted_skew_slopes.index(max(test_weighted_skew_slopes))])
    #put the list of thresholds in the return dict
    rdict[i]=best_thresholds
