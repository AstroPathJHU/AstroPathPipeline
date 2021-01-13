#imports
from astropath_calibration.flatfield.utilities import chunkListOfFilepaths, readImagesMT
from astropath_calibration.flatfield.config import CONST
from astropath_calibration.utilities.img_file_io import getImageHWLFromXMLFile, getSlideMedianExposureTimesByLayer, LayerOffset, smoothImageWorker
#from astropath_calibration.utilities.img_file_io import writeImageToFile
from astropath_calibration.utilities.tableio import readtable, writetable
from astropath_calibration.utilities.misc import cd, addCommonArgumentsToParser, cropAndOverwriteImage
from astropath_calibration.utilities import units
from astropath_calibration.baseclasses.csvclasses import constantsdict
from argparse import ArgumentParser
import numpy as np, matplotlib.pyplot as plt, multiprocessing as mp
import logging, os, glob, cv2, dataclasses, scipy.stats

#constants
RAWFILE_EXT                = '.Data.dat'
LOCAL_MEAN_KERNEL          = np.array([[0.0,0.2,0.0],
                                       [0.2,0.2,0.2],
                                       [0.0,0.2,0.0]])
#WINDOW_EL                  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(37,37))
WINDOW_EL                  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(41,41))
MASK_LAYER_GROUPS          = [(1,9),(10,18),(19,25),(26,32),(33,35)]
BRIGHTEST_LAYERS           = [5,11,21,29,34]
DAPI_LAYER_GROUP_INDEX     = 0
TISSUE_MIN_SIZE            = 2500
BLUR_NLV_CUT               = 0.5
BLUR_MIN_SIZE              = 15000
BLUR_MASK_FLAG_CUTS        = [7,7,5,5,1]
BLUR_MIN_LAYER_GROUPS      = 3
BLUR_FLAG_STRING           = 'blur'

#logger
logger = logging.getLogger("flagging_hpf_regions")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s  [%(funcName)s]","%Y-%m-%d %H:%M:%S"))
logger.addHandler(handler)

#mask region information helper class
@dataclasses.dataclass(eq=False)
class LabelledMaskRegion :
    image_key          : str
    cellview_x         : float
    cellview_y         : float
    region_index       : int
    layers             : str
    n_pixels           : int
    reason_flagged     : str

#################### IMAGE HANDLING UTILITIY FUNCTIONS ####################

#return a binary mask with all of the areas smaller than min_size removed
def getSizeFilteredMask(mask,min_size,both=True,invert=False) :
    if invert :
        mask = (np.where(mask==1,0,1)).astype(mask.dtype)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    sizes = stats[1:, -1]; nb_components = nb_components - 1
    new_mask = np.zeros_like(mask)
    for i in range(0, nb_components):
        if sizes[i] >= min_size :
            new_mask[output == i + 1] = 1
    if invert :
        new_mask = (np.where(new_mask==1,0,1)).astype(mask.dtype)
    if both :
        return getSizeFilteredMask(new_mask,min_size,both=False,invert=(not invert))
    return new_mask

#return a binary mask with all of the areas whose distributions of reference values have skew less than min_skew removed
def getSkewFilteredMask(mask,ref,min_skew,invert=True) :
    if invert :
        mask = (np.where(mask==1,0,1)).astype(mask.dtype)
    n_regions, regions_im = cv2.connectedComponents(mask)
    new_mask = np.zeros_like(mask)
    for region_i in range(1,n_regions) :
        if scipy.stats.skew(ref[regions_im==region_i])<min_skew :
            continue
        new_mask[regions_im==region_i] = mask[regions_im==region_i]
    if invert :
        new_mask = (np.where(new_mask==1,0,1)).astype(mask.dtype)
    return new_mask

#return a binary mask with all of the areas whose distributions of reference values have mean greater than max_mean removed
def getMeanFilteredMask(mask,ref,max_mean,invert=True) :
    if invert :
        mask = (np.where(mask==1,0,1)).astype(mask.dtype)
    n_regions, regions_im = cv2.connectedComponents(mask)
    new_mask = np.zeros_like(mask)
    for region_i in range(1,n_regions) :
        if np.mean(ref[regions_im==region_i])>max_mean :
            continue
        new_mask[regions_im==region_i] = mask[regions_im==region_i]
    if invert :
        new_mask = (np.where(new_mask==1,0,1)).astype(mask.dtype)
    return new_mask

#return the minimally-transformed tissue mask for a single image layer
def getImageLayerTissueMask(img_layer,bkg_threshold) :
    sm_layer = smoothImageWorker(img_layer,CONST.GENTLE_GAUSSIAN_SMOOTHING_SIGMA)
    img_mask = (np.where(sm_layer>bkg_threshold,1,0)).astype(np.uint8)
    img_mask = cv2.morphologyEx(img_mask,cv2.MORPH_CLOSE,CONST.CO1_EL,borderType=cv2.BORDER_REPLICATE)
    img_mask = cv2.morphologyEx(img_mask,cv2.MORPH_OPEN,CONST.CO1_EL,borderType=cv2.BORDER_REPLICATE)
    return img_mask

#return the fully-determined single tissue mask for a multilayer image
def getImageTissueMask(image_arr,bkg_thresholds) :
    #mask each layer individually first
    layer_masks = []
    for li in range(image_arr.shape[-1]) :
        layer_masks.append(getImageLayerTissueMask(image_arr[:,:,li],bkg_thresholds[li]))
    #find the well-defined tissue and background in each layer group
    overall_tissue_mask = np.zeros_like(layer_masks[0])
    overall_background_mask = np.zeros_like(layer_masks[0])
    total_stacked_masks = np.zeros_like(layer_masks[0])
    #for each layer group
    for lgi,lgb in enumerate(MASK_LAYER_GROUPS) :
        stacked_masks = np.zeros_like(layer_masks[0])
        for ln in range(lgb[0],lgb[1]+1) :
            stacked_masks+=layer_masks[ln-1]
        total_stacked_masks+=stacked_masks
        #well-defined tissue is anything called tissue in at least all but two layers
        overall_tissue_mask[stacked_masks>(lgb[1]-lgb[0]-1)]+= 10 if lgi==DAPI_LAYER_GROUP_INDEX else 1
        #well-defined background is anything called background in at least half the layers
        overall_background_mask[stacked_masks<(lgb[1]-lgb[0]+1)/2.]+= 10 if lgi==DAPI_LAYER_GROUP_INDEX else 1
    #threshold tissue/background masks to include only those from the DAPI and at least one other layer group
    overall_tissue_mask = (np.where(overall_tissue_mask>10,1,0)).astype(np.uint8)
    overall_background_mask = (np.where(overall_background_mask>10,1,0)).astype(np.uint8)
    #final mask has tissue=1, background=0
    final_mask = np.zeros_like(layer_masks[0])+2
    final_mask[overall_tissue_mask==1] = 1
    final_mask[overall_background_mask==1] = 0
    #anything left over is signal if it's stacked in at least half the total number of layers
    thresholded_stacked_masks = np.where(total_stacked_masks>(image_arr.shape[-1]/2.),1,0)
    final_mask[final_mask==2] = thresholded_stacked_masks[final_mask==2]
    #filter the tissue and background portions to get rid of the small islands
    final_mask = getSizeFilteredMask(final_mask,min_size=TISSUE_MIN_SIZE)
    return final_mask

#function to compute and return the variance of the normalized laplacian for a given image layer
def getImageLayerLocalVarianceOfNormalizedLaplacian(img_layer,tissue_mask=None) :
    #build the laplacian image and normalize it to get the curvature
    img_laplacian = cv2.Laplacian(img_layer,cv2.CV_32F,borderType=cv2.BORDER_REFLECT)
    img_lap_norm = cv2.filter2D(img_layer,cv2.CV_32F,LOCAL_MEAN_KERNEL,borderType=cv2.BORDER_REFLECT)
    img_norm_lap = img_laplacian
    img_norm_lap[img_lap_norm!=0] /= img_lap_norm[img_lap_norm!=0]
    img_norm_lap[img_lap_norm==0] = 0
    #find the variance of the normalized laplacian in the neighborhood window, disregarding the background if a tissue mask is given
    if tissue_mask is not None :
        norm_lap_loc_mean = tissue_mask*cv2.filter2D(tissue_mask*img_norm_lap,cv2.CV_32F,WINDOW_EL,borderType=cv2.BORDER_REFLECT)
        norm_lap_2_loc_mean = tissue_mask*cv2.filter2D(tissue_mask*np.power(img_norm_lap,2),cv2.CV_32F,WINDOW_EL,borderType=cv2.BORDER_REFLECT)
        local_mask_norm = tissue_mask*cv2.filter2D(tissue_mask,cv2.CV_8U,WINDOW_EL,borderType=cv2.BORDER_REFLECT)
    else :
        norm_lap_loc_mean = cv2.filter2D(img_norm_lap,cv2.CV_32F,WINDOW_EL,borderType=cv2.BORDER_REFLECT)
        norm_lap_2_loc_mean = cv2.filter2D(np.power(img_norm_lap,2),cv2.CV_32F,WINDOW_EL,borderType=cv2.BORDER_REFLECT)
        local_mask_norm = cv2.filter2D(np.ones(img_layer.shape,dtype=np.float32),cv2.CV_8U,WINDOW_EL,borderType=cv2.BORDER_REFLECT)
    norm_lap_loc_mean[local_mask_norm!=0] /= local_mask_norm[local_mask_norm!=0]
    norm_lap_loc_mean[local_mask_norm==0] = 0
    norm_lap_2_loc_mean[local_mask_norm!=0] /= local_mask_norm[local_mask_norm!=0]
    norm_lap_2_loc_mean[local_mask_norm==0] = 0
    local_norm_lap_var = np.abs(norm_lap_2_loc_mean-np.power(norm_lap_loc_mean,2))
    if tissue_mask is not None :
        local_norm_lap_var[tissue_mask==0] = np.max(local_norm_lap_var)
    #return the local variance of the normalized laplacian
    return local_norm_lap_var

#function to return a blur mask for a given image layer, along with a dictionary of to plots to add to the group for this image
def getImageLayerGroupBlurMaskAndPlots(img_array,layer_group_bounds,brightest_layer_n,flag_cut) :
    #start by making a mask for every layer in the group
    stacked_masks = np.zeros(img_array.shape[:-1],dtype=np.uint8)
    brightest_layer_nlv = None
    for ln in range(layer_group_bounds[0],layer_group_bounds[1]+1) :
        #get the local variance of the normalized laplacian image
        img_nlv = getImageLayerLocalVarianceOfNormalizedLaplacian(img_array[:,:,ln-1])
        if ln==brightest_layer_n :
            brightest_layer_nlv = img_nlv
        #threshold it to make a binary mask
        layer_mask = (np.where(img_nlv>BLUR_NLV_CUT,1,0)).astype(np.uint8)
        ##filter out the small areas
        #layer_mask = getSizeFilteredMask(layer_mask,min_size=BLUR_MIN_SIZE)
        ##filter out anything with skew of nlv values < MIN_SKEW or mean of nlv values > MAX_MEAN (false positives)
        #layer_mask = getSkewFilteredMask(layer_mask,img_nlv,BLUR_MIN_SKEW)
        #layer_mask = getMeanFilteredMask(layer_mask,img_nlv,BLUR_MAX_MEAN)
        stacked_masks+=layer_mask
    #determine the final mask for this group by thresholding on how many individual layers contribute
    group_blur_mask = (np.where(stacked_masks>flag_cut,1,0)).astype(np.uint8)    
    ##small open/close to refine it
    #group_blur_mask = (cv2.morphologyEx(group_blur_mask,cv2.MORPH_OPEN,CONST.CO1_EL,borderType=cv2.BORDER_REPLICATE))
    #group_blur_mask = (cv2.morphologyEx(group_blur_mask,cv2.MORPH_CLOSE,CONST.CO1_EL,borderType=cv2.BORDER_REPLICATE))
    ##filter out small areas 
    #group_blur_mask = getSizeFilteredMask(group_blur_mask,min_size=BLUR_MIN_SIZE)
    #set up the plots to return
    plot_img_layer = img_array[:,:,brightest_layer_n-1]
    im_gs = (plot_img_layer*group_blur_mask).astype(np.float32); im_gs /= np.max(im_gs)
    overlay_gs = np.array([im_gs,im_gs,0.15*group_blur_mask]).transpose(1,2,0)
    norm = 128./np.mean(plot_img_layer[group_blur_mask==1]); im_c = (np.clip(norm*plot_img_layer,0,255)).astype(np.uint8)
    overlay_c = np.array([im_c,im_c*group_blur_mask,im_c*group_blur_mask]).transpose(1,2,0)
    plots = [{'image':plot_img_layer,'title':f'raw IMAGE layer {brightest_layer_n}'},
             {'image':overlay_c,'title':f'layer {layer_group_bounds[0]}-{layer_group_bounds[1]} blur mask overlay (clipped)'},
             {'image':overlay_gs,'title':f'layer {layer_group_bounds[0]}-{layer_group_bounds[1]} blur mask overlay (grayscale)'},
             {'image':brightest_layer_nlv,'title':'local variance of normalized laplacian'},
             {'hist':brightest_layer_nlv.flatten(),'xlabel':'variance of normalized laplacian','line_at':BLUR_NLV_CUT},
             {'image':stacked_masks,'title':f'stacked layer masks (cut at {flag_cut})','cmap':'gist_ncar','vmin':0,'vmax':layer_group_bounds[1]-layer_group_bounds[0]+1},
             {'image':group_blur_mask,'title':f'layer {layer_group_bounds[0]}-{layer_group_bounds[1]} blur mask','vmin':0,'vmax':1},
            ]
    #return the blur mask for the layer and the dictionary of plots
    return group_blur_mask, plots

#################### HELPER FUNCTIONS ####################

#helper function to write out a sheet of masking information plots for an image
def doMaskingPlotsForImage(image_key,tissue_mask,plot_dict_lists,full_mask,workingdir=None) :
    #figure out how many rows/columns will be in the sheet and set up the plots
    n_rows = len(plot_dict_lists)+1
    n_cols = max(n_rows,len(plot_dict_lists[0]))
    for pdi in range(1,len(plot_dict_lists)) :
        if len(plot_dict_lists[pdi]) > n_cols :
            n_cols = len(plot_dict_lists[pdi])
    f,ax = plt.subplots(n_rows,n_cols,figsize=(n_cols*6.4,n_rows*tissue_mask.shape[0]/tissue_mask.shape[1]*6.4))
    #add the masking plots for each layer group
    for row,plot_dicts in enumerate(plot_dict_lists) :
        for col,pd in enumerate(plot_dicts) :
            dkeys = pd.keys()
            #imshow plots
            if 'image' in dkeys :
                imshowkwargs = {}
                possible_keys = ['cmap','vmin','vmax']
                for pk in possible_keys :
                    if pk in dkeys :
                        imshowkwargs[pk]=pd[pk]
                pos = ax[row][col].imshow(pd['image'],**imshowkwargs)
                f.colorbar(pos,ax=ax[row][col])
                if 'title' in dkeys :
                    title_text = pd['title'].replace('IMAGE',image_key)
                    ax[row][col].set_title(title_text)
            #histogram plots
            elif 'hist' in dkeys :
                ax[row][col].hist(pd['hist'],100)
                if 'xlabel' in dkeys :
                    xlabel_text = pd['xlabel'].replace('IMAGE',image_key)
                    ax[row][col].set_xlabel(xlabel_text)
                if 'line_at' in dkeys :
                    ax[row][col].plot([pd['line_at'],pd['line_at']],
                                    [0.8*y for y in ax[row][col].get_ylim()],
                                    linewidth=2,color='tab:red',label=pd['line_at'])
                    ax[row][col].legend(loc='best')
            #remove axes from any extra slots
            if col==len(plot_dicts)-1 and col<n_cols-1 :
                for ci in range(col+1,n_cols) :
                    ax[row][ci].axis('off')
    #add the plot of the overlaid tissue and layer group masks
    superimposed_masks = tissue_mask
    max_sim_val = 1
    for mci,plot_dicts in enumerate(plot_dict_lists) :
        for pd in plot_dicts :
            if 'image' in pd.keys() and 'title' in pd.keys() and pd['title'].endswith('blur mask') :
                superimposed_masks+=(2**(mci+1))*pd['image']
                max_sim_val+=(2**(mci+1))
    pos = ax[n_rows-1][0].imshow(superimposed_masks,vmin=0.,vmax=max_sim_val,cmap='gist_ncar')
    f.colorbar(pos,ax=ax[n_rows-1][0])
    ax[n_rows-1][0].set_title('superimposed masks')
    #add the plots of the enumerated mask layer groups
    enumerated_mask_max = np.max(full_mask)
    for lgi,lgb in enumerate(MASK_LAYER_GROUPS) :
        pos = ax[n_rows-1][lgi+1].imshow(full_mask[:,:,lgb[0]-1],vmin=0.,vmax=enumerated_mask_max,cmap='gist_ncar')
        f.colorbar(pos,ax=ax[n_rows-1][lgi+1])
        ax[n_rows-1][lgi+1].set_title(f'full mask, layers {lgb[0]}-{lgb[1]}')
    #empty the other unused axes in the last row
    for ci in range(n_rows,n_cols) :
        ax[n_rows-1][ci].axis('off')
    #show/save the plot
    if workingdir is None :
        plt.show()
    else :
        with cd(workingdir) :
            fn = f'{image_key}_masking_plots.png'
            plt.savefig(fn); plt.close(); cropAndOverwriteImage(fn)

#helper function to change a mask from zeroes and ones to region indices and zeroes
def getEnumeratedMask(layer_mask,start_i) :
    #first invert the mask to get the "bad" regions as "signal"
    inverted_mask = np.zeros_like(layer_mask); inverted_mask[layer_mask==0] = 1; inverted_mask[layer_mask==1] = 0
    #label each connected region uniquely starting at the supplied index
    n_labels, labels_im = cv2.connectedComponents(inverted_mask)
    return_mask = np.zeros_like(layer_mask)
    for label_i in range(1,n_labels) :
        return_mask[labels_im==label_i] = start_i+label_i-1
    #return the mask
    return return_mask

#helper function to calculate and add the subimage infos for a single image to a shared dictionary (run in parallel)
def getLabelledMaskRegionsWorker(img_array,key,thresholds,xpos,ypos,pscale,workingdir,return_list) :
    #start by creating the tissue mask
    tissue_mask = getImageTissueMask(img_array,thresholds)
    #next make blur masks for each of the layer groups
    blur_masks_by_layer_group = []; blur_mask_plots_by_layer_group = []
    for lgi,lgb in enumerate(MASK_LAYER_GROUPS) :
        lgbm, lgbmps = getImageLayerGroupBlurMaskAndPlots(img_array,lgb,BRIGHTEST_LAYERS[lgi],BLUR_MASK_FLAG_CUTS[lgi])
        blur_masks_by_layer_group.append(lgbm)
        blur_mask_plots_by_layer_group.append(lgbmps)
    #combine the layer group blur masks to get the final mask for all layers
    stacked_blur_masks = np.zeros_like(blur_masks_by_layer_group[0])
    for layer_group_blur_mask in blur_masks_by_layer_group :
        stacked_blur_masks+=layer_group_blur_mask
    final_blur_mask = np.where(stacked_blur_masks<BLUR_MIN_LAYER_GROUPS,1,0)
    #small open/close to refine it
    final_blur_mask = (cv2.morphologyEx(final_blur_mask,cv2.MORPH_OPEN,CONST.CO1_EL,borderType=cv2.BORDER_REPLICATE))
    final_blur_mask = (cv2.morphologyEx(final_blur_mask,cv2.MORPH_CLOSE,CONST.CO1_EL,borderType=cv2.BORDER_REPLICATE))
    #remove any remaining small spots
    final_blur_mask = getSizeFilteredMask(final_blur_mask,min_size=BLUR_MIN_SIZE)
    #if there is anything flagged in the final blur mask, write out some plots and the mask file/csv file lines
    if np.min(final_blur_mask)<1 :
        #figure out where the image is in cellview
        key_x = float(key.split(',')[0].split('[')[1])
        key_y = float(key.split(',')[1].split(']')[0])
        cvx = pscale*key_x-xpos
        cvy = pscale*key_y-ypos
        #the mask starts as all ones (0=background, 1=good tissue, >=2 is a flagged region)
        output_mask = np.ones(img_array.shape,dtype=np.uint8)
        #add in the blur mask, starting with index 2
        start_i = 2
        layers_string = '-1'
        enumerated_mask = getEnumeratedMask(final_blur_mask,start_i)
        for li in range(img_array.shape[-1]) :
            output_mask[:,:,li] = np.where(enumerated_mask!=0,enumerated_mask,output_mask[:,:,li])
        start_i = np.max(enumerated_mask)+1
        #add a line for each region to the return_list (will be written to the .csv file)
        region_indices = list(range(np.min(enumerated_mask[enumerated_mask!=0]),np.max(enumerated_mask)+1))
        for ri in region_indices :
            r_size = np.sum(enumerated_mask==ri)
            return_list.append(LabelledMaskRegion(key,cvx,cvy,ri,layers_string,r_size,BLUR_FLAG_STRING))
        ##add in the masks for each layer group, starting with index 2
        #start_i = 2
        #for lgi,lgb in enumerate(MASK_LAYER_GROUPS) :
        #    if np.min(blur_masks_by_layer_group[lgi])==1 :
        #        continue
        #    layers_string = (''.join(f'{ln}-' for ln in range(lgb[0],lgb[1]+1)))[:-1]
        #    enumerated_mask = getEnumeratedMask(blur_masks_by_layer_group[lgi],start_i)
        #    for ln in range(lgb[0],lgb[1]+1) :
        #        (output_mask[:,:,ln-1])[output_mask[:,:,ln-1]==1] = (np.where(enumerated_mask!=0,enumerated_mask,output_mask[:,:,ln-1]))[output_mask[:,:,ln-1]==1]
        #    start_i = np.max(enumerated_mask)+1
        #    #add a line for each region to the return_list (will be written to the .csv file)
        #    region_indices = list(range(np.min(enumerated_mask[enumerated_mask!=0]),np.max(enumerated_mask)+1))
        #    for ri in region_indices :
        #        r_size = np.sum(enumerated_mask==ri)
        #        return_list.append(LabelledMaskRegion(key,cvx,cvy,ri,layers_string,r_size,BLUR_FLAG_STRING))
        #next add in the tissue mask (all the background is zero in every layer, unless already flagged otherwise)
        for li in range(img_array.shape[-1]) :
            output_mask[:,:,li] = np.where(output_mask[:,:,li]==1,tissue_mask,output_mask[:,:,li])
        #make and write out the plots for this image
        doMaskingPlotsForImage(key,tissue_mask,blur_mask_plots_by_layer_group,output_mask,workingdir)
        ##write out the mask in the working directory
        #with cd(workingdir) :
        #    writeImageToFile(output_mask,f'{key}_mask.png',dtype=np.uint8)

#helper function to get a list of all the labelled mask regions for a chunk of files
def getLabelledMaskRegionsForChunk(fris,metsbl,etcobl,thresholds,xpos,ypos,pscale,workingdir) :
    #get the image arrays
    img_arrays = readImagesMT(fris,smoothed=False,med_exposure_times_by_layer=metsbl,et_corr_offsets_by_layer=etcobl)
    #get all of the labelled mask region objects as the images are masked
    manager = mp.Manager()
    return_list = manager.list()
    procs = []
    for i,im_array in enumerate(img_arrays) :
        msg = f'Masking {fris[i].rawfile_path} {fris[i].sequence_print}'
        logger.info(msg)
        key = (os.path.basename(fris[i].rawfile_path)).rstrip(RAWFILE_EXT)
        p = mp.Process(target=getLabelledMaskRegionsWorker,args=(im_array,key,thresholds,xpos,ypos,pscale,workingdir,return_list))
        procs.append(p)
        p.start()
    for proc in procs:
        proc.join()
    #return just a regular list of all the labelled region info objects
    return list(return_list)

#helper function to make sure all the necessary information is available from the command line arguments
def checkArgs(args) :
    #rawfile_top_dir/[slideID] must exist
    rawfile_dir = os.path.join(args.rawfile_top_dir,args.slideID)
    if not os.path.isdir(rawfile_dir) :
        raise ValueError(f'ERROR: rawfile directory {rawfile_dir} does not exist!')
    #root dir must exist
    if not os.path.isdir(args.root_dir) :
        raise ValueError(f'ERROR: root_dir argument ({args.root_dir}) does not point to a valid directory!')
    #images must be corrected for exposure time, and exposure time correction file must exist
    if (args.skip_exposure_time_correction) :   
        raise ValueError('ERROR: exposure time offset file must be provided.')
    if not os.path.isfile(args.exposure_time_offset_file) :
        raise ValueError(f'ERROR: exposure_time_offset_file {args.exposure_time_offset_file} does not exist!')
    #need the threshold file
    if args.threshold_file_dir is None :
        raise ValueError('ERROR: must provide a threshold file dir.')
    tfp = os.path.join(args.threshold_file_dir,f'{args.slideID}_background_thresholds.txt')
    if not os.path.isfile(tfp) :
        raise ValueError(f'ERROR: threshold file path {tfp} does not exist!')
    #create the working directory if it doesn't already exist
    if not os.path.isdir(args.workingdir) :
        os.mkdir(args.workingdir)

#################### MAIN SCRIPT ####################

def main(args=None) :
    parser = ArgumentParser()
    #add the common options to the parser
    addCommonArgumentsToParser(parser,flatfielding=False,warping=False)
    #threshold file directory
    parser.add_argument('--threshold_file_dir',
                        help='Path to the directory holding the slide [slideID]_background_thresholds.txt file')
    #group for other run options
    run_option_group = parser.add_argument_group('run options', 'other options for this run')
    run_option_group.add_argument('--n_threads', default=10, type=int,
                                  help='Maximum number of threads to use in reading/processing files')
    run_option_group.add_argument('--max_files', default=-1, type=int,
                                  help='Maximum number of files to use (default = -1 runs all files for the requested slide)')
    run_option_group.add_argument('--max_masks', default=-1, type=int,
                                  help='Maximum number of masks to write out (prevents writing too much while testing, default=-1 writes out everything)')
    args = parser.parse_args(args=args)
    #check the arguments
    checkArgs(args)
    #get all the rawfile paths
    with cd(os.path.join(args.rawfile_top_dir,args.slideID)) :
        all_rfps = [os.path.join(args.rawfile_top_dir,args.slideID,fn) for fn in glob.glob(f'*{RAWFILE_EXT}')]
    if args.max_files>0 :
        all_rfps=all_rfps[:args.max_files]
    #get the correction details and other slide information stuff
    dims   = getImageHWLFromXMLFile(args.root_dir,args.slideID)
    metsbl = getSlideMedianExposureTimesByLayer(args.root_dir,args.slideID)
    etcobl = [lo.offset for lo in readtable(args.exposure_time_offset_file,LayerOffset)]
    with open(os.path.join(args.threshold_file_dir,f'{args.slideID}_background_thresholds.txt')) as fp :
        bgtbl = [int(v) for v in fp.readlines() if v!='']
    if len(bgtbl)!=dims[-1] :
        raise RuntimeError(f'ERROR: found {len(bgtbl)} background thresholds but images have {dims[-1]} layers!')
    thresholds = [bgtbl[li] for li in range(dims[-1])]
    slide_constants_dict = constantsdict(os.path.join(args.root_dir,args.slideID,'dbload',f'{args.slideID}_constants.csv'))
    xpos = float(units.pixels(slide_constants_dict['xposition']))
    ypos = float(units.pixels(slide_constants_dict['yposition']))
    pscale = slide_constants_dict['pscale']
    #chunk up the rawfile read information 
    if args.max_files!=-1 :
        all_rfps = all_rfps[:args.max_files]
    fri_chunks = chunkListOfFilepaths(all_rfps,dims,args.root_dir,args.n_threads)
    #get the masked regions for each chunk
    all_lmrs = []
    for fri_chunk in fri_chunks :
        new_lmrs = getLabelledMaskRegionsForChunk(fri_chunk,metsbl,etcobl,thresholds,xpos,ypos,pscale,args.workingdir)
        all_lmrs+=new_lmrs
        if len(set(lmr.image_key for lmr in all_lmrs))>=args.max_masks :
            break
    #write out the table with all of the labelled region information
    if len(all_lmrs)>0 :
        fn = f'{args.slideID}_labelled_mask_regions.csv'
        with cd(args.workingdir) :
            writetable(fn,sorted(all_lmrs,key=lambda x: f'{x.image_key}_{x.region_index}'))

if __name__=='__main__' :
    main()
