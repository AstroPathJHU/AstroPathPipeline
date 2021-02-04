#imports
from astropath_calibration.flatfield.utilities import chunkListOfFilepaths, readImagesMT
from astropath_calibration.flatfield.config import CONST
from astropath_calibration.utilities.img_file_io import getImageHWLFromXMLFile, getSlideMedianExposureTimesByLayer, LayerOffset
from astropath_calibration.utilities.img_file_io import smoothImageWorker, getExposureTimesByLayer
#from astropath_calibration.utilities.img_file_io import writeImageToFile
from astropath_calibration.utilities.tableio import readtable, writetable
from astropath_calibration.utilities.misc import cd, addCommonArgumentsToParser, cropAndOverwriteImage
from astropath_calibration.utilities import units
from astropath_calibration.baseclasses.csvclasses import constantsdict
from astropath_calibration.utilities.dataclasses import MyDataClass
from argparse import ArgumentParser
import numpy as np, matplotlib.pyplot as plt, multiprocessing as mp
import logging, os, glob, cv2, dataclasses, scipy.stats

#constants
RAWFILE_EXT                = '.Data.dat'

#logger
logger = logging.getLogger("flagging_hpf_regions")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s  [%(funcName)s]","%Y-%m-%d %H:%M:%S"))
logger.addHandler(handler)

#################### HELPER FUNCTIONS ####################

#helper function to plot all of the hpf locations for a slide with their reasons for being flagged
def plotFlaggedHPFLocations(sid,all_rfps,all_lmrs,pscale,xpos,ypos,truncated,workingdir) :
    all_flagged_hpf_keys = [lmr.image_key for lmr in all_lmrs]
    hpf_identifiers = []
    for rfp in all_rfps :
        key = (os.path.basename(rfp)).rstrip(RAWFILE_EXT)
        key_x = float(key.split(',')[0].split('[')[1])
        key_y = float(key.split(',')[1].split(']')[0])
        cvx = pscale*key_x-xpos
        cvy = pscale*key_y-ypos
        if key in all_flagged_hpf_keys :
            key_strings = set([lmr.reason_flagged for lmr in all_lmrs if lmr.image_key==key])
            fold_flagged = 1 if FOLD_FLAG_STRING in key_strings else 0
            dust_flagged = 1 if DUST_STRING in key_strings else 0
            saturation_flagged = 1 if SATURATION_FLAG_STRING in key_strings else 0
            flagged_int = 1*fold_flagged+2*dust_flagged+4*saturation_flagged
        else :
            flagged_int = 0
        hpf_identifiers.append({'x':cvx,'y':cvy,'flagged':flagged_int})
    colors_by_flag_int = ['gray','royalblue','gold','limegreen','firebrick','mediumorchid','darkorange','aqua']
    labels_by_flag_int = ['not flagged','tissue fold flagged','single-layer dust flagged','dust and tissue folds',
                            'saturation flagged','saturation and tissue folds','saturation and dust','saturation and dust and tissue folds']
    w = max([identifier['x'] for identifier in hpf_identifiers])-min([identifier['x'] for identifier in hpf_identifiers])
    h = max([identifier['y'] for identifier in hpf_identifiers])-min([identifier['y'] for identifier in hpf_identifiers])
    if h>w :
        f,ax = plt.subplots(figsize=(((1.1*w)/(1.1*h))*9.6,9.6))
    else :
        f,ax = plt.subplots(figsize=(9.6,9.6*((1.1*h)/(1.1*w))))
    for i in range(len(colors_by_flag_int)) :
        hpf_ids_to_plot = [identifier for identifier in hpf_identifiers if identifier['flagged']==i]
        if len(hpf_ids_to_plot)<1 :
            continue
        ax.scatter([hpfid['x'] for hpfid in hpf_ids_to_plot],
                   [hpfid['y'] for hpfid in hpf_ids_to_plot],
                   marker='o',
                   color=colors_by_flag_int[i],
                   label=labels_by_flag_int[i])
    ax.set_xlim(ax.get_xlim()[0]-0.05*w,ax.get_xlim()[1]+0.05*w)
    ax.set_ylim(ax.get_ylim()[0]-0.05*h,ax.get_ylim()[1]+0.05*h)
    ax.invert_yaxis()
    title_text = f'{sid} HPF center locations, ({len([hpfid for hpfid in hpf_identifiers if hpfid["flagged"]!=0])} flagged out of {len(all_rfps)} read)'
    if truncated :
        title_text+=' (stopped early)'
    ax.set_title(title_text,fontsize=16)
    ax.legend(loc='best',fontsize=10)
    ax.set_xlabel('HPF CellView x position',fontsize=16)
    ax.set_ylabel('HPF CellView y position',fontsize=16)
    fn = f'{sid}_flagged_hpf_locations.png'
    with cd(workingdir) :
        plt.savefig(fn)
        cropAndOverwriteImage(fn)

#helper function to calculate and add the subimage infos for a single image to a shared dictionary (run in parallel)
def getLabelledMaskRegionsWorker(img_array,exposure_times,key,thresholds,xpos,ypos,pscale,workingdir,exp_time_hists,return_list) :
    #start by creating the tissue mask
    tissue_mask = getImageTissueMask(img_array,thresholds)
    #next get the tissue fold mask and its associated plots
    tissue_fold_mask,tissue_fold_plots_by_layer_group = getImageTissueFoldMask(img_array,exposure_times,tissue_mask,exp_time_hists,return_plots=True)
    #tissue_fold_mask,tissue_fold_plots_by_layer_group = np.ones(img_array.shape,dtype=np.uint8),None
    #get masks for the blurriest areas of the DAPI layer group
    sm_img_array = smoothImageWorker(img_array,1)
    dapi_dust_mask,dapi_dust_plots = getImageLayerGroupBlurMask(sm_img_array,
                                                                exposure_times,
                                                                MASK_LAYER_GROUPS[DAPI_LAYER_GROUP_INDEX],
                                                                DUST_NLV_CUT,
                                                                0.5*(MASK_LAYER_GROUPS[DAPI_LAYER_GROUP_INDEX][1]-MASK_LAYER_GROUPS[DAPI_LAYER_GROUP_INDEX][0]+1),
                                                                DUST_MAX_MEAN,
                                                                BRIGHTEST_LAYERS[DAPI_LAYER_GROUP_INDEX],
                                                                exp_time_hists[DAPI_LAYER_GROUP_INDEX],
                                                                False)
    #same morphology transformations as before
    dapi_dust_mask = getMorphedAndFilteredMask(dapi_dust_mask,tissue_mask,WINDOW_EL,DUST_MIN_PIXELS,DUST_MIN_SIZE)
    #make sure any regions in that mask are sufficiently exclusive w.r.t. what's already flagged
    dapi_dust_mask = getExclusiveMask(dapi_dust_mask,tissue_fold_mask,0.25)
    #dapi_dust_mask,dapi_dust_plots = np.ones(img_array.shape,dtype=np.uint8),None
    #get masks for the saturated regions in each layer group
    layer_group_saturation_masks = []; layer_group_saturation_mask_plots = []
    for lgi,lgb in enumerate(MASK_LAYER_GROUPS) :
        lgsm,saturation_mask_plots = getImageLayerGroupSaturationMask(img_array,
                                                                      exposure_times,
                                                                      lgb,
                                                                      SATURATION_INTENSITY_CUTS[lgi],
                                                                      (MASK_LAYER_GROUPS[lgi][1]-MASK_LAYER_GROUPS[lgi][0]),
                                                                      BRIGHTEST_LAYERS[lgi],
                                                                      exp_time_hists[lgi],
                                                                      return_plots=False)
        #lgsm,saturation_mask_plots = np.ones(img_array.shape,dtype=np.uint8), None
        layer_group_saturation_masks.append(lgsm)
        layer_group_saturation_mask_plots.append(saturation_mask_plots)
    #if there is anything flagged in the final masks, write out some plots and the mask file/csv file lines
    is_masked = np.min(tissue_fold_mask)<1 or np.min(dapi_dust_mask)<1
    if not is_masked :
        for lgsm in layer_group_saturation_masks :
            if np.min(lgsm)<1 :
                is_masked=True
                break
    if is_masked :
        #figure out where the image is in cellview
        key_x = float(key.split(',')[0].split('[')[1])
        key_y = float(key.split(',')[1].split(']')[0])
        cvx = pscale*key_x-xpos
        cvy = pscale*key_y-ypos
        #the mask starts as all ones (0=background, 1=good tissue, >=2 is a flagged region)
        output_mask = np.ones(img_array.shape,dtype=np.uint8)
        #add in the tissue fold mask, starting with index 2
        start_i = 2
        if np.min(tissue_fold_mask)<1 :
            layers_string = '-1'
            enumerated_fold_mask = getEnumeratedMask(tissue_fold_mask,start_i)
            for li in range(img_array.shape[-1]) :
                output_mask[:,:,li] = np.where(enumerated_fold_mask!=0,enumerated_fold_mask,output_mask[:,:,li])
            start_i = np.max(enumerated_fold_mask)+1
            region_indices = list(range(np.min(enumerated_fold_mask[enumerated_fold_mask!=0]),np.max(enumerated_fold_mask)+1))
            for ri in region_indices :
                r_size = np.sum(enumerated_fold_mask==ri)
                return_list.append(LabelledMaskRegion(key,cvx,cvy,ri,layers_string,r_size,FOLD_FLAG_STRING))
        #add in the mask for the dapi layer group
        if np.min(dapi_dust_mask)<1 :
            layers_string = f'{MASK_LAYER_GROUPS[DAPI_LAYER_GROUP_INDEX][0]}-{MASK_LAYER_GROUPS[DAPI_LAYER_GROUP_INDEX][1]}'
            enumerated_dapi_mask = getEnumeratedMask(dapi_dust_mask,start_i)
            for ln in range(MASK_LAYER_GROUPS[DAPI_LAYER_GROUP_INDEX][0],MASK_LAYER_GROUPS[DAPI_LAYER_GROUP_INDEX][1]+1) :
                output_mask[:,:,ln-1] = np.where(enumerated_dapi_mask!=0,enumerated_dapi_mask,output_mask[:,:,ln-1])
            start_i = np.max(enumerated_dapi_mask)+1
            region_indices = list(range(np.min(enumerated_dapi_mask[enumerated_dapi_mask!=0]),np.max(enumerated_dapi_mask)+1))
            for ri in region_indices :
                r_size = np.sum(enumerated_dapi_mask==ri)
                return_list.append(LabelledMaskRegion(key,cvx,cvy,ri,layers_string,r_size,DUST_STRING))
        #add in the saturation masks 
        for lgi,lgsm in enumerate(layer_group_saturation_masks) :
            if np.min(lgsm)<1 :
                layers_string = f'{MASK_LAYER_GROUPS[lgi][0]}-{MASK_LAYER_GROUPS[lgi][1]}'
                enumerated_sat_mask = getEnumeratedMask(lgsm,start_i)
                for ln in range(MASK_LAYER_GROUPS[lgi][0],MASK_LAYER_GROUPS[lgi][1]+1) :
                    output_mask[:,:,ln-1] = np.where(enumerated_sat_mask!=0,enumerated_sat_mask,output_mask[:,:,ln-1])
                start_i = np.max(enumerated_sat_mask)+1
                region_indices = list(range(np.min(enumerated_sat_mask[enumerated_sat_mask!=0]),np.max(enumerated_sat_mask)+1))
                for ri in region_indices :
                    r_size = np.sum(enumerated_sat_mask==ri)
                    return_list.append(LabelledMaskRegion(key,cvx,cvy,ri,layers_string,r_size,SATURATION_FLAG_STRING))
        #finally add in the tissue mask (all the background is zero in every layer, unless already flagged otherwise)
        for li in range(img_array.shape[-1]) :
            output_mask[:,:,li] = np.where(output_mask[:,:,li]==1,tissue_mask,output_mask[:,:,li])
        #make and write out the plots for this image
        all_plot_dict_lists = []
        if tissue_fold_plots_by_layer_group is not None :
            all_plot_dict_lists += tissue_fold_plots_by_layer_group
        if dapi_dust_plots is not None :
            all_plot_dict_lists += [dapi_dust_plots]
        sat_plots = []
        for saturation_mask_plot_dicts in layer_group_saturation_mask_plots :
            if saturation_mask_plot_dicts is not None :
                sat_plots.append(saturation_mask_plot_dicts)
        all_plot_dict_lists += sat_plots
        doMaskingPlotsForImage(key,tissue_mask,all_plot_dict_lists,output_mask,workingdir)
        ##write out the mask in the working directory
        #with cd(workingdir) :
        #    writeImageToFile(output_mask,f'{key}_mask.png',dtype=np.uint8)

#helper function to get a list of all the labelled mask regions for a chunk of files
def getLabelledMaskRegionsForChunk(fris,metsbl,etcobl,thresholds,xpos,ypos,pscale,workingdir,root_dir,exp_time_hists) :
    #get the image arrays
    img_arrays = readImagesMT(fris,smoothed=False,med_exposure_times_by_layer=metsbl,et_corr_offsets_by_layer=etcobl)
    #get all of the labelled mask region objects as the images are masked
    manager = mp.Manager()
    return_list = manager.list()
    procs = []
    for i,im_array in enumerate(img_arrays) :
        msg = f'Masking {fris[i].rawfile_path} {fris[i].sequence_print}'
        logger.info(msg)
        exp_times = getExposureTimesByLayer(fris[i].rawfile_path,im_array.shape[-1],root_dir)
        key = (os.path.basename(fris[i].rawfile_path)).rstrip(RAWFILE_EXT)
        p = mp.Process(target=getLabelledMaskRegionsWorker,args=(im_array,exp_times,key,thresholds,xpos,ypos,pscale,workingdir,exp_time_hists,return_list))
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
    all_exp_times = []
    for lgi in range(len(MASK_LAYER_GROUPS)) :
        all_exp_times.append([])
    for rfp in all_rfps :
        etsbl = getExposureTimesByLayer(rfp,dims[-1],args.root_dir)
        for lgi,lgb in enumerate(MASK_LAYER_GROUPS) :
            all_exp_times[lgi].append(etsbl[lgb[0]-1])
    exp_time_hists = []
    for lgi in range(len(MASK_LAYER_GROUPS)) :
        newhist,newbins = np.histogram(all_exp_times[lgi],bins=100)
        exp_time_hists.append((newhist,newbins))
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
    truncated=False
    all_lmrs = []
    for fri_chunk in fri_chunks :
        new_lmrs = getLabelledMaskRegionsForChunk(fri_chunk,metsbl,etcobl,thresholds,xpos,ypos,pscale,args.workingdir,args.root_dir,exp_time_hists)
        all_lmrs+=new_lmrs
        if args.max_masks!=-1 and len(set(lmr.image_key for lmr in all_lmrs))>=args.max_masks :
            truncated=True
            break
    #write out the table with all of the labelled region information
    if len(all_lmrs)>0 :
        fn = f'{args.slideID}_labelled_mask_regions.csv'
        with cd(args.workingdir) :
            writetable(fn,sorted(all_lmrs,key=lambda x: f'{x.image_key}_{x.region_index}'))
    #make the plot of all the HPF locations and whether they have something masked/flagged
    plotFlaggedHPFLocations(args.slideID,all_rfps,all_lmrs,pscale,xpos,ypos,truncated,args.workingdir)

if __name__=='__main__' :
    main()
