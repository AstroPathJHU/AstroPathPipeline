#imports
from .alignmentset import AlignmentSetForExposureTime
from .utilities import getFirstLayerInGroup, getOverlapsWithExposureTimeDifferences
from .config import CONST
from ..flatfield.utilities import FlatfieldSlideInfo
from ..utilities.img_file_io import LayerOffset, getExposureTimesByLayer, getImageHWLFromXMLFile, getRawAsHWL, getSlideMedianExposureTimesByLayer
from ..utilities.img_correction import correctImageLayerForExposureTime
from ..utilities.tableio import readtable, writetable
from ..utilities.misc import cd
from argparse import ArgumentParser
import numpy as np, multiprocessing as mp
import os, time, logging, glob, dataclasses, platform

#################### CONSTANTS ####################

LOGFILE_STEM = 'evaluate_exposure_time_log.txt'
RESULT_FILE_STEM = 'overlap_correction_results.csv'

#################### UTILITIES ####################

#helper dataclass for results of the exposure time corrections
@dataclasses.dataclass
class OverlapCorrectionResult :
    slideID        : str
    layer_n        : int
    overlap_n      : int
    n_pixels       : float
    et_diff        : float
    raw_cost       : float
    raw_diff       : float
    naive_cost     : float
    naive_diff     : float
    corrected_cost : float
    corrected_diff : float

#helper function to get a list of nlayers dictionaries holding the exposure times in each layer keyed by filename stem
def getExposureTimeDicts(samp_name,rtd,nlayers) :
    return_list = []
    for li in range(nlayers) :
        return_list.append({})
    with cd(os.path.join(rtd,samp_name)) :
        fps = [os.path.join(rtd,samp_name,fn) for fn in glob.glob(f'*{CONST.RAW_EXT}')]
    for fp in fps :
        fstem = os.path.basename(os.path.normpath(fp)).rstrip(CONST.RAW_EXT)
        this_file_exp_times = getExposureTimesByLayer(fp,nlayers,rtd)
        for li in range(nlayers) :
            return_list[li][fstem] = this_file_exp_times[li]
    return return_list

#helper function to get a single overlap's result 
def getOverlapResult(sid,layer,overlap,exp_times,med_exp_time,offset,fss_by_rect_n) :
    raw_p1, raw_p2 = overlap.shifted
    npixels = 0.5*((raw_p1.shape[0]*raw_p1.shape[1])+(raw_p2.shape[0]*raw_p2.shape[1]))
    p1_et = exp_times[fss_by_rect_n[overlap.p1]]
    p2_et = exp_times[fss_by_rect_n[overlap.p2]]
    naive_p1 = correctImageLayerForExposureTime(raw_p1,p1_et,med_exp_time,0.)
    naive_p2 = correctImageLayerForExposureTime(raw_p2,p2_et,med_exp_time,0.)
    corr_p1 = correctImageLayerForExposureTime(raw_p1,p1_et,med_exp_time,offset)
    corr_p2 = correctImageLayerForExposureTime(raw_p2,p2_et,med_exp_time,offset)
    raw_cost = np.sum(np.abs(raw_p1-raw_p2))/npixels
    raw_diff = np.sum(np.abs((raw_p1/p1_et)-(raw_p2/p2_et)))/npixels
    naive_cost = np.sum(np.abs(naive_p1-naive_p2))/npixels
    naive_diff = np.sum(np.abs((naive_p1-naive_p2)/med_exp_time))/npixels
    corr_cost = np.sum(np.abs(corr_p1-corr_p2))/npixels
    corr_diff = np.sum(np.abs((corr_p1-corr_p2)/med_exp_time))/npixels
    return OverlapCorrectionResult(sid,layer,overlap.n,npixels,(p1_et-p2_et),raw_cost,raw_diff,naive_cost,naive_diff,corr_cost,corr_diff)

#helper function to get a list of correction results for a single slide
def writeResultsForSlide(slide,offsets,ff_file,workingdir,smoothsigma,allow_edges) :
    #make a logger
    logfile_path = os.path.join(workingdir,f'{slide.name}_{LOGFILE_STEM}')
    logger = logging.getLogger(f'evaluate_exposure_time_{slide.name}')
    logger.setLevel(logging.DEBUG)
    logformat = logging.Formatter("[%(asctime)s] %(message)s  [%(funcName)s]","%Y-%m-%d %H:%M:%S")
    streamhandler = logging.StreamHandler(); streamhandler.setFormatter(logformat); logger.addHandler(streamhandler)
    filehandler = logging.FileHandler(logfile_path); filehandler.setFormatter(logformat); logger.addHandler(filehandler)
    #check to see if this slide is already done
    skip = False
    done_msg = f'Done evaluating exposure time results for {slide.name}'
    if os.path.isfile(logfile_path) :
        with open(logfile_path,'r') as fp :
            for line in fp.readlines() :
                if done_msg in line :
                    skip=True
                    break
        if skip :
            logger.info(done_msg)
            return                    
    #get the image dimensions for files from this slide
    h,w,nlayers = getImageHWLFromXMLFile(slide.rawfile_top_dir,slide.name)
    #read the flatfield from the file
    flatfield = getRawAsHWL(ff_file,h,w,nlayers,CONST.FLATFIELD_DTYPE)
    #get the median exposure times for this slide by layer
    logger.info(f'Getting median exposure times for {slide.name}')
    med_exp_times_by_layer = getSlideMedianExposureTimesByLayer(slide.rawfile_top_dir,slide.name)
    #get all of the exposure times keyed by file stem
    logger.info(f'Getting all exposure times for {slide.name}')
    exp_time_dicts = getExposureTimeDicts(slide.name,slide.rawfile_top_dir,nlayers)
    #start up the dictionary of overlaps with different exposure times by layer group
    olaps_with_et_diffs_dict = {}
    #for each layer
    for li in range(nlayers) :
        #see if this layer has already been done
        output_fn = f'{slide.name}_layer_{li+1}_{RESULT_FILE_STEM}'
        if os.path.isfile(os.path.join(workingdir,output_fn)) :
            logger.info(f'Skipping {slide.name} layer {li+1}; results file already exists.')
            continue
        skip = False
        no_offset_skip_msg = f'No offset found for layer {li+1}; skipping results in this layer'
        mult_offset_skip_msg = f'Multiple offsets found for layer {li+1}; skipping results in this layer'
        no_overlaps_skip_msg = f'Skipping {slide.name} layer {li+1} (no overlaps with exposure time differences)'
        with open(logfile_path,'r') as fp :
            for line in fp.readlines() :
                if (no_offset_skip_msg in line) or (mult_offset_skip_msg in line) or (no_overlaps_skip_msg) in line :
                    skip=True
                    break
        if skip :
            continue
        #get the offset to compare with for this layer
        this_layer_offset = [o.offset for o in offsets if o.layer_n==li+1]
        if len(this_layer_offset)<1 :
            logger.info(no_offset_skip_msg)
            continue
        elif len(this_layer_offset)>1 :
            logger.info(mult_offset_skip_msg)
            continue
        else :
            this_layer_offset = this_layer_offset[0]
        #find the overlaps with different exposure times
        if getFirstLayerInGroup(li+1,nlayers) in olaps_with_et_diffs_dict.keys() :
            olaps_with_et_diffs = olaps_with_et_diffs_dict[getFirstLayerInGroup(li+1,nlayers)]
        else :
            olaps_with_et_diffs = getOverlapsWithExposureTimeDifferences(slide.rawfile_top_dir,slide.metadata_top_dir,slide.name,
                                                                         exp_time_dicts[li],li+1,include_tissue_edges=allow_edges)
            olaps_with_et_diffs_dict[getFirstLayerInGroup(li+1,nlayers)] = olaps_with_et_diffs
        if len(olaps_with_et_diffs)<1 :
            logger.info(no_overlaps_skip_msg)
            continue
        #start the list of results
        these_results = []
        #make an AlignmentSet for just those overlaps and align it
        logger.info(f'Getting {len(olaps_with_et_diffs)} overlaps with different exposure times for {slide.name} layer {li+1}')
        use_GPU = platform.system()!='Darwin'
        a = AlignmentSetForExposureTime(slide.metadata_top_dir,slide.rawfile_top_dir,slide.name,
                                        selectoverlaps=olaps_with_et_diffs,onlyrectanglesinoverlaps=True,
                                        nclip=CONST.N_CLIP,useGPU=use_GPU,readlayerfile=False,layer=li+1,filetype='raw',
                                        smoothsigma=smoothsigma,flatfield=flatfield[:,:,li])
        a.getDAPI()
        logger.info(f'Aligning {slide.name} layer {li+1} overlaps with corrected/smoothed images....')
        a.align(alreadyalignedstrategy='overwrite')
        #make the dictionary of rectangle file stems by number
        filestems_by_rect_n = {}
        for r in a.rectangles :
            fs = r.file.rstrip('.im3')
            filestems_by_rect_n[r.n] = fs
        #add the results from each overlap
        for io,olap in enumerate(a.overlaps,start=1) :
            if olap.result.exit!=0 :
                logger.info(f'Skipping overlap {olap.n} ({io} of {len(a.overlaps)}) in {slide.name} layer {li+1} (not aligned)')
                continue
            if (olap.p1 not in filestems_by_rect_n.keys()) or (olap.p2 not in filestems_by_rect_n.keys()) :
                logger.info(f'Skipping overlap {olap.n} ({io} of {len(a.overlaps)}) in {slide.name} layer {li+1} (missing rectangle exposure time)')
                continue
            logger.info(f'Getting results for overlap {olap.n} ({io} of {len(a.overlaps)}) in {slide.name} layer {li+1}')
            these_results.append(getOverlapResult(slide.name,li+1,olap,exp_time_dicts[li],med_exp_times_by_layer[li],this_layer_offset,filestems_by_rect_n))
        with cd(workingdir) :
            writetable(output_fn,these_results,logger=logger)
    logger.info(done_msg)

#################### MAIN SCRIPT ####################

def main(args=None) :
    #define and get the command-line arguments
    parser = ArgumentParser()
    parser.add_argument('slides',
                        help='Path to .csv file listing FlatfieldSlideInfo objects to use slides from multiple raw/metadata file paths')
    parser.add_argument('exposure_time_offset_file',
                        help='Path to the .csv file specifying layer-dependent exposure time correction offsets for the slides in question')
    parser.add_argument('flatfield_file',
                        help='Path to the .bin file of the flatfield corrections to apply')
    parser.add_argument('workingdir_name', 
                        help='Name of working directory to save created files in')
    parser.add_argument('--n_threads',             default=5,   type=int,         
                        help='Maximum number of threads/processes to run at once (different slides run in parallel).')
    parser.add_argument('--smooth_sigma',         default=3., type=float,
                        help='sigma (in pixels) for initial Gaussian blur of images')
    parser.add_argument('--allow_edge_HPFs', action='store_true',
                        help='Add this flag to allow overlaps with HPFs on the tissue edges')
    args = parser.parse_args(args=args)
    #read in all the slides and the exposure time offsets
    slides = readtable(args.slides,FlatfieldSlideInfo)
    offsets = readtable(args.exposure_time_offset_file,LayerOffset)
    #make the working directory
    if not os.path.isdir(args.workingdir_name) :
        os.mkdir(args.workingdir_name)
    #process each slide
    if args.n_threads<=1 :
        for slide in slides :
            writeResultsForSlide(slide,offsets,args.flatfield_file,args.workingdir_name,args.smooth_sigma,args.allow_edge_HPFs)
    else :
        procs = []
        for slide in slides :
            p = mp.Process(target=writeResultsForSlide,
                           args=(slide,offsets,args.flatfield_file,args.workingdir_name,args.smooth_sigma,args.allow_edge_HPFs))
            p.start()
            procs.append(p)
            while len(procs)>=args.n_threads :
                for proc in procs :
                    if not proc.is_alive() :
                        proc.join()
                        delete_p = procs.pop(procs.index(proc))
                        delete_p = delete_p
                        del delete_p
                time.sleep(10)
        for proc in procs :
            proc.join()

if __name__=='__main__' :
    main()
