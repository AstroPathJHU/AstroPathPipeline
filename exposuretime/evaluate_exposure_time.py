from .alignmentset import AlignmentSetForExposureTime
from .utilities import getOverlapsWithExposureTimeDifferences
from .config import CONST
from ..flatfield.utilities import FlatfieldSampleInfo
from ..utilities.img_file_io import LayerOffset, getExposureTimesByLayer, getImageHWLFromXMLFile, getRawAsHWL, getSampleMedianExposureTimesByLayer, correctImageLayerForExposureTime
from ..utilities.tableio import readtable, writetable
from ..utilities.misc import cd
from argparse import ArgumentParser
import numpy as np, multiprocessing as mp
import os, time, logging, glob, dataclasses

#################### CONSTANTS ####################

LOGFILE_STEM = 'evaluate_exposure_time_log.txt'
RESULT_FILE_STEM = 'overlap_correction_results.csv'

#################### UTILITIES ####################

#helper dataclass for results of the exposure time corrections
@dataclasses.dataclass
class OverlapCorrectionResult :
    sample         : str
    layer_n        : int
    overlap_n      : int
    n_pixels       : int
    et_diff        : float
    raw_cost       : float
    naive_cost     : float
    corrected_cost : float

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
def getOverlapResult(sn,layer,overlap,exp_times,med_exp_time,offset,fss_by_rect_n) :
    raw_p1, raw_p2 = overlap.shifted
    npixels = 0.5*((raw_p1.shape[0]*raw_p1.shape[1])+(raw_p2.shape[0]*raw_p2.shape[1]))
    p1_et = exp_times[fss_by_rect_n[overlap.p1]]
    p2_et = exp_times[fss_by_rect_n[overlap.p2]]
    naive_p1 = correctImageLayerForExposureTime(raw_p1,p1_et,med_exp_time,0.)
    naive_p2 = correctImageLayerForExposureTime(raw_p2,p2_et,med_exp_time,0.)
    corr_p1 = correctImageLayerForExposureTime(raw_p1,p1_et,med_exp_time,offset)
    corr_p2 = correctImageLayerForExposureTime(raw_p2,p2_et,med_exp_time,offset)
    raw_cost = np.sum(np.abs(raw_p1-raw_p2))/npixels
    naive_cost = np.sum(np.abs(naive_p1-naive_p2))/npixels
    corr_cost = np.sum(np.abs(corr_p1-corr_p2))/npixels
    return OverlapCorrectionResult(sn,layer,overlap.n,npixels,(p1_et-p2_et),raw_cost,naive_cost,corr_cost)

#helper function to get a list of correction results for a single sample
def writeResultsForSample(sample,offsets,ff_file,workingdir,smoothsigma,allow_edges) :
    #make a logger
    logger = logging.getLogger(f'evaluate_exposure_time_{sample.name}')
    logger.setLevel(logging.DEBUG)
    logformat = logging.Formatter("[%(asctime)s] %(message)s  [%(funcName)s]","%Y-%m-%d %H:%M:%S")
    streamhandler = logging.StreamHandler(); streamhandler.setFormatter(logformat); logger.addHandler(streamhandler)
    filehandler = logging.FileHandler(os.path.join(workingdir,f'{sample.name}_{LOGFILE_STEM}')); filehandler.setFormatter(logformat); logger.addHandler(filehandler)
    #get the image dimensions for files from this sample
    h,w,nlayers = getImageHWLFromXMLFile(sample.rawfile_top_dir,sample.name)
    #read the flatfield from the file
    flatfield = getRawAsHWL(ff_file,h,w,nlayers,CONST.FLATFIELD_DTYPE)
    #get the median exposure times for this sample by layer
    logger.info(f'Getting median exposure times for {sample.name}')
    med_exp_times_by_layer = getSampleMedianExposureTimesByLayer(sample.rawfile_top_dir,sample.name)
    #get all of the exposure times keyed by file stem
    logger.info(f'Getting all exposure times for {sample.name}')
    exp_time_dicts = getExposureTimeDicts(sample.name,sample.rawfile_top_dir,nlayers)
    #for each layer
    for li in range(nlayers) :
        #get the offset to compare with for this layer
        this_layer_offset = [o.offset for o in offsets if o.layer_n==li+1]
        if len(this_layer_offset)<1 :
            logger.info(f'No offset found for layer {li+1}; skipping results in this layer')
            continue
        elif len(this_layer_offset)>1 :
            logger.info(f'Multiple offsets found for layer {li+1}; skipping results in this layer')
            continue
        else :
            this_layer_offset = this_layer_offset[0]
        #find the overlaps with different exposure times
        olaps_with_et_diffs = getOverlapsWithExposureTimeDifferences(sample.rawfile_top_dir,sample.metadata_top_dir,sample.name,
                                                                     exp_time_dicts[li],li+1,include_tissue_edges=allow_edges)
        if len(olaps_with_et_diffs)<1 :
            logger.info(f'Skipping {sample.name} layer {li+1} (no overlaps with exposure time differences)')
            continue
        #start the list of results
        these_results = []
        #make an AlignmentSet for just those overlaps and align it
        logger.info(f'Getting {len(olaps_with_et_diffs)} overlaps with different exposure times for {sample.name} layer {li+1}')
        a = AlignmentSetForExposureTime(sample.metadata_top_dir,sample.rawfile_top_dir,sample.name,
                                        selectoverlaps=olaps_with_et_diffs,onlyrectanglesinoverlaps=True,
                                        nclip=CONST.N_CLIP,readlayerfile=False,layer=li+1,filetype='raw',
                                        smoothsigma=smoothsigma,flatfield=flatfield[:,:,li])
        a.getDAPI()
        logger.info(f'Aligning {sample.name} layer {li+1} overlaps with corrected/smoothed images....')
        a.align(alreadyalignedstrategy='overwrite')
        #make the dictionary of rectangle file stems by number
        filestems_by_rect_n = {}
        for r in a.rectangles :
            fs = r.file.rstrip('.im3')
            filestems_by_rect_n[r.n] = fs
        #add the results from each overlap
        for io,olap in enumerate(a.overlaps,start=1) :
            if olap.result.exit!=0 :
                logger.info(f'Skipping overlap {olap.n} ({io} of {len(a.overlaps)}) in {sample.name} layer {li+1} (not aligned)')
                continue
            if (olap.p1 not in filestems_by_rect_n.keys()) or (olap.p2 not in filestems_by_rect_n.keys()) :
                logger.info(f'Skipping overlap {olap.n} ({io} of {len(a.overlaps)}) in {sample.name} layer {li+1} (missing rectangle exposure time)')
                continue
            logger.info(f'Getting results for overlap {olap.n} ({io} of {len(a.overlaps)}) in {sample.name} layer {li+1}')
            these_results.append(getOverlapResult(sample.name,li+1,olap,exp_time_dicts[li],med_exp_times_by_layer[li],this_layer_offset,filestems_by_rect_n))
        with cd(workingdir) :
            writetable(f'{sample.name}_layer_{li+1}_{RESULT_FILE_STEM}',these_results)
    logger.info(f'Done evaluating exposure time results for {sample.name}')

#################### MAIN SCRIPT ####################

def main() :
    #define and get the command-line arguments
    parser = ArgumentParser()
    parser.add_argument('samples',
                        help='Path to .csv file listing FlatfieldSampleInfo objects to use samples from multiple raw/metadata file paths')
    parser.add_argument('exposure_time_offset_file',
                        help='Path to the .csv file specifying layer-dependent exposure time correction offsets for the samples in question')
    parser.add_argument('flatfield_file',
                        help='Path to the .bin file of the flatfield corrections to apply')
    parser.add_argument('workingdir_name', 
                        help='Name of working directory to save created files in')
    parser.add_argument('--n_threads',             default=5,   type=int,         
                        help='Maximum number of threads/processes to run at once (different samples run in parallel).')
    parser.add_argument('--smooth_sigma',         default=3., type=float,
                        help='sigma (in pixels) for initial Gaussian blur of images')
    parser.add_argument('--allow_edge_HPFs', action='store_true',
                        help='Add this flag to allow overlaps with HPFs on the tissue edges')
    args = parser.parse_args()
    #read in all the samples and the exposure time offsets
    samples = readtable(args.samples,FlatfieldSampleInfo)
    offsets = readtable(args.exposure_time_offset_file,LayerOffset)
    #make the working directory
    if not os.path.isdir(args.workingdir_name) :
        os.mkdir(args.workingdir_name)
    #process each sample
    if args.n_threads<=1 :
        for sample in samples :
            writeResultsForSample(sample,offsets,args.flatfield_file,args.workingdir_name,args.smooth_sigma,args.allow_edge_HPFs)
    else :
        procs = []
        for sample in samples :
            p = mp.Process(target=writeResultsForSample,
                           args=(sample,offsets,args.flatfield_file,args.workingdir_name,args.smooth_sigma,args.allow_edge_HPFs))
            p.start()
            procs.append(p)
            if len(procs)>=args.n_threads :
                for proc in procs :
                    proc.join()
                    procs = []
        for proc in procs :
            proc.join()

if __name__=='__main__' :
    main()