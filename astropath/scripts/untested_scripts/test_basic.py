#imports
import pathlib, datetime
import numpy as np
from argparse import ArgumentParser
from astropath.utilities import units
from astropath.utilities.tableio import readtable
from astropath.utilities.img_file_io import get_raw_as_hwl, smooth_image_worker
from astropath.hpfs.imagecorrection.utilities import CorrectionModelTableEntry
from astropath.hpfs.flatfield.meanimagesample import MeanImageSample
from astropath.hpfs.warping.warpingsample import WarpingSample

#fast units setup
units.setup('fast')

#constants
APPROC = pathlib.Path('//bki04/astropath_processing')
CORRECTION_MODEL_FILE = APPROC/'AstroPathCorrectionModels.csv'
FLATFIELD_DIR = APPROC/'flatfield'

#helper functions

def timestamp() :
    return f'[{datetime.datetime.now().strftime("%m/%d/%Y %H:%M:%S")}]'

def get_arguments() :
    parser = ArgumentParser()
    parser.add_argument('root',type=pathlib.Path,help='the path to the "Clinical_Specimen" directory for the sample')
    parser.add_argument('rawfile_root',type=pathlib.Path,
                        help='the path to the root directoryholding raw files for the sample')
    parser.add_argument('slideID',help='the Slide ID of the sample')
    parser.add_argument('workingdir',type=pathlib.Path,help='path to the directory that should hold the output')
    parser.add_argument('--n_threads',type=int,default=10,help='The number of parallel threads to use')
    args = parser.parse_args()
    if not args.workingdir.is_dir() :
        args.workingdir.mkdir(parents=True)
    return args

def run_basic(samp,save_dirpath) :
    dims = (samp.fheight,samp.fwidth,samp.nlayersim3)
    return np.ones(dims,dtype=np.float64), np.ones(dims,dtype=np.float64)

def illumination_variation_plots(samp,sm_uncorr_mi,sm_mi_corr_mi,sm_basic_corr_mi,central=False,save_dirpath=None) :
    pass

def get_overlap_comparisons(samp,basic_ff,basic_df,save_dirpath) :
    return []

def overlap_mse_reduction_plots(overlap_comparisons_by_layer_n) :
    pass

#main function

def main() :
    #create the argument parser
    args = get_arguments()
    #create the mean image sample
    print(f'{timestamp()} creating MeanImageSample for {args.slideID}')
    meanimage_sample = MeanImageSample(args.root,samp=args.slideID,shardedim3root=args.rawfile_root,
                                       et_offset_file=None,
                                       #don't apply ANY corrections before running BaSiC
                                       skip_et_corrections=True, 
                                       flatfield_file=None,warping_file=None,correction_model_file=None,
                                       filetype='raw',
                                       )
    dims = (meanimage_sample.fheight,meanimage_sample.fwidth,meanimage_sample.nlayersim3)
    print(f'{timestamp()} done creating MeanImageSample for {args.slideID}')
    #create and save the basic flatfield
    print(f'{timestamp()} running BaSiC for {args.slideID}')
    basic_flatfield, basic_darkfield = run_basic(meanimage_sample,args.workingdir)
    print(f'{timestamp()} done running BaSiC for {args.slideID}')
    #create the illumination variation plots
    print(f'{timestamp()} getting meanimage flatfield and smoothing pre/post-correction meanimages for {args.slideID}')
    meanimage_fp = args.root/args.slideID/'im3'/'meanimage'/f'{args.slideID}-mean_image.bin'
    meanimage = get_raw_as_hwl(meanimage_fp,*dims,np.float64)
    correction_model_entries = readtable(CORRECTION_MODEL_FILE,CorrectionModelTableEntry)
    meanimage_ff_name = [te.FlatfieldVersion for te in correction_model_entries if te.SlideID==args.slideID]
    meanimage_ff_fp = FLATFIELD_DIR/f'flatfield_{meanimage_ff_name[0]}.bin'
    meanimage_ff = get_raw_as_hwl(meanimage_ff_fp,*dims,np.float64)
    smoothed_meanimage = smooth_image_worker(meanimage,100,gpu=True)
    mi_corrected_meanimage = meanimage/meanimage_ff
    smoothed_mi_corrected_meanimage = smooth_image_worker(mi_corrected_meanimage,100,gpu=True)
    basic_corrected_meanimage = meanimage/basic_flatfield
    smoothed_basic_corrected_meanimage = smooth_image_worker(basic_corrected_meanimage,100,gpu=True)
    print(f'{timestamp()} making meanimage illumination variation plots for {args.slideID}')
    illumination_variation_plots(meanimage_sample,
                                 smoothed_meanimage,
                                 smoothed_mi_corrected_meanimage,
                                 smoothed_basic_corrected_meanimage,
                                 central=False,
                                 save_dirpath=args.workingdir)
    #create the warping sample
    print(f'{timestamp()} creating warping sample for {args.slideID}')
    warping_sample = WarpingSample(args.root,samp=args.slideID,shardedim3root=args.rawfile_root,
                                   et_offset_file=None,
                                   skip_et_corrections=False,
                                   flatfield_file=meanimage_ff_fp,warping_file=None,correction_model_file=None,
                                   filetype='raw',
                                  )
    print(f'{timestamp()} getting overlap comparisons for {args.slideID}')
    overlap_comparisons = get_overlap_comparisons(warping_sample,basic_flatfield,basic_darkfield,args.workingdir)
    #create the overlap MSE reduction comparison plots
    overlap_mse_reduction_plots(overlap_comparisons)
    print(f'{timestamp()} Done')

if __name__=='__main__' :
    main()