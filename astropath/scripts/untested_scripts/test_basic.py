#imports
import pathlib, datetime, pybasic
import numpy as np, matplotlib.pyplot as plt
from argparse import ArgumentParser
from threading import Thread
from queue import Queue
from astropath.utilities import units
from astropath.utilities.miscplotting import save_figure_in_dir
from astropath.utilities.tableio import readtable
from astropath.utilities.img_file_io import get_raw_as_hwl, get_raw_as_hw, smooth_image_worker, write_image_to_file
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
    parser.add_argument('--no_darkfield',action='store_true',
                        help='include this flag to skip computing a BaSiC darkfield')
    args = parser.parse_args()
    if not args.workingdir.is_dir() :
        args.workingdir.mkdir(parents=True)
    return args

def add_rect_image_to_queue(rect,queue,layer_i) :
    with rect.using_corrected_im3() as im :
        im_layer = im[:,:,layer_i]
        queue.put(im_layer)

def run_basic(samp,save_dirpath,n_threads,no_darkfield) :
    dims = (samp.fheight,samp.fwidth,samp.nlayersim3)
    basic_ff_fp = save_dirpath/'basic_flatfield.bin'
    basic_df_fp = save_dirpath/'basic_darkfield.bin'
    if basic_ff_fp.is_file() and basic_df_fp.is_file() :
        print(f'{timestamp()} flatfield/darkfield found in files from a previous run in {save_dirpath}')
        basic_flatfield = get_raw_as_hwl(basic_ff_fp,*dims,np.float64)
        basic_darkfield = get_raw_as_hwl(basic_df_fp,*dims,np.float64)
    else :
        basic_flatfield = np.ones(dims,dtype=np.float64)
        basic_darkfield = np.zeros_like(basic_flatfield)
        for li in range(samp.nlayersim3) :
            basic_ff_layer_fp = save_dirpath/f'basic_flatfield_layer_{li+1}.bin'
            basic_df_layer_fp = save_dirpath/f'basic_darkfield_layer_{li+1}.bin'
            if basic_ff_layer_fp.is_file() and basic_df_layer_fp.is_file() :
                print(f'{timestamp()}   flat/darkfield layer {li+1} found in files from a previous run in {save_dirpath}')
                ff_layer = get_raw_as_hw(basic_ff_layer_fp,*dims[:-1],np.float64)
                df_layer = get_raw_as_hw(basic_df_layer_fp,*dims[:-1],np.float64)
            else :
                image_queue = Queue()
                threads = []
                print(f'{timestamp()}   getting rectangle images for layer {li+1} of {samp.SlideID}')
                for ir,r in enumerate(samp.tissue_bulk_rects) :
                    while len(threads)>=n_threads :
                        thread = threads.pop(0)
                        thread.join()
                    if ir%50==0 :
                        print(f'{timestamp()}       getting image for rectangle {ir}(/{len(samp.tissue_bulk_rects)})')
                    new_thread = Thread(target=add_rect_image_to_queue,args=(r,image_queue,li))
                    new_thread.start()
                    threads.append(new_thread)
                for thread in threads :
                    thread.join()
                image_queue.put(None)
                raw_layer_images = []
                image = image_queue.get()
                while image is not None :
                    raw_layer_images.append(image)
                    image = image_queue.get()
                print(f'{timestamp()}   running BaSiC for layer {li+1} of {samp.SlideID}')
                ff_layer, df_layer = pybasic.basic(raw_layer_images, darkfield=(not no_darkfield),max_reweight_iterations=50)
                write_image_to_file(ff_layer,basic_ff_layer_fp)
                write_image_to_file(df_layer,basic_df_layer_fp)
            basic_flatfield[:,:,li] = ff_layer
            if not no_darkfield :
                basic_darkfield[:,:,li] = df_layer
        print(f'{timestamp()} writing out BaSiC flatfield and darkfield for {samp.SlideID}')
        write_image_to_file(basic_flatfield,basic_ff_fp)
        write_image_to_file(basic_darkfield,basic_df_fp)
        if basic_ff_fp.is_file() and basic_df_fp.is_file() :
            for fp in save_dirpath.glob('basic_flatfield_layer_*.bin') :
                fp.unlink()
            for fp in save_dirpath.glob('basic_darkfield_layer_*.bin') :
                fp.unlink()
    print(f'{timestamp()} plotting BaSiC flatfield and darkfield layers for {samp.SlideID}')
    layer_dir = save_dirpath/'layer_plots'
    if not layer_dir.is_dir() :
        layer_dir.mkdir(parents=True)
    for li in range(samp.nlayersim3) :
        f,ax=plt.subplots(figsize=(7.,7.*dims[0]/dims[1]))
        pos=ax.imshow(basic_flatfield[:,:,li])
        ax.set_title(f'BaSiC flatfield, layer {li+1}')
        f.colorbar(pos,ax=ax,fraction=0.046,pad=0.04)
        save_figure_in_dir(plt,f'flatfield_layer_{li+1}.png',layer_dir)
        plt.close()
        f,ax=plt.subplots(figsize=(7.,7.*dims[0]/dims[1]))
        pos=ax.imshow(basic_darkfield[:,:,li])
        ax.set_title(f'BaSiC darkfield, layer {li+1}')
        f.colorbar(pos,ax=ax,fraction=0.046,pad=0.04)
        save_figure_in_dir(plt,f'darkfield_layer_{li+1}.png',layer_dir)
        plt.close()
    return basic_flatfield, basic_darkfield

def illumination_variation_plots(samp,sm_uncorr_mi,sm_mi_corr_mi,sm_basic_corr_mi,central=False,save_dirpath=None) :
    pass

def get_overlap_comparisons(samp,basic_ff,basic_df,save_dirpath) :
    return {}

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
    basic_flatfield, basic_darkfield = run_basic(meanimage_sample,args.workingdir,args.n_threads,args.no_darkfield)
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