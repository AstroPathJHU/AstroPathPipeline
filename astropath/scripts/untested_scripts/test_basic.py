#imports
import pathlib, datetime, pybasic
import numpy as np, matplotlib.pyplot as plt
from argparse import ArgumentParser
from threading import Thread
from queue import Queue
from dataclasses import dataclass
from astropath.utilities import units
from astropath.utilities.miscplotting import save_figure_in_dir
from astropath.utilities.tableio import readtable
from astropath.utilities.img_file_io import get_raw_as_hwl, get_raw_as_hw, smooth_image_worker, write_image_to_file
from astropath.hpfs.imagecorrection.utilities import CorrectionModelTableEntry
from astropath.hpfs.flatfield.meanimagesample import MeanImageSample
from astropath.hpfs.warping.warpingsample import WarpingSample

#fast units setup
units.setup('fast')

#alignment overlap comparison dataclass
@dataclass
class AlignmentOverlapComparison :
    n : int
    tag : int
    npix : int
    orig_dx : float
    orig_dy : float
    orig_mse1 : float
    orig_mse_diff : float
    basic_dx : float
    basic_dy : float
    basic_mse1 : float
    basic_mse_diff : float
    meanimage_dx : float
    meanimage_dy : float
    meanimage_mse1 : float
    meanimage_mse_diff : float 

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
        ff_plot_fn = f'flatfield_layer_{li+1}.png'
        if not (layer_dir/ff_plot_fn).is_file() :
            f,ax=plt.subplots(figsize=(7.,7.*dims[0]/dims[1]))
            pos=ax.imshow(basic_flatfield[:,:,li])
            ax.set_title(f'BaSiC flatfield, layer {li+1}')
            f.colorbar(pos,ax=ax,fraction=0.046,pad=0.04)
            save_figure_in_dir(plt,ff_plot_fn,layer_dir)
            plt.close()
        df_plot_fn = f'darkfield_layer_{li+1}.png'
        if not (layer_dir/df_plot_fn).is_file() :
            f,ax=plt.subplots(figsize=(7.,7.*dims[0]/dims[1]))
            pos=ax.imshow(basic_darkfield[:,:,li])
            ax.set_title(f'BaSiC darkfield, layer {li+1}')
            f.colorbar(pos,ax=ax,fraction=0.046,pad=0.04)
            save_figure_in_dir(plt,df_plot_fn,layer_dir)
            plt.close()
    return basic_flatfield, basic_darkfield

def illumination_variation_plots(samp,sm_uncorr_mi,sm_mi_corr_mi,sm_basic_corr_mi,central=False,save_dirpath=None) :
    #clip the outer edges off if the plots are for the central region only
    if central :
        yclip = int(sm_uncorr_mi.shape[0]*0.1)
        xclip = int(sm_uncorr_mi.shape[1]*0.1)
        sm_uncorr_mi = sm_uncorr_mi[yclip:-yclip,xclip:-xclip,:]
        sm_mi_corr_mi = sm_mi_corr_mi[yclip:-yclip,xclip:-xclip,:]
    #figure out the number of layers and filter breaks
    nlayers=sm_uncorr_mi.shape[-1]
    last_filter_layers = [lg[1] for lg in list(samp.layer_groups.values())[:-1]] 
    #keep track of the uncorrected and corrected images' minimum and maximum (and 5/95%ile) pixel intensities 
    u_low_pixel_intensities=[]; u_high_pixel_intensities=[]
    mi_c_low_pixel_intensities=[]; mi_c_high_pixel_intensities=[]
    basic_c_low_pixel_intensities=[]; basic_c_high_pixel_intensities=[]
    u_std_devs=[]; mi_c_std_devs=[]; basic_c_std_devs=[]
    plt.figure(figsize=(16.8,(27./64.)*16.8))
    xaxis_vals = list(range(1,nlayers+1))
    #iterate over the layers
    for layer_i in range(nlayers) :
        #find the min, max, and 5/95%ile pixel intensities for this uncorrected and corrected image layer
        sorted_u_layer = np.sort((sm_uncorr_mi[:,:,layer_i]).flatten())/np.mean(sm_uncorr_mi[:,:,layer_i])
        u_low_pixel_intensities.append(sorted_u_layer[int(0.05*len(sorted_u_layer))])
        u_stddev = np.std(sorted_u_layer); u_std_devs.append(u_stddev)
        u_high_pixel_intensities.append(sorted_u_layer[int(0.95*len(sorted_u_layer))])
        sorted_mi_c_layer = np.sort((sm_mi_corr_mi[:,:,layer_i]).flatten())
        sorted_mi_c_layer/=np.mean(sm_mi_corr_mi[:,:,layer_i])
        mi_c_low_pixel_intensities.append(sorted_mi_c_layer[int(0.05*len(sorted_mi_c_layer))])
        mi_c_stddev = np.std(sorted_mi_c_layer); mi_c_std_devs.append(mi_c_stddev)
        mi_c_high_pixel_intensities.append(sorted_mi_c_layer[int(0.95*len(sorted_mi_c_layer))])
        sorted_basic_c_layer = np.sort((sm_basic_corr_mi[:,:,layer_i]).flatten())
        sorted_basic_c_layer/=np.mean(sm_basic_corr_mi[:,:,layer_i])
        basic_c_low_pixel_intensities.append(sorted_basic_c_layer[int(0.05*len(sorted_basic_c_layer))])
        basic_c_stddev = np.std(sorted_basic_c_layer); basic_c_std_devs.append(basic_c_stddev)
        basic_c_high_pixel_intensities.append(sorted_basic_c_layer[int(0.95*len(sorted_basic_c_layer))])
        if layer_i==0 :
            plt.fill([layer_i+0.5,layer_i+0.5,layer_i+1.5,layer_i+1.5],
                     [1.-u_stddev,1.+u_stddev,1.+u_stddev,1.-u_stddev],
                     'mediumseagreen',alpha=0.5,label='uncorrected std. dev.')
            plt.fill([layer_i+0.5,layer_i+0.5,layer_i+1.5,layer_i+1.5],
                     [1.-mi_c_stddev,1.+mi_c_stddev,1.+mi_c_stddev,1.-mi_c_stddev],
                     'goldenrod',alpha=0.5,label='meanimage corrected std. dev.')
            plt.fill([layer_i+0.5,layer_i+0.5,layer_i+1.5,layer_i+1.5],
                     [1.-basic_c_stddev,1.+basic_c_stddev,1.+basic_c_stddev,1.-basic_c_stddev],
                     'darkorchid',alpha=0.5,label='BaSiC corrected std. dev.')
        else :
            plt.fill([layer_i+0.5,layer_i+0.5,layer_i+1.5,layer_i+1.5],
                     [1.-u_stddev,1.+u_stddev,1.+u_stddev,1.-u_stddev],'mediumseagreen',alpha=0.5)
            plt.fill([layer_i+0.5,layer_i+0.5,layer_i+1.5,layer_i+1.5],
                     [1.-mi_c_stddev,1.+mi_c_stddev,1.+mi_c_stddev,1.-mi_c_stddev],'goldenrod',alpha=0.5)
            plt.fill([layer_i+0.5,layer_i+0.5,layer_i+1.5,layer_i+1.5],
                     [1.-basic_c_stddev,1.+basic_c_stddev,1.+basic_c_stddev,1.-basic_c_stddev],'darkorchid',alpha=0.5)
    #plot the relative intensity plots together, with the broadband filter breaks
    plt.plot([xaxis_vals[0],xaxis_vals[-1]],[1.0,1.0],color='darkgreen',linestyle='dashed',label='mean')
    totalmin=min(min(u_low_pixel_intensities),min(mi_c_low_pixel_intensities),min(basic_c_low_pixel_intensities))
    totalmax=max(max(u_high_pixel_intensities),max(mi_c_high_pixel_intensities),max(basic_c_high_pixel_intensities))
    for i in range(len(last_filter_layers)+1) :
        f_i = 0 if i==0 else last_filter_layers[i-1]
        l_i = xaxis_vals[-1] if i==len(last_filter_layers) else last_filter_layers[i]
        if i==0 :
            plt.plot(xaxis_vals[f_i:l_i],u_low_pixel_intensities[f_i:l_i],
                     color='darkred',marker='v',label=r'uncorrected 5th %ile')
            plt.plot(xaxis_vals[f_i:l_i],u_high_pixel_intensities[f_i:l_i],
                     color='darkred',marker='^',label=r'uncorrected 95th %ile')
            plt.plot(xaxis_vals[f_i:l_i],mi_c_low_pixel_intensities[f_i:l_i],
                     color='darkblue',marker='v',label=r'meanimage corrected 5th %ile')
            plt.plot(xaxis_vals[f_i:l_i],mi_c_high_pixel_intensities[f_i:l_i],
                     color='darkblue',marker='^',label=r'meanimage corrected 95th %ile')
            plt.plot(xaxis_vals[f_i:l_i],basic_c_low_pixel_intensities[f_i:l_i],
                     color='darkorange',marker='v',label=r'BaSiC corrected 5th %ile')
            plt.plot(xaxis_vals[f_i:l_i],basic_c_high_pixel_intensities[f_i:l_i],
                     color='darkorange',marker='^',label=r'BaSiC corrected 95th %ile')
            plt.plot([l_i+0.5,l_i+0.5],[totalmin-0.01,totalmax+0.01],
                     color='black',linewidth=2,linestyle='dotted',label='broadband filter changeover')
        else :
            plt.plot(xaxis_vals[f_i:l_i],u_low_pixel_intensities[f_i:l_i],color='darkred',marker='v')
            plt.plot(xaxis_vals[f_i:l_i],u_high_pixel_intensities[f_i:l_i],color='darkred',marker='^')
            plt.plot(xaxis_vals[f_i:l_i],mi_c_low_pixel_intensities[f_i:l_i],color='darkblue',marker='v')
            plt.plot(xaxis_vals[f_i:l_i],mi_c_high_pixel_intensities[f_i:l_i],color='darkblue',marker='^')
            plt.plot(xaxis_vals[f_i:l_i],basic_c_low_pixel_intensities[f_i:l_i],color='darkorange',marker='v')
            plt.plot(xaxis_vals[f_i:l_i],basic_c_high_pixel_intensities[f_i:l_i],color='darkorange',marker='^')
            if i!=len(last_filter_layers) :
                plt.plot([l_i+0.5,l_i+0.5],[totalmin-0.01,totalmax+0.01],color='black',linewidth=2,linestyle='dotted')
    plt.title('uncorrected/corrected smoothed mean image relative pixel intensities',fontsize=14)
    plt.xlabel('layer number',fontsize=14)
    #fix the range on the x-axis to accommodate the legend
    plt.xlim(0,nlayers+(10 if nlayers==35 else 12))
    plt.ylabel('pixel intensity relative to layer mean',fontsize=14)
    plt.legend(loc='lower right')
    fn = 'smoothed_mean_image_pixel_intensities'
    if central :
        fn+='_central'
    fn+='.png'
    save_figure_in_dir(plt,fn,save_dirpath)

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