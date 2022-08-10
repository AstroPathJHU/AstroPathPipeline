#imports
import pathlib, datetime, pybasic
import numpy as np, matplotlib.pyplot as plt
from argparse import ArgumentParser
from threading import Thread
from queue import Queue
from astropath.utilities import units
from astropath.utilities.miscplotting import save_figure_in_dir
from astropath.utilities.tableio import readtable, writetable
from astropath.utilities.img_file_io import get_raw_as_hwl, get_raw_as_hw, write_image_to_file
from astropath.hpfs.imagecorrection.utilities import CorrectionModelTableEntry
from astropath.hpfs.flatfield.meanimagesample import MeanImageSample
from astropath.hpfs.warping.warpingsample import WarpingSample
from utilities import timestamp, AlignmentOverlapComparison, add_rect_image_to_queue
from utilities import get_pre_and_post_correction_rect_layer_images_by_index, add_overlap_comparison_to_queue
from plotting import illumination_variation_plots, overlap_mse_reduction_plots
from plotting import overlap_mse_reduction_comparison_plot, overlap_mse_reduction_comparison_box_plot
from plotting import overlap_correction_score_difference_box_plot, overlap_correction_score_plots, ks_test_plot

#fast units setup
units.setup('fast')

#constants
APPROC = pathlib.Path('//bki04/astropath_processing')
CORRECTION_MODEL_FILE = APPROC/'AstroPathCorrectionModels.csv'
FLATFIELD_DIR = APPROC/'flatfield'

#helper functions

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
    parser.add_argument('--no_different_exposure_times',action='store_true',
                        help='include this flag to only plot overlaps where both rectangles had the same exposure time')
    parser.add_argument('--max_shift_diff',type=float,default=-900,
                        help='''Overlaps with >= this amount of difference (pixels) in x/y alignment between meanimage 
                                and BaSiC corrections will be trimmed before plotting or calculating statistics''')
    parser.add_argument('--max_mse_diff',type=float,default=-900,
                        help='''Overlaps with >= this amount of relative MSE difference pre or post-correction with 
                                either method will be trimmed before plotting or calculating statistics''')
    parser.add_argument('--skip_rerun_check',action='store_true',
                        help='''Add this flag to skip creating a bunch of WarpSamples to make sure all overlap 
                                comparisons have been run''')
    args = parser.parse_args()
    if not args.workingdir.is_dir() :
        args.workingdir.mkdir(parents=True)
    return args

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
    layer_dir = save_dirpath/'flatfield_and_darkfield_layers'
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

def get_overlap_comparisons(uncorr_samp,args,mi_ff_fp,basic_ff,basic_df,max_shift_diff,max_mse_diff,save_dirpath,n_threads,skip_rerun_check) :
    overlap_comparison_fp = save_dirpath/'overlap_comparisons.csv'
    if overlap_comparison_fp.is_file() :
        overlap_comparisons_found = readtable(overlap_comparison_fp,AlignmentOverlapComparison)
        msg = f'{timestamp()}   Found {len(overlap_comparisons_found)} existing overlap comparisons for '
        msg+= f'{uncorr_samp.SlideID} in {overlap_comparison_fp}'
        print(msg)
    else :
        overlap_comparisons_found = []
    overlap_comparisons = {}
    #do one layer at a time
    for li in range(uncorr_samp.nlayersim3) :
        if not skip_rerun_check :
            warp_samp = WarpingSample(args.root,samp=args.slideID,shardedim3root=args.rawfile_root,
                                      et_offset_file=None,
                                      skip_et_corrections=False,
                                      flatfield_file=mi_ff_fp,warping_file=None,correction_model_file=None,
                                      filetype='raw',layer=li+1,
                                      )
        overlap_comparisons[li+1] = [oc for oc in overlap_comparisons_found if oc.layer_n==li+1]
        comp_ns_found = [oc.n for oc in overlap_comparisons[li+1]]
        p1_p2_pairs_done = set([(oc.p1,oc.p2) for oc in overlap_comparisons[li+1]])
        needed_rect_ns = set()
        if not skip_rerun_check :
            for overlap in warp_samp.overlaps :
                if ( (overlap.n not in comp_ns_found) and 
                     ((overlap.p1,overlap.p2) not in p1_p2_pairs_done) and 
                     ((overlap.p2,overlap.p1) not in p1_p2_pairs_done) ):
                    needed_rect_ns.add(overlap.p1)
                    needed_rect_ns.add(overlap.p2)
        if len(needed_rect_ns)>0 :
            msg=f'{timestamp()}   Getting {len(needed_rect_ns)} uncorrected/corrected rectangle images for layer '
            msg+=f'{li+1} of {uncorr_samp.SlideID}'
            print(msg)
            layer_images_by_rect_n = get_pre_and_post_correction_rect_layer_images_by_index(uncorr_samp,warp_samp,
                                                                                            basic_ff,basic_df,
                                                                                            li,needed_rect_ns,n_threads)
            print(f'{timestamp()}   Getting overlap comparisons for layer {li+1} of {uncorr_samp.SlideID}')
            threads = []
            overlap_comp_queue = Queue()
            last_writeout=datetime.datetime.now()
            new_comps = []
            for io,overlap in enumerate(warp_samp.overlaps) :
                if ( (overlap.n in comp_ns_found) or 
                     ((overlap.p1,overlap.p2) in p1_p2_pairs_done) or 
                     ((overlap.p2,overlap.p1) in p1_p2_pairs_done) ):
                    continue
                if io%50==0 :
                    print(f'{timestamp()}       getting comparisons for overlap {io}(/{len(warp_samp.overlaps)})')
                while len(threads)>=n_threads :
                    thread = threads.pop(0)
                    thread.join(1)
                    if thread.is_alive() :
                        threads.append(thread)
                while not overlap_comp_queue.empty() :
                    new_comp = overlap_comp_queue.get()
                    new_comps.append(new_comp)
                    overlap_comparisons[li+1].append(new_comp)
                if len(new_comps)>0 and (datetime.datetime.now()-last_writeout).total_seconds()>=10 :
                    writetable(overlap_comparison_fp,new_comps,append=True)
                    new_comps = []
                    last_writeout = datetime.datetime.now()
                new_thread = Thread(target=add_overlap_comparison_to_queue,
                                    args=(overlap,warp_samp.rectangles,layer_images_by_rect_n,li,overlap_comp_queue))
                new_thread.start()
                threads.append(new_thread)
                p1_p2_pairs_done.add((overlap.p1,overlap.p2))
            for thread in threads :
                thread.join()
            overlap_comp_queue.put(None)
            olap_comp = overlap_comp_queue.get()
            while olap_comp is not None :
                new_comps.append(olap_comp)
                overlap_comparisons[li+1].append(olap_comp)
                olap_comp = overlap_comp_queue.get()
            if len(new_comps)>0 :
                writetable(overlap_comparison_fp,new_comps,append=True)
        overlap_comparisons[li+1] = [oc for oc in overlap_comparisons[li+1] if oc.error_code==0]
        comps = overlap_comparisons[li+1]
        if max_shift_diff!=-900. :
            comps = [oc for oc in comps if abs(oc.basic_dx-oc.meanimage_dx)<max_shift_diff and abs(oc.basic_dy-oc.meanimage_dy)<max_shift_diff]
        if max_mse_diff!=-900. :
            comps = [oc for oc in comps if oc.orig_mse_diff/oc.orig_mse1<max_mse_diff and oc.basic_mse_diff/oc.basic_mse1<max_mse_diff and oc.meanimage_mse_diff/oc.meanimage_mse1<max_mse_diff]
        if max_shift_diff!=-900. or max_mse_diff!=-900. :
            n_trimmed = len(overlap_comparisons[li+1]) - len(comps)
            pct_trimmed = 100.*(1.-((1.*len(comps))/(1.*len(overlap_comparisons[li+1]))))
            msg=f'{timestamp()} Trimmed {n_trimmed} overlaps ({pct_trimmed:.4f}%) in layer {li+1} that were poorly '
            msg+= 'aligned or had large differences in alignment shifts'
            print(msg)
        overlap_comparisons[li+1] = comps
    return overlap_comparisons

#main function

def main() :
    #create the argument parser
    args = get_arguments()
    #figure out where the meanimage-based flatfield for this sample is
    correction_model_entries = readtable(CORRECTION_MODEL_FILE,CorrectionModelTableEntry)
    meanimage_ff_name = [te.FlatfieldVersion for te in correction_model_entries if te.SlideID==args.slideID]
    meanimage_ff_fp = FLATFIELD_DIR/f'flatfield_{meanimage_ff_name[0]}.bin'
    #create the mean image sample
    print(f'{timestamp()} creating MeanImageSample for {args.slideID}')
    uncorrected_sample = MeanImageSample(args.root,samp=args.slideID,shardedim3root=args.rawfile_root,
                                       et_offset_file=None,
                                       #don't apply ANY corrections before running BaSiC
                                       skip_et_corrections=True, 
                                       flatfield_file=None,warping_file=None,correction_model_file=None,
                                       filetype='raw',
                                       )
    dims = (uncorrected_sample.fheight,uncorrected_sample.fwidth,uncorrected_sample.nlayersim3)
    print(f'{timestamp()} done creating MeanImageSample for {args.slideID}')
    #create and save the basic flatfield
    print(f'{timestamp()} running BaSiC for {args.slideID}')
    basic_flatfield, basic_darkfield = run_basic(uncorrected_sample,args.workingdir,args.n_threads,args.no_darkfield)
    print(f'{timestamp()} done running BaSiC for {args.slideID}')
    #create the illumination variation plots
    print(f'{timestamp()} getting meanimage flatfield and pre/post-correction meanimages for {args.slideID}')
    meanimage_fp = args.root/args.slideID/'im3'/'meanimage'/f'{args.slideID}-mean_image.bin'
    meanimage = get_raw_as_hwl(meanimage_fp,*dims,np.float64)
    meanimage_ff = get_raw_as_hwl(meanimage_ff_fp,*dims,np.float64)
    mi_corrected_meanimage = meanimage/meanimage_ff
    basic_corrected_meanimage = meanimage/basic_flatfield
    print(f'{timestamp()} making meanimage illumination variation plots for {args.slideID}')
    illumination_variation_plots(uncorrected_sample,
                                 meanimage,
                                 mi_corrected_meanimage,
                                 basic_corrected_meanimage,
                                 central=False,
                                 save_dirpath=args.workingdir)
    #create the warping sample
    print(f'{timestamp()} getting overlap comparisons for {args.slideID}')
    overlap_comparisons = get_overlap_comparisons(uncorrected_sample,
                                                  args,
                                                  meanimage_ff_fp,
                                                  basic_flatfield,
                                                  basic_darkfield,
                                                  args.max_shift_diff,args.max_mse_diff,
                                                  args.workingdir,
                                                  args.n_threads,
                                                  args.skip_rerun_check)
    #create the overlap MSE reduction/correction score comparison plots
    print(f'{timestamp()} creating correction score and overlap mse reduction plots for {args.slideID}')
    overlap_mse_reduction_plots(overlap_comparisons,args.workingdir,args.no_different_exposure_times)
    overlap_correction_score_plots(overlap_comparisons,args.workingdir,args.no_different_exposure_times)
    #create the comparison plots over all layers
    print(f'{timestamp()} creating final summary plots for {args.slideID}')
    overlap_mse_reduction_comparison_plot(uncorrected_sample,overlap_comparisons,args.workingdir,args.no_different_exposure_times)
    overlap_mse_reduction_comparison_box_plot(uncorrected_sample,overlap_comparisons,args.workingdir,args.no_different_exposure_times)
    overlap_correction_score_difference_box_plot(uncorrected_sample,overlap_comparisons,args.workingdir,args.no_different_exposure_times)
    #ks_test_plot(uncorrected_sample,overlap_comparisons,args.workingdir)
    print(f'{timestamp()} Done')

if __name__=='__main__' :
    main()
