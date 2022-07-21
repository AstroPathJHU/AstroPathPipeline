#imports
import pathlib, datetime, pybasic, copy
import numpy as np, matplotlib.pyplot as plt, matplotlib as mpl
from argparse import ArgumentParser
from threading import Thread
from queue import Queue
from dataclasses import dataclass
from astropath.utilities import units
from astropath.utilities.miscplotting import save_figure_in_dir
from astropath.utilities.tableio import readtable, writetable
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
    layer_n : int
    p1 : int
    p2 : int
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

def add_rect_image_and_index_to_queue(rect,queue,layer_i) :
    with rect.using_corrected_im3() as im :
        im_layer = im[:,:,layer_i]
        queue.put((rect.n,im_layer))

def add_basic_corr_rect_image_and_index_to_queue(rect,basic_ff,basic_df,queue,layer_i) :
    with rect.using_corrected_im3() as im :
        im_layer = im[:,:,layer_i]
    corr_im_layer = pybasic.correct_illumination([im_layer],basic_ff,basic_df)
    queue.put((rect.n,corr_im_layer))

def add_overlap_comparison_to_queue(overlap,rects,rect_images_by_n,li,queue) :
    p1_rect = [r for r in rects if r.n==overlap.p1]
    assert len(p1_rect)==1
    p1_rect = p1_rect[0]
    p2_rect = [r for r in rects if r.n==overlap.p2]
    assert len(p2_rect)==1
    p2_rect = p2_rect[0]
    p1_rect.image = rect_images_by_n[p1_rect.n]['uncorr']
    p2_rect.image = rect_images_by_n[p2_rect.n]['uncorr']
    overlap.myupdaterectangles([p1_rect,p2_rect])
    orig_result = copy.deepcopy(overlap.align(alreadyalignedstrategy='overwrite'))
    if orig_result.exit!=0 :
        return
    #overlap.showimages(normalize=1000.)
    orig_dx = orig_result.dxvec[0].nominal_value
    orig_dy = orig_result.dxvec[1].nominal_value
    p1_rect.image = rect_images_by_n[p1_rect.n]['basic_c']
    p2_rect.image = rect_images_by_n[p2_rect.n]['basic_c']
    overlap.myupdaterectangles([p1_rect,p2_rect])
    basic_result = copy.deepcopy(overlap.align(alreadyalignedstrategy='overwrite'))
    if basic_result.exit!=0 :
        return
    #overlap.showimages(normalize=1000.)
    basic_dx = basic_result.dxvec[0].nominal_value
    basic_dy = basic_result.dxvec[1].nominal_value
    p1_rect.image = rect_images_by_n[p1_rect.n]['mi_c']
    p2_rect.image = rect_images_by_n[p2_rect.n]['mi_c']
    overlap.myupdaterectangles([p1_rect,p2_rect])
    meanimage_result = copy.deepcopy(overlap.align(alreadyalignedstrategy='overwrite'))
    if meanimage_result.exit!=0 :
        return
    #overlap.showimages(normalize=1000.)
    meanimage_dx = meanimage_result.dxvec[0].nominal_value
    meanimage_dy = meanimage_result.dxvec[1].nominal_value
    queue.put(
        AlignmentOverlapComparison(overlap.n,li+1,overlap.p1,overlap.p2,overlap.tag,overlap.overlap_npix,
                                    orig_dx,orig_dy,orig_result.mse[0],orig_result.mse[2],
                                    basic_dx,basic_dy,basic_result.mse[0],basic_result.mse[2],
                                    meanimage_dx,meanimage_dy,meanimage_result.mse[0],meanimage_result.mse[2])
        )

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

def get_pre_and_post_correction_rect_layer_images_by_index(mi_samp,warp_samp,basic_ff,basic_df,li,needed_ns,n_threads) :
    layer_images_by_rect_n = {}
    threads = []
    uncorr_queue = Queue(); mi_c_queue = Queue(); basic_c_queue = Queue()
    for ir,(mi_r,w_r) in enumerate(zip(mi_samp.rectangles,warp_samp.rectangles)) :
        if mi_r.n not in needed_ns :
            continue
        if ir%50==0 :
            print(f'{timestamp()}       getting images for rectangle {ir}(/{len(mi_samp.rectangles)})')
        while len(threads)>=(n_threads-3) :
            thread = threads.pop(0)
            thread.join()
        uncorr_thread = Thread(target=add_rect_image_and_index_to_queue,args=(mi_r,uncorr_queue,li))
        uncorr_thread.start()
        threads.append(uncorr_thread)
        mi_c_thread = Thread(target=add_rect_image_and_index_to_queue,args=(w_r,mi_c_queue,li))
        mi_c_thread.start()
        threads.append(mi_c_thread)
        basic_c_thread = Thread(target=add_basic_corr_rect_image_and_index_to_queue,
                                args=(mi_r,basic_ff,basic_df,basic_c_queue,li))
        basic_c_thread.start()
        threads.append(basic_c_thread)
    for thread in threads :
        thread.join()
    uncorr_queue.put((None,None)); mi_c_queue.put((None,None)); basic_c_queue.put((None,None));
    rn, uncorr_im_layer = uncorr_queue.get()
    while rn is not None and uncorr_im_layer is not None :
        if rn not in layer_images_by_rect_n.keys() :
            layer_images_by_rect_n[rn] = {}
        layer_images_by_rect_n[rn]['uncorr'] = uncorr_im_layer
        rn, uncorr_im_layer = uncorr_queue.get()
    rn, mi_c_im_layer = mi_c_queue.get()
    while rn is not None and mi_c_im_layer is not None :
        layer_images_by_rect_n[rn]['mi_c'] = mi_c_im_layer
        rn, mi_c_im_layer = mi_c_queue.get()
    rn, basic_c_im_layer = basic_c_queue.get()
    while rn is not None and basic_c_im_layer is not None :
        layer_images_by_rect_n[rn]['basic_c'] = basic_c_im_layer
        rn, basic_c_im_layer = basic_c_queue.get()
    return layer_images_by_rect_n

def get_overlap_comparisons(mi_samp,warp_samp,basic_ff,basic_df,save_dirpath,n_threads) :
    overlap_comparison_fp = save_dirpath/'overlap_comparisons.csv'
    if overlap_comparison_fp.is_file() :
        overlap_comparisons_found = readtable(overlap_comparison_fp,AlignmentOverlapComparison)
        msg = f'{timestamp()}   Found {len(overlap_comparisons_found)} existing overlap comparisons for '
        msg+= f'{mi_samp.SlideID} in {overlap_comparison_fp}'
        print(msg)
    else :
        overlap_comparisons_found = []
    overlap_comparisons = {}
    #do one layer at a time
    for li in range(mi_samp.nlayersim3) :
        overlap_comparisons[li+1] = [oc for oc in overlap_comparisons_found if oc.layer_n==li+1]
        comp_ns_found = [oc.n for oc in overlap_comparisons[li+1]]
        p1_p2_pairs_done = set([(oc.p1,oc.p2) for oc in overlap_comparisons[li+1]])
        needed_rect_ns = set()
        for overlap in warp_samp.overlaps :
            if ( (overlap.n not in comp_ns_found) and 
                 ((overlap.p1,overlap.p2) not in p1_p2_pairs_done) and 
                 ((overlap.p2,overlap.p1) not in p1_p2_pairs_done) ):
                needed_rect_ns.add(overlap.p1)
                needed_rect_ns.add(overlap.p2)
        if len(needed_rect_ns)>0 :
            msg=f'{timestamp()}   Getting {len(needed_rect_ns)} uncorrected/corrected rectangle images for layer '
            msg+=f'{li+1} of {mi_samp.SlideID}'
            print(msg)
            layer_images_by_rect_n = get_pre_and_post_correction_rect_layer_images_by_index(mi_samp,warp_samp,
                                                                                            basic_ff,basic_df,
                                                                                            li,needed_rect_ns,n_threads)
            print(f'{timestamp()}   Getting overlap comparisons for layer {li+1} of {mi_samp.SlideID}')
            threads = []
            overlap_comp_queue = Queue()
            for io,overlap in enumerate(warp_samp.overlaps) :
                if ( (overlap.n in comp_ns_found) or 
                     ((overlap.p1,overlap.p2) in p1_p2_pairs_done) or 
                     ((overlap.p2,overlap.p1) in p1_p2_pairs_done) ):
                    continue
                if io%50==0 :
                    print(f'{timestamp()}       getting comparisons for overlap {io}(/{len(warp_samp.overlaps)})')
                while len(threads)>=n_threads :
                    thread = threads.pop(0)
                    thread.join()
                while not overlap_comp_queue.empty() :
                    new_comp = overlap_comp_queue.get()
                    writetable(overlap_comparison_fp,[new_comp],append=True)
                    overlap_comparisons[li+1].append(new_comp)
                new_thread = Thread(target=add_overlap_comparison_to_queue,
                                    args=(overlap,warp_samp.rectangles,layer_images_by_rect_n,overlap_comp_queue))
                new_thread.start()
                threads.append(new_thread)
            for thread in threads :
                thread.join()
            overlap_comp_queue.append(None)
            olap_comp = overlap_comp_queue.get()
            while olap_comp is not None :
                overlap_comparisons[li+1].append(olap_comp)
                olap_comp = overlap_comp_queue.get()
    return overlap_comparisons

def overlap_mse_reduction_plots(overlap_comparisons_by_layer_n,save_dirpath) :
    layer_dir = save_dirpath/'mse_reduction_layer_plots'
    if not layer_dir.is_dir() :
        layer_dir.mkdir(parents=True)
    for layer_n in overlap_comparisons_by_layer_n.keys() :
        overlap_comparisons = overlap_comparisons_by_layer_n[layer_n]
        overlap_comparisons = [oc for oc in overlap_comparisons if abs(oc.basic_dx-oc.meanimage_dx)<0.01 and abs(oc.basic_dy-oc.meanimage_dy)<0.01]
        overlap_comparisons = [oc for oc in overlap_comparisons if oc.orig_mse_diff/oc.orig_mse1<0.03 and oc.basic_mse_diff/oc.basic_mse1<0.03 and oc.meanimage_mse_diff/oc.meanimage_mse1<0.03]
        pct_trimmed = 100.*(1.-((1.*len(overlap_comparisons))/1.*len(overlap_comparisons_by_layer_n[layer_n])))
        msg=f'{timestamp()} Trimmed {pct_trimmed:.4f}% of overlaps in layer {layer_n} that were poorly aligned or '
        msg+= 'had large differences in alignment shifts'
        print(msg)
        f,ax = plt.subplots(4,4,figsize=(4*6.4,4*4.6))
        sum_weights = np.sum(np.array([oc.npix for oc in overlap_comparisons]))
        orig_rel_residuals = [oc.orig_mse_diff/oc.orig_mse1 for oc in overlap_comparisons]
        stddev_orig_rel_residuals = np.std(np.array(orig_rel_residuals))
        w_mean_orig_rel_residuals = np.sum(np.array([oc.npix*(oc.orig_mse_diff/oc.orig_mse1) for oc in overlap_comparisons]))/sum_weights
        meanimage_rel_residuals = [oc.meanimage_mse_diff/oc.meanimage_mse1 for oc in overlap_comparisons]
        stddev_meanimage_rel_residuals = np.std(np.array(meanimage_rel_residuals))
        w_mean_meanimage_rel_residuals = np.sum(np.array([oc.npix*(oc.meanimage_mse_diff/oc.meanimage_mse1) for oc in overlap_comparisons]))/sum_weights
        basic_rel_residuals = [oc.basic_mse_diff/oc.basic_mse1 for oc in overlap_comparisons]
        stddev_basic_rel_residuals = np.std(np.array(basic_rel_residuals))
        w_mean_basic_rel_residuals = np.sum(np.array([oc.npix*(oc.basic_mse_diff/oc.basic_mse1) for oc in overlap_comparisons]))/sum_weights
        alignment_x_diffs = [oc.basic_dx-oc.meanimage_dx for oc in overlap_comparisons]
        alignment_y_diffs = [oc.basic_dy-oc.meanimage_dy for oc in overlap_comparisons]
        ax[0][0].hist(orig_rel_residuals,bins=80)
        ax[0][0].set_title('uncorrected relative residuals')
        ax[0][0].axvline(w_mean_orig_rel_residuals,label=f'w. mean = {w_mean_orig_rel_residuals:.4f}+/-{stddev_orig_rel_residuals:.4f}',color='r',linewidth=2)
        ax[0][0].legend()
        ax[0][1].hist(meanimage_rel_residuals,bins=80)
        ax[0][1].set_title('meanimage relative residuals')
        ax[0][1].axvline(w_mean_meanimage_rel_residuals,label=f'w. mean = {w_mean_meanimage_rel_residuals:.4f}+/-{stddev_meanimage_rel_residuals:.4f}',color='r',linewidth=2)
        ax[0][1].legend()
        ax[0][2].hist(basic_rel_residuals,bins=80)
        ax[0][2].set_title('BaSiC relative residuals')
        ax[0][2].axvline(w_mean_basic_rel_residuals,label=f'w. mean = {w_mean_basic_rel_residuals:.4f}+/-{stddev_basic_rel_residuals:.4f}',color='r',linewidth=2)
        ax[0][2].legend()
        ax[0][3].hist2d(alignment_x_diffs,alignment_y_diffs,bins=60,norm=mpl.colors.LogNorm())
        ax[0][3].set_title('alignment x/y differences')
        for ti,tag in enumerate([1,2,3,4]) :
            tag_comparisons = [oc for oc in overlap_comparisons if oc.tag==tag]
            weights = [oc.npix for oc in tag_comparisons]
            sum_weights = np.sum(np.array(weights))
            rel_residual_diffs = [(oc.basic_mse_diff/oc.basic_mse1)-(oc.meanimage_mse_diff/oc.meanimage_mse1) for oc in tag_comparisons]
            stddev_rel_residual_diffs = np.std(np.array(rel_residual_diffs))
            w_mean_rel_residual_diffs = np.sum(np.array([weight*rel_resid_diff for weight,rel_resid_diff in zip(weights,rel_residual_diffs)]))/sum_weights
            meanimage_rel_residual_reduxes = [1.-(oc.meanimage_mse_diff/oc.meanimage_mse1)/(oc.orig_mse_diff/oc.orig_mse1) for oc in tag_comparisons]
            basic_rel_residual_reduxes = [1.-(oc.basic_mse_diff/oc.basic_mse1)/(oc.orig_mse_diff/oc.orig_mse1) for oc in tag_comparisons]
            rel_residual_redux_diffs = [brrr-mirrr for brrr,mirrr in zip(basic_rel_residual_reduxes,meanimage_rel_residual_reduxes)]
            stddev_rel_residual_redux_diffs = np.std(np.array(rel_residual_redux_diffs))
            w_mean_rel_residual_redux_diffs = np.sum(np.array([weight*rel_resid_redux_diff for weight,rel_resid_redux_diff in zip(weights,rel_residual_redux_diffs)]))/sum_weights
            ax[1][ti].hist(rel_residual_diffs,bins=80)
            ax[1][ti].set_title(f'BaSiC-meanimage rel. residuals, tag={tag}')
            ax[1][ti].axvline(w_mean_rel_residual_diffs,label=f'w. mean = {w_mean_rel_residual_diffs:.4f}+/-{stddev_rel_residual_diffs:.4f}',color='r',linewidth=2)
            ax[1][ti].legend()
            ax[2][ti].hist2d(meanimage_rel_residual_reduxes,basic_rel_residual_reduxes,bins=60,norm=mpl.colors.LogNorm())
            ax[2][ti].set_title(f'BaSiC vs. meanimage rel. residual reductions, tag={tag}')
            ax[2][ti].set_xlabel('meanimage rel. residual reductions')
            ax[2][ti].set_ylabel('BaSiC rel. residual reductions')
            ax[2][ti].axvline(0.,color='k',linewidth=2)
            ax[2][ti].axhline(0.,color='k',linewidth=2)
            ax[3][ti].hist(rel_residual_redux_diffs,bins=80)
            ax[3][ti].set_title(f'BaSiC-meanimage rel. residual reductions, tag={tag}')
            ax[3][ti].axvline(w_mean_rel_residual_redux_diffs,label=f'w. mean = {w_mean_rel_residual_redux_diffs:.4f}+/-{stddev_rel_residual_redux_diffs:.4f}',color='r',linewidth=2)
            ax[3][ti].legend()
        plt.tight_layout()
        save_figure_in_dir(plt,f'overlap_mse_reduction_comparisons_layer_{layer_n}.png',layer_dir)
        plt.close()

def overlap_mse_reduction_comparison_plot(samp,overlap_comparisons_by_layer_n,save_dirpath) :
    rel_mse_redux_diff_means = []
    rel_mse_redux_diff_stds = []
    for layer_n in overlap_comparisons_by_layer_n.keys() :
        rel_mse_redux_diff_means.append([])
        rel_mse_redux_diff_stds.append([])
        overlap_comparisons = overlap_comparisons_by_layer_n[layer_n]
        overlap_comparisons = [oc for oc in overlap_comparisons if abs(oc.basic_dx-oc.meanimage_dx)<0.01 and abs(oc.basic_dy-oc.meanimage_dy)<0.01]
        overlap_comparisons = [oc for oc in overlap_comparisons if oc.orig_mse_diff/oc.orig_mse1<0.03 and oc.basic_mse_diff/oc.basic_mse1<0.03 and oc.meanimage_mse_diff/oc.meanimage_mse1<0.03]
        for tag in (1,2,3,4) :
            tag_comparisons = [oc for oc in overlap_comparisons if oc.tag==tag]
            weights = [oc.npix for oc in tag_comparisons]
            sum_weights = np.sum(np.array(weights))
            meanimage_rel_mse_reduxes = [1.-(oc.meanimage_mse_diff/oc.meanimage_mse1)/(oc.orig_mse_diff/oc.orig_mse1) for oc in tag_comparisons]
            basic_rel_mse_reduxes = [1.-(oc.basic_mse_diff/oc.basic_mse1)/(oc.orig_mse_diff/oc.orig_mse1) for oc in tag_comparisons]
            rel_mse_redux_diffs = [brrr-mirrr for brrr,mirrr in zip(basic_rel_mse_reduxes,meanimage_rel_mse_reduxes)]
            stddev_rel_mse_redux_diffs = np.std(np.array(rel_mse_redux_diffs))
            w_mean_rel_mse_redux_diffs = np.sum(np.array([weight*rel_resid_redux_diff for weight,rel_resid_redux_diff in zip(weights,rel_mse_redux_diffs)]))/sum_weights
            rel_mse_redux_diff_means[-1].append(w_mean_rel_mse_redux_diffs)
            rel_mse_redux_diff_stds[-1].append(stddev_rel_mse_redux_diffs)
    rel_mse_redux_diff_means = (np.array(rel_mse_redux_diff_means)).T
    rel_mse_redux_diff_stds = (np.array(rel_mse_redux_diff_stds)).T
    xaxis_vals = list(range(1,len(overlap_comparisons_by_layer_n.keys())))
    f,ax = plt.subplots(figsize=(10.,4.6))
    ax.axhline(0.0,color='gray',linestyle='dotted')
    last_filter_layers = [lg[1] for lg in list(samp.layer_groups.values())[:-1]] 
    for i in range(len(last_filter_layers)+1) :
        f_i = 0 if i==0 else last_filter_layers[i-1]
        l_i = xaxis_vals[-1] if i==len(last_filter_layers) else last_filter_layers[i]
        if i==0 :
            ax.errorbar(xaxis_vals[f_i:l_i],rel_mse_redux_diff_means[0,f_i:l_i],
                        yerr=rel_mse_redux_diff_stds[0,f_i:l_i],
                        color='darkblue',marker='^',alpha=0.85,label='tag 1')
            ax.errorbar(xaxis_vals[f_i:l_i]+0.1,rel_mse_redux_diff_means[1,f_i:l_i],
                        yerr=rel_mse_redux_diff_stds[1,f_i:l_i],
                        color='darkorange',marker='v',alpha=0.85,label='tag 2')
            ax.errorbar(xaxis_vals[f_i:l_i]+0.2,rel_mse_redux_diff_means[2,f_i:l_i],
                        yerr=rel_mse_redux_diff_stds[2,f_i:l_i],
                        color='darkgreen',marker='<',alpha=0.85,label='tag 3')
            ax.errorbar(xaxis_vals[f_i:l_i]+0.3,rel_mse_redux_diff_means[3,f_i:l_i],
                        yerr=rel_mse_redux_diff_stds[3,f_i:l_i],
                        color='darkmagenta',marker='>',alpha=0.85,label='tag 4')
            ax.axvline(l_i+0.5,color='black',linewidth=2,linestyle='dotted',label='broadband filter changeover')
        else :
            ax.errorbar(xaxis_vals[f_i:l_i],rel_mse_redux_diff_means[0,f_i:l_i],
                        yerr=rel_mse_redux_diff_stds[0,f_i:l_i],
                        color='darkblue',marker='^',alpha=0.85)
            ax.errorbar(xaxis_vals[f_i:l_i]+0.1,rel_mse_redux_diff_means[1,f_i:l_i],
                        yerr=rel_mse_redux_diff_stds[1,f_i:l_i],
                        color='darkorange',marker='v',alpha=0.85)
            ax.errorbar(xaxis_vals[f_i:l_i]+0.2,rel_mse_redux_diff_means[2,f_i:l_i],
                        yerr=rel_mse_redux_diff_stds[2,f_i:l_i],
                        color='darkgreen',marker='<',alpha=0.85)
            ax.errorbar(xaxis_vals[f_i:l_i]+0.3,rel_mse_redux_diff_means[3,f_i:l_i],
                        yerr=rel_mse_redux_diff_stds[3,f_i:l_i],
                        color='darkmagenta',marker='>',alpha=0.85)
            if i!=len(last_filter_layers) :
                ax.axvline(l_i+0.5,color='black',linewidth=2,linestyle='dotted')
    ax.set_xticks(xaxis_vals)
    ax.set_xticklabels(xaxis_vals)
    ax.set_xlabel('image layer')
    ax.set_ylabel('BaSiC-meanimage relative overlap MSE reductions')
    ax.legend(loc='best')
    save_figure_in_dir(plt,'mse_reduction_differences.png',save_dirpath)
    plt.close()

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
    overlap_comparisons = get_overlap_comparisons(meanimage_sample,
                                                  warping_sample,
                                                  basic_flatfield,
                                                  basic_darkfield,
                                                  args.workingdir,
                                                  args.n_threads)
    #create the overlap MSE reduction comparison plots
    print(f'{timestamp()} creating overlap mse reduction plots for {args.slideID}')
    overlap_mse_reduction_plots(overlap_comparisons,args.workingdir)
    #create the single comparison plot over all layers
    print(f'{timestamp()} creating final summary plot for {args.slideID}')
    overlap_mse_reduction_comparison_plot(meanimage_sample,overlap_comparisons,args.workingdir)
    print(f'{timestamp()} Done')

if __name__=='__main__' :
    main()