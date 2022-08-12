#imports
import copy, datetime, pybasic
from threading import Thread
from queue import Queue
from astropath.utilities.dataclasses import MyDataClass

#alignment overlap comparison dataclass
class AlignmentOverlapComparison(MyDataClass) :
    n : int
    layer_n : int
    p1 : int
    p2 : int
    exp_time_1 : float
    exp_time_2 : float
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
    error_code : int

def timestamp() :
    return f'[{datetime.datetime.now().strftime("%m/%d/%Y %H:%M:%S")}]'

def add_rect_image_to_queue(rect,queue,layer_i) :
    with rect.using_corrected_im3() as im :
        im_layer = im[:,:,layer_i]
    queue.put(im_layer.copy())

def add_rect_image_and_index_to_queue(rect,queue,layer_i=None) :
    with rect.using_corrected_im3() as im :
        im_layer = im[:,:,layer_i] if layer_i is not None else im
    queue.put((rect.n,im_layer.copy()))

def add_mi_corr_rect_image_and_index_to_queue(rect,meanimage_ff,queue,layer_i) :
    with rect.using_corrected_im3() as im :
        im_layer = im[:,:,layer_i]
        im_layer = im_layer/meanimage_ff[:,:,layer_i]
    queue.put((rect.n,im_layer.copy()))

def add_basic_corr_rect_image_and_index_to_queue(rect,basic_ff,basic_df,queue,layer_i) :
    with rect.using_corrected_im3() as im :
        im_layer = im[:,:,layer_i]
    corr_im_layer = pybasic.correct_illumination([im_layer],basic_ff[:,:,layer_i],basic_df[:,:,layer_i])
    corr_im_layer = corr_im_layer[0]
    queue.put((rect.n,corr_im_layer.copy()))

def get_pre_and_post_correction_rect_layer_images_by_index(uncorr_samp,warp_samp,basic_ff,basic_df,li,needed_ns,n_threads) :
    layer_images_by_rect_n = {}
    threads = []
    uncorr_queue = Queue(); mi_c_queue = Queue(); basic_c_queue = Queue()
    for ir,(uc_r,w_r) in enumerate(zip(uncorr_samp.rectangles,warp_samp.rectangles)) :
        if uc_r.n not in needed_ns :
            continue
        if ir%50==0 :
            print(f'{timestamp()}       getting images for rectangle {ir}(/{len(needed_ns)})')
        while len(threads)>=(n_threads-3) :
            thread = threads.pop(0)
            thread.join(1)
            if thread.is_alive() :
                threads.append(thread)
        uncorr_thread = Thread(target=add_rect_image_and_index_to_queue,args=(uc_r,uncorr_queue,li))
        uncorr_thread.start()
        threads.append(uncorr_thread)
        mi_c_thread = Thread(target=add_rect_image_and_index_to_queue,args=(w_r,mi_c_queue))
        mi_c_thread.start()
        threads.append(mi_c_thread)
        basic_c_thread = Thread(target=add_basic_corr_rect_image_and_index_to_queue,
                                args=(uc_r,basic_ff,basic_df,basic_c_queue,li))
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

def add_overlap_comparison_to_queue(overlap,rects,rect_images_by_n,li,queue) :
    overlap = copy.deepcopy(overlap)
    p1_rect = [r for r in rects if r.n==overlap.p1]
    assert len(p1_rect)==1
    p1_rect = copy.deepcopy(p1_rect[0])
    p2_rect = [r for r in rects if r.n==overlap.p2]
    assert len(p2_rect)==1
    p2_rect = copy.deepcopy(p2_rect[0])
    p1_rect.image = rect_images_by_n[p1_rect.n]['uncorr']
    p2_rect.image = rect_images_by_n[p2_rect.n]['uncorr']
    overlap.myupdaterectangles([p1_rect,p2_rect])
    orig_result = copy.deepcopy(overlap.align(alreadyalignedstrategy='overwrite'))
    if orig_result.exit!=0 :
        queue.put(
        AlignmentOverlapComparison(overlap.n,li+1,overlap.p1,overlap.p2,overlap.tag,overlap.overlap_npix,
                                    -900.,-900.,-900.,-900.,
                                    -900.,-900.,-900.,-900.,
                                    -900.,-900.,-900.,-900.,1)
        )
        return
    #overlap.showimages(normalize=1000.)
    orig_dx = orig_result.dxvec[0].nominal_value
    orig_dy = orig_result.dxvec[1].nominal_value
    p1_rect.image = rect_images_by_n[p1_rect.n]['basic_c']
    p2_rect.image = rect_images_by_n[p2_rect.n]['basic_c']
    overlap.myupdaterectangles([p1_rect,p2_rect])
    basic_result = copy.deepcopy(overlap.align(alreadyalignedstrategy='overwrite'))
    if basic_result.exit!=0 :
        queue.put(
        AlignmentOverlapComparison(overlap.n,li+1,overlap.p1,overlap.p2,overlap.tag,overlap.overlap_npix,
                                    orig_dx,orig_dy,orig_result.mse[0],orig_result.mse[2],
                                    -900.,-900.,-900.,-900.,
                                    -900.,-900.,-900.,-900.,2)
        )
        return
    #overlap.showimages(normalize=1000.)
    basic_dx = basic_result.dxvec[0].nominal_value
    basic_dy = basic_result.dxvec[1].nominal_value
    p1_rect.image = rect_images_by_n[p1_rect.n]['mi_c']
    p2_rect.image = rect_images_by_n[p2_rect.n]['mi_c']
    overlap.myupdaterectangles([p1_rect,p2_rect])
    meanimage_result = copy.deepcopy(overlap.align(alreadyalignedstrategy='overwrite'))
    if meanimage_result.exit!=0 :
        queue.put(
        AlignmentOverlapComparison(overlap.n,li+1,overlap.p1,overlap.p2,overlap.tag,overlap.overlap_npix,
                                    orig_dx,orig_dy,orig_result.mse[0],orig_result.mse[2],
                                    basic_dx,basic_dy,basic_result.mse[0],basic_result.mse[2],
                                    -900.,-900.,-900.,-900.,3)
        )
        return
    #overlap.showimages(normalize=1000.)
    meanimage_dx = meanimage_result.dxvec[0].nominal_value
    meanimage_dy = meanimage_result.dxvec[1].nominal_value
    queue.put(
        AlignmentOverlapComparison(overlap.n,li+1,overlap.p1,overlap.p2,overlap.tag,overlap.overlap_npix,
                                    orig_dx,orig_dy,orig_result.mse[0],orig_result.mse[2],
                                    basic_dx,basic_dy,basic_result.mse[0],basic_result.mse[2],
                                    meanimage_dx,meanimage_dy,meanimage_result.mse[0],meanimage_result.mse[2],0)
        )
