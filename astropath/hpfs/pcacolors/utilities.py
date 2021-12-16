#imports
import numpy as np
from scipy.ndimage import distance_transform_edt
from numba import njit, prange

def expand_labels(label_image, distance=1):
    """
    Direct rip of Scikit-image expand_labels which doesn't exist in 0.17
    https://github.com/scikit-image/scikit-image/blob/main/skimage/segmentation/_expand_labels.py#L16-L106
    """
    distances, nearest_label_coords = distance_transform_edt(label_image == 0, return_indices=True)
    labels_out = np.zeros_like(label_image)
    dilate_mask = distances <= distance
    masked_nearest_label_coords = [
        dimension_indices[dilate_mask]
        for dimension_indices in nearest_label_coords
    ]
    nearest_labels = label_image[tuple(masked_nearest_label_coords)]
    labels_out[dilate_mask] = nearest_labels
    return labels_out

@njit(parallel=True)
def get_median_im_compiled(im,n_regions,regions_im,pixels_to_use_ri) :
    """
    Replace all masked pixels with the medians of their surrounding pixels

    im = the original image with no pixels replaced
    n_regions = the number of independent image regions that show masked-out nucleus pixels
    regions_im = the image labeling the independent cell nuclei regions
    pixels_to_use_ri = an image where the pixels that should be used to compute the median 
                       for each region have the same label as their corresponding region to 
                       replace in regions_im
    """
    im_c = np.copy(im)
    nlayers = im.shape[-1]
    ydim = im.shape[0]
    xdim = im.shape[1]
    #for each interconnected region of nucleus pixels
    for ri in prange(1,n_regions) :
        npix = 0
        for iy in range(ydim) :
            for ix in range(xdim) :
                if pixels_to_use_ri[iy,ix]==ri :
                    npix+=1
        for li in prange(nlayers) :
            pixels = np.zeros((npix,),dtype=im.dtype)
            ipix=0
            for iy in range(ydim) :
                for ix in range(xdim) :
                    if pixels_to_use_ri[iy,ix]==ri :
                        pixels[ipix]=im[iy,ix,li]
                        ipix+=1
            mv = np.median(pixels)
            for iy in range(ydim) :
                for ix in range(xdim) :
                    if regions_im[iy,ix]==ri :
                        im_c[iy,ix,li] = mv
    return im_c
