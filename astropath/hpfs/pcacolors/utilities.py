#imports
import cv2
import numpy as np
from numba import njit, prange
from skimage.segmentation import expand_labels
from ...shared.image_masking.utilities import get_size_filtered_mask
from ...utilities.dataclasses import MyDataClass
from ...utilities.img_file_io import smooth_image_worker

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
    this is some new text : )
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
        if npix<=1 :
            continue
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

def get_homogenized_pca_image(im,tissue_mask,pca,dapi_layer_index=0,threshold=0.9,expand_distance=5,
                              small_morph_size=9,large_morph_size=15,min_cluster_size=500,dapi_smooth_sigma=3) :
    """
    Given a raw image and the PCA transformation for the image's sample, return the normalized PCA image
    with the colors of cell nuclei replaced with the median colors of the pixels just outside them.

    Pixels corresponding to cell nuclei are determined via a simple thresholding on a single layer 
    of the PCA image (index given) corresponding to mostly DAPI signal. Large clusters of masked pixels 
    have some of the DAPI signal added back in to better represent the colors of dense clusters of cells

    im = the original HPF image
    tissue_mask = the tissue mask for the image (background pixels are excluded from the median calculation)
    pca = the PCA for the sample to which the image belongs
    dapi_layer_index = the index of the layer of PCA images corresponding mostly to DAPI signal
    threshold = the threshold in the DAPI layer above which a pixel should be masked as corresponding to a cell nucleus
    expand_distance = how large the ring around each masked pixel should extend for the median color calculation
    small/large_morph_size = sizes of the windows that should be used in the morphology transformations to determine 
        clusters of masked pixels that should have some DAPI signal added back
    min_cluster_size = the minimum size of masked pixel clusters that should have some DAPI signal added back
    dapi_smooth_sigma = the sigma for the Gaussian filter applied to the DAPI layer before adding it back 
        to masked pixels in large clusters
    """
    #do the PCA of the raw image
    x = im.reshape(im.shape[0]*im.shape[1],im.shape[2])
    x_transformed = pca.transform(x)
    im_pca = x_transformed.reshape(im.shape[0],im.shape[1],im.shape[2])
    #normalize it by its intensity
    im_intensity = np.sqrt(np.sum(np.power(im_pca,2),axis=2))
    im_normed = im_pca/im_intensity[:,:,np.newaxis]
    #threshold on the normalized DAPI layer to mask out nuclei pixels
    nuclei_mask = np.zeros(im.shape[:-1],dtype=np.uint8)
    nuclei_mask[im_normed[:,:,dapi_layer_index]>threshold] = 1
    #define the pixels to use to compute the median for each region of nuclear pixels
    n_regions, regions_im = cv2.connectedComponents(nuclei_mask)
    expanded_regions_im = expand_labels(regions_im,distance=expand_distance)
    pixels_to_use_ri = np.copy(expanded_regions_im)
    pixels_to_use_ri[(nuclei_mask==1) | (tissue_mask==0)] = 0
    #replace masked pixels with the medians of the pixels surrounding them
    h_pca_im = get_median_im_compiled(im_normed,n_regions,regions_im,pixels_to_use_ri)
    #finding large chunks of nuclei pixels that should have some of the DAPI signal added back in
    SMALL_EL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(small_morph_size,small_morph_size))
    LARGE_EL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(large_morph_size,large_morph_size))
    nuclei_mask_umat = cv2.UMat(nuclei_mask)
    nuclei_mask_morphed_umat = cv2.UMat(np.zeros_like(nuclei_mask))
    cv2.morphologyEx(nuclei_mask_umat,cv2.MORPH_OPEN,SMALL_EL,nuclei_mask_morphed_umat,borderType=cv2.BORDER_REPLICATE)
    cv2.morphologyEx(nuclei_mask_morphed_umat,cv2.MORPH_CLOSE,LARGE_EL,nuclei_mask_morphed_umat,
                     borderType=cv2.BORDER_REPLICATE)
    cv2.morphologyEx(nuclei_mask_morphed_umat,cv2.MORPH_OPEN,SMALL_EL,nuclei_mask_morphed_umat,
                     iterations=2,borderType=cv2.BORDER_REPLICATE)
    cv2.morphologyEx(nuclei_mask_morphed_umat,cv2.MORPH_CLOSE,LARGE_EL,nuclei_mask_morphed_umat,
                     iterations=2,borderType=cv2.BORDER_REPLICATE)
    cv2.morphologyEx(nuclei_mask_morphed_umat,cv2.MORPH_OPEN,LARGE_EL,nuclei_mask_morphed_umat,
                     borderType=cv2.BORDER_REPLICATE)
    nuclei_mask_morphed = nuclei_mask_morphed_umat.get()
    nuclei_mask_morphed = get_size_filtered_mask(nuclei_mask_morphed,min_cluster_size)
    nuclei_mask_morphed_umat = cv2.UMat(nuclei_mask_morphed)
    cv2.morphologyEx(nuclei_mask_morphed_umat,cv2.MORPH_DILATE,LARGE_EL,nuclei_mask_morphed_umat,
                     borderType=cv2.BORDER_REPLICATE)
    nuclei_mask_morphed = nuclei_mask_morphed_umat.get()
    #find the pixels whose PCA DAPI contents should be replaced
    pixels_to_replace = nuclei_mask*nuclei_mask_morphed
    #smooth the PCA DAPI layer
    smoothed_pca_dapi_layer = smooth_image_worker(im_normed[:,:,dapi_layer_index],dapi_smooth_sigma,gpu=True)
    #replace the homogenized PCA image DAPI contents
    p_slice = pixels_to_replace==1
    h_pca_im[:,:,dapi_layer_index][p_slice] = smoothed_pca_dapi_layer[p_slice]
    return h_pca_im

class PCAColorsSegmentTableEntry(MyDataClass) :
    im_key : str
    segment : int
    npix : int
    comp_1_mean : float
    comp_1_median : float
    comp_1_std : float
    comp_2_mean : float
    comp_2_median : float
    comp_2_std : float
    comp_3_mean : float
    comp_3_median : float
    comp_3_std : float
    comp_4_mean : float
    comp_4_median : float
    comp_4_std : float
    comp_5_mean : float
    comp_5_median : float
    comp_5_std : float
    comp_6_mean : float
    comp_6_median : float
    comp_6_std : float
    comp_7_mean : float
    comp_7_median : float
    comp_7_std : float
    comp_8_mean : float
    comp_8_median : float
    comp_8_std : float
    comp_9_mean : float
    comp_9_median : float
    comp_9_std : float
    comp_10_mean : float
    comp_10_median : float
    comp_10_std : float
