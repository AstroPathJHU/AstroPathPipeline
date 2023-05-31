#imports
import copy, cv2
import numpy as np
from ...utilities.dataclasses import MyDataClass
from .config import CONST

#A single octet of overlaps
class OverlapOctet(MyDataClass) :
    slide_ID                : str
    layer                   : int
    counts_threshold        : float
    counts_per_ms_threshold : float
    p1_rect_n               : int
    olap_1_n                : int
    olap_2_n                : int
    olap_3_n                : int
    olap_4_n                : int
    olap_6_n                : int
    olap_7_n                : int
    olap_8_n                : int
    olap_9_n                : int
    olap_1_p1_pixel_frac    : float
    olap_2_p1_pixel_frac    : float
    olap_3_p1_pixel_frac    : float
    olap_4_p1_pixel_frac    : float
    olap_6_p1_pixel_frac    : float
    olap_7_p1_pixel_frac    : float
    olap_8_p1_pixel_frac    : float
    olap_9_p1_pixel_frac    : float
    olap_1_p2_pixel_frac    : float
    olap_2_p2_pixel_frac    : float
    olap_3_p2_pixel_frac    : float
    olap_4_p2_pixel_frac    : float
    olap_6_p2_pixel_frac    : float
    olap_7_p2_pixel_frac    : float
    olap_8_p2_pixel_frac    : float
    olap_9_p2_pixel_frac    : float
    @property
    def overlap_ns(self) :
        return [self.olap_1_n,self.olap_2_n,self.olap_3_n,self.olap_4_n,
                self.olap_6_n,self.olap_7_n,self.olap_8_n,self.olap_9_n]

#utility class for logging warping parameters and the slide they come from
class WarpingSummary(MyDataClass) :
    slide_ID        : str
    project         : int
    cohort          : int
    microscope_name : str
    first_layer_n   : int
    last_layer_n    : int
    n               : int
    m               : int
    cx              : float
    cy              : float
    fx              : float
    fy              : float
    k1              : float
    k2              : float
    k3              : float
    p1              : float
    p2              : float

#utility class to represent a warp fit result
class WarpFitResult(MyDataClass) :
    slide_ID       : str
    octet_center_n : int
    n              : int
    m              : int
    cx             : float
    cy             : float
    fx             : float
    fy             : float
    k1             : float
    k2             : float
    k3             : float
    p1             : float
    p2             : float
    max_rad_warp   : float
    max_tan_warp   : float
    fit_its        : int
    fit_time       : float
    raw_cost       : float
    best_cost      : float
    cost_reduction : float

#utilitiy class for logging fields used in fits
class FieldLog(MyDataClass) :
    file   : str
    rect_n : int

#helper function to find the limit on a parameter that produces the maximum warp
def find_default_parameter_limit(parindex,parincrement,warplimit,warpamtfunc,testpars) :
    warpamt=0.; testparval=0.
    while warpamt<warplimit :
        testparval+=parincrement
        testpars[parindex]=testparval
        warpamt=warpamtfunc(tuple(testpars))
    return testparval

#helper function to make the default list of parameter constraints
def build_default_parameter_bounds_dict(warp,max_rad_warp,max_tan_warp) :
    bounds = {}
    # cx/cy bounds are +/- 25% of the center point
    bounds['cx']=(0.5*(warp.n/2.),1.5*(warp.n/2.))
    bounds['cy']=(0.5*(warp.m/2.),1.5*(warp.m/2.))
    # fx/fy bounds are +/- 2% of the nominal values 
    bounds['fx']=(0.98*CONST.MICROSCOPE_OBJECTIVE_FOCAL_LENGTH,1.02*CONST.MICROSCOPE_OBJECTIVE_FOCAL_LENGTH)
    bounds['fy']=(0.98*CONST.MICROSCOPE_OBJECTIVE_FOCAL_LENGTH,1.02*CONST.MICROSCOPE_OBJECTIVE_FOCAL_LENGTH)
    # k1/k2/k3 and p1/p2 bounds are 2x those that would produce the max radial and tangential warp, respectively, 
    # with all others zero (except k1 can't be negative)
    testpars=[warp.cx,warp.cy,warp.fx,warp.fy,0.,0.,0.,0.,0.]
    maxk1 = find_default_parameter_limit(4,1,max_rad_warp,warp.maxRadialDistortAmount,copy.deepcopy(testpars))
    bounds['k1']=(0.,2.0*maxk1)
    maxk2 = find_default_parameter_limit(5,1000,max_rad_warp,warp.maxRadialDistortAmount,copy.deepcopy(testpars))
    bounds['k2']=(-2.0*maxk2,2.0*maxk2)
    maxk3 = find_default_parameter_limit(6,10000000,max_rad_warp,warp.maxRadialDistortAmount,copy.deepcopy(testpars))
    bounds['k3']=(-2.0*maxk3,2.0*maxk3)
    maxp1 = find_default_parameter_limit(7,0.01,max_tan_warp,warp.maxTangentialDistortAmount,copy.deepcopy(testpars))
    bounds['p1']=(-2.0*maxp1,2.0*maxp1)
    maxp2 = find_default_parameter_limit(8,0.01,max_tan_warp,warp.maxTangentialDistortAmount,copy.deepcopy(testpars))
    bounds['p2']=(-2.0*maxp2,2.0*maxp2)
    return bounds

#helper function to correct an image layer with given warp dx and dy fields
def correct_image_layer_with_warp_fields(raw_img_layer,dx_warps,dy_warps,interp_method=cv2.INTER_LINEAR,dest=None) :
    grid = np.mgrid[0:raw_img_layer.shape[0],0:raw_img_layer.shape[1]]
    xpos, ypos = grid[1], grid[0]
    map_x = (xpos-dx_warps).astype(np.float32) 
    map_y = (ypos-dy_warps).astype(np.float32)
    if dest is not None :
        return cv2.remap(raw_img_layer,map_x,map_y,interp_method,dest)
    else :
        return cv2.remap(raw_img_layer,map_x,map_y,interp_method)
