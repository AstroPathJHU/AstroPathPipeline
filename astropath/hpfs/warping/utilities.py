#imports
from ...utilities.dataclasses import MyDataClass

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
    mindate         : str
    maxdate         : str
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
