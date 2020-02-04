#imports
import numpy as np
from .plotting import *
import math 
import cv2

class WarpingError(Exception) :
    """
    Class for errors encountered during warping
    """
    pass

class Warp :
    """
    Main class for applying warping to images
    """
    def __init__(self) :
        self.warp_field=None
        self.cam_matrix=None
        self.dist_pars=None

    #function to build initial guesses at the camera matrix and distortion parameters
    def initCamWarp(self,fx=1.,fy=1.,cx=584,cy=600,k1=0.,k2=0.,k3=0.,k4=0.,k5=0.,k6=0.,p1=0.,p2=0.) :
        """
        Initialize a camera matrix and vector of distortion parameters for a camera warp transformation
        See https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html for explanations of parameters/functions
        fx         = x focal length (pixels)
        fy         = y focal length (pixels)
        cx         = center principal point x pixel
        cy         = center principal point y pixel
        k1, k2, k3 = radial distortion by third-order polynomial in r^2 (numerator)
        k4, k5, k6 = radial distortion by third-order polynomial in r^2 (denominator)
        p1, p2     = tangential distortion parameters
        """
        self.cam_matrix = np.array([[fx,0.,cx],[0.,fy,cy],[0.,0.,1.]])
        self.dist_pars  = np.array([k1,k2,p1,p2,k3,k4,k5,k6])

    #Alex's function to define a warp field
    def initWarpField(self,n=1344,m=1004,xc=584,yc=600,max_warp=1.85,pdegree=3,psq=False,plot_fit=False,plot_warpfields=False) :
        """
        Initializes the object's warp_field based on a polynomial fit to scaled radial distance or scaled radial distance squared
        Fit range and warp parameters are hardcoded except for maximum warp at furthest location
        n               = image width (pixels)
        m               = image height (pixels)
        xc              = principal center point x coordinate
        yc              = principal center point y coordinate
        max_warp        = warping factor for furthest-from-center point in fit
        pdegree         = degree of polynomial fit to use
        psq             = if True, fit to a polynomial in r^2 instead of in r
        plot_fit        = if True, show plot of polynomial fit
        plot_warpfields = if True, show heatmaps of warp as radius and components of resulting gradient warp field
        """
        self.n = n
        self.m = m
        #define distance fields
        grid = np.mgrid[1:m+1,1:n+1]
        rescale = math.floor(min([xc,abs(n-xc),yc,abs(m-yc)])) #scale radius to be tangential to nearest edge
        x=(grid[1]-xc)/rescale #scaled x displacement from center
        y=(grid[0]-yc)/rescale #scaled y displacement from center
        r=np.sqrt(x**2+y**2)   #scaled total distance from center
        #fit polynomial to data
        coeffs = self._polyFit(max_warp,pdegree,psq,plot=plot_fit)
        #make field of r-dependent corrections
        r_warps = np.zeros(r.shape)
        for i,c in enumerate(np.flip(coeffs)) :
            if psq :
                r_warps += coeffs[len(coeffs)-1-i]*np.power(r,2*i)
            else :
                r_warps += coeffs[len(coeffs)-1-i]*np.power(r,i)
        #translate r-dependent corrections to dx and dy shifts
        d_warps = np.zeros(r.shape+(2,))
        d_warps[:,:,0] = r_warps*x
        d_warps[:,:,1] = r_warps*y
        #plot warp fields if requested
        if plot_warpfields :
            plotWarpFields(r_warps,d_warps)
        #set double layer field of x and y shifts
        self.warp_field = d_warps

    #Alex's helper function to fit a polynomial to the warping given a max amount of warp
    def _polyFit(self,max_warp,deg,squared=False,plot=False) :
        #define warping as a function of distance or distance^2
        r_points = np.array([0.0,0.2,0.4,0.8,1.4])
        if squared :
            r_points = r_points**2
        warp_amt = np.array([0.0,0.0,0.0,0.2,max_warp])
        #fit polynomial (***** may want to do this with scipy.optimize instead *****)
        coeffs = np.polyfit(r_points,warp_amt,deg)
        #make plot if requested
        if plot :
            plotPolyFit(r_points,warp_amt,coeffs,squared)
        #return coefficients
        return coeffs

    #function to apply a warp field to multiple layers of one image and write each as a new file
    def warpImageWithField(self,infname,layers=[*range(35)],nlayers=35,interpolation=cv2.INTER_LINEAR) :
        """
        Read in an image, warp layer-by-layer with remap based on warp_field, and save each warped layer as its own new file
        infname       = name of image file to split, warp, and re-save
        layers        = list of integers of layers to split out, warp, and save (index starting from 0)
        nlayers       = number of layers in original image file that's opened
        interpolation = openCV interpolation parameter (see https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html)
        """
        if self.warp_field is None:
            raise WarpingError("ERROR: must call initWarpField before warpImageWithField")
        #get the .raw file as a vector of uint16s
        img = im3readraw(infname)
        #reshape it to match the warping field
        try :
            img_a = np.reshape(img,(nlayers,)+self.warp_field.shape[:2][::-1],order="F") #dim. rev. due to MATLAB/python order mismatch
        except ValueError :
            msg = f"ERROR: Raw image file shape ({nlayers} layers, {len(img)} total bytes) is mismatched to"
            msg+= f" warp field dimensions (shape={self.warp_field.shape})!"
            raise WarpingError(msg)
        #flip x and y dimensions to match warp field, move layers to z-axis
        img_to_warp = np.transpose(img_a,(2,1,0))
        #use the warp matrix to calculate the map matrices for remap
        grid = np.mgrid[1:self.m+1,1:self.n+1]
        xpos, ypos = grid[1], grid[0]
        map_x = (xpos-self.warp_field[:,:,0]).astype(np.float32); map_y = (ypos-self.warp_field[:,:,1]).astype(np.float32) #maybe use maps from convertMaps() instead later on?
        #remap each layer
        for i in layers :
            layer_warped = cv2.remap(img_to_warp[:,:,i],map_x,map_y,interpolation)
            #transpose the resulting layer image back to its original dimensions
            layer_transposed = np.transpose(layer_warped,(1,0))
            #write out image flattened in fortran order
            outfname = infname.split(".")[0]+f".fieldWarp_layer{(i+1):02d}"
            im3writeraw(outfname,layer_transposed.flatten(order="F").astype(np.uint16))

#helper function to read the binary dump of a raw im3 file 
def im3readraw(f) :
    with open(f,mode='rb') as fp : #read as binary
        content = np.fromfile(fp,dtype=np.uint16)
    return content

#helper function to write an array of uint16s as an im3 file
def im3writeraw(outname,a) :
    with open(outname,mode='wb') as fp : #write as binary
        a.tofile(fp)
