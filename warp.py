#imports
import numpy as np
from .plotting import *
import os, math, cv2

class WarpingError(Exception) :
    """
    Class for errors encountered during warping
    """
    pass

class Warp :
    """
    Main superclass for applying warping to images
    """
    def __init__(self,n,m) :
        """
        Initializes a general warp to apply to images of a certain size
        n               = image width (pixels)
        m               = image height (pixels)
        """
        self.n = n
        self.m = m
        self.checkerboard = self.__makeCheckerboard()

    def getHWLFromRaw(self,fname,nlayers=35) :
        """
        Function to read a '.raw' binary file into an array of dimensions (height,width,nlayers) or (m,n,nlayers)
        """
        #get the .raw file as a vector of uint16s
        img = im3readraw(fname)
        #reshape it to match the warping field
        try :
            img_a = np.reshape(img,(nlayers,self.n,self.m),order="F")
        except ValueError :
            msg = f"ERROR: Raw image file shape ({nlayers} layers, {len(img)} total bytes) is mismatched to"
            msg+= f" dimensions (layers, width={self.n}, height={self.m})!"
            raise WarpingError(msg)
        #flip x and y dimensions to display image correctly, move layers to z-axis
        img_to_warp = np.transpose(img_a,(2,1,0))
        return img_to_warp

    def getSingleLayerImage(self,fname) :
        """
        Function to read a file that contains one layer of an image into an array with the Warp's dimensions 
        """
        #get the file as a vector of uint16s
        img = im3readraw(fname)
        #reshape it to match the warping field
        try :
            img_a = np.reshape(img,(self.m,self.n),order="F")
        except ValueError :
            msg = f"ERROR: single layer image file ({len(img)} total bytes) shape is mismatched to"
            msg+= f" dimensions (width={self.n}, height={self.m})!"
            raise WarpingError(msg)
        return img_a

    def writeSingleLayerImage(self,im,outfname) :
        """
        Function to write out an image as a properly transformed and flattened vector of uints 
        """
        #write out image flattened in fortran order
        im3writeraw(outfname,im.flatten(order="F").astype(np.uint16))

    def __makeCheckerboard(self) :
        """
        Function to create and return a checkerboard image of the appropriate size for visualizing warp effects
        """
        #find a good size to use for the squares
        square_sizes=range(1,176)
        square_pixels = max([s for s in square_sizes if self.m%s==0]+[s for s in square_sizes if self.n%s==0])
        #make an initial black image
        data = np.zeros((self.m,self.n),dtype=np.uint16)
        #make the white squares
        for i in range(data.shape[0]) :
            for j in range(data.shape[1]) :
                if math.floor(i/square_pixels)%2==math.floor(j/square_pixels)%2 :
                    data[i,j]=1
        return data


class PolyFieldWarp(Warp) :
    """
    Subclass for applying warping to images based on a polynomial fit to datapoints of warp factors vs. (scaled) distance or distance^2
    """
    def __init__(self,n=1344,m=1004,xc=584,yc=600,max_warp=1.85,pdegree=3,psq=False,interpolation=cv2.INTER_LINEAR,plot_fit=False,plot_warpfields=False) :
        """
        Initializes the warp_field based on a polynomial fit to scaled radial distance or scaled radial distance squared
        Fit range and warp parameters are hardcoded except for maximum warp at furthest location
        xc              = principal center point x coordinate
        yc              = principal center point y coordinate
        max_warp        = warping factor for furthest-from-center point in fit
        pdegree         = degree of polynomial fit to use
        psq             = if True, fit to a polynomial in r^2 instead of in r
        interpolation   = openCV interpolation parameter (see https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html)
        plot_fit        = if True, show plot of polynomial fit
        plot_warpfields = if True, show heatmaps of warp as radius and components of resulting gradient warp field
        """
        super().__init__(n,m)
        self.r_warps, self.x_warps, self.y_warps = self.__getWarpFields(max_warp,pdegree,psq,plot_fit)
        self.interp=interpolation
        #plot warp fields if requested
        if plot_warpfields : plotWarpFields(self.r_warps,self.x_warps,self.y_warps)

    def warpImage(self,infname,nlayers=35,layers=[0]) :
        """
        Read in an image, warp layer-by-layer with remap, and save each warped layer as its own new file in the current directory
        infname       = name of image file to split, warp, and re-save
        nlayers       = number of layers in original image file that's opened
        layers        = list (of integers) of layers to split out, warp, and save (index starting from 0)
        """
        #get the reshaped image from the raw file
        img_to_warp = self.getHWLFromRaw(infname,nlayers)
        #remap each layer
        for i in layers :
            self.warpLayer(img_to_warp[:,:,i],i,infname)

    def warpLayer(self,layer,layernumber,rawfilename) :
        """
        Quickly warps a single inputted image layer array with the current parameters and save it
        """
        self.writeSingleLayerImage(self.getWarpedLayer(layer),self._getWarpedLayerFilename(rawfilename,layernumber))

    def getWarpedLayer(self,layer) :
        """
        Quickly warps and returns a single inputted image layer array
        """
        map_x, map_y = self.__getMapMatrices()
        return cv2.remap(layer,map_x,map_y,self.interpolation)

    def showCheckerboard(self) :
        """
        Plot a checkerboard image before and after application of the warp
        """
        plotCheckerboards(self.checkerboard,self.getWarpedLayer(self.checkerboard))
    
    #helper function to make and return r_warps (field of warp factors) and x/y_warps (two fields of warp gradient dx/dy)
    def __getWarpFields(self,max_warp,pdegree,psq,plot_fit) :
        #define distance fields
        grid = np.mgrid[1:self.m+1,1:self.n+1]
        rescale = math.floor(min([self.xc,abs(self.n-self.xc),self.yc,abs(self.m-self.yc)])) #scale radius to be tangential to nearest edge
        x=(grid[1]-self.xc)/rescale #scaled x displacement from center
        y=(grid[0]-self.yc)/rescale #scaled y displacement from center
        r=np.sqrt(x**2+y**2)   #scaled total distance from center
        #fit polynomial to data
        coeffs = self.__polyFit(max_warp,pdegree,psq,plot=plot_fit)
        #make field of r-dependent corrections
        r_warps = np.zeros(r.shape)
        for i,c in enumerate(np.flip(coeffs)) :
            if psq :
                r_warps += coeffs[len(coeffs)-1-i]*np.power(r,2*i)
            else :
                r_warps += coeffs[len(coeffs)-1-i]*np.power(r,i)
        #translate r-dependent corrections to dx and dy shifts and return
        return r_warps, r_warps*x, r_warps*y

    #Alex's helper function to fit a polynomial to the warping given a max amount of warp
    def __polyFit(self,max_warp,deg,squared=False,plot=False) :
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

    def __getMapMatrices(self) :
        """
        calculate and return the map matrices for remap
        """
        grid = np.mgrid[1:self.m+1,1:self.n+1]
        xpos, ypos = grid[1], grid[0]
        map_x = (xpos-self.x_warps).astype(np.float32) 
        map_y = (ypos-self.y_warps).astype(np.float32) #maybe use maps from convertMaps() instead later on?
        return map_x, map_y

    #helper function to convert a raw file name and a layer into a fieldwarped single layer filename
    def _getWarpedLayerFilename(self,rawname,layer) :
        return (rawname.split(os.path.sep)[-1]).split(".")[0]+f".fieldWarp_layer{(layer+1):02d}"

class CameraWarp(Warp) :
    """
    Subclass for applying warping to images based on a camera matrix and distortion parameters
    """
    def __init__(self,n=1344,m=1004,cx=0.,cy=0.,fx=1.,fy=1.,k1=0.,k2=0.,p1=0.,p2=0.,k3=None,k4=None,k5=None,k6=None) :
        """
        Initialize a camera matrix and vector of distortion parameters for a camera warp transformation
        See https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html for explanations of parameters/functions
        cx         = principal center point x coordinate
        cy         = principal center point y coordinate
        fx         = x focal length (pixels)
        fy         = y focal length (pixels)
        k1, k2, k3 = radial distortion by third-order polynomial in r^2 (numerator)
        k4, k5, k6 = radial distortion by third-order polynomial in r^2 (denominator)
        p1, p2     = tangential distortion parameters
        """
        super().__init__(n,m)
        self.cam_matrix = np.array([[fx,0.,cx],[0.,fy,cy],[0.,0.,1.]])
        dplist = [k1,k2,p1,p2]
        for extrapar in [k3,k4,k5,k6] :
            if extrapar is not None : dplist.append(extrapar)
        self.dist_pars  = np.array(dplist)
        self.n_dist_pars = len(self.dist_pars)

    def warpImage(self,infname,nlayers=35,layers=[0]) :
        """
        Read in an image, warp layer-by-layer with undistort, and save each warped layer as its own new file in the current directory
        infname       = name of image file to split, warp, and re-save
        nlayers       = number of layers in original image file that's opened
        layers        = list (of integers) of layers to split out, warp, and save (index starting from 0)
        """
        #get the reshaped image from the raw file
        img_to_warp = self.getHWLFromRaw(infname,nlayers)
        #undistort each layer
        for i in layers :
            self.warpLayer(img_to_warp[:,:,i],i,infname)

    def warpLayer(self,layer,layernumber,rawfilename) :
        """
        Quickly warps a single inputted image layer array with the current parameters and save it
        """
        self.writeSingleLayerImage(self.getWarpedLayer(layer),self._getWarpedLayerFilename(rawfilename,layernumber))

    def getWarpedLayer(self,layer) :
        """
        Quickly warps and returns a single inputted image layer array
        """
        return cv2.undistort(layer,self.cam_matrix,self.dist_pars)

    def updateParams(self,pars) :
        """
        Update the camera matrix and distortion parameters for a new transformation on the same images
        pars = list of transformation parameters in order cx, cy, fx, fy, [dist_vec] (len 4-8, depending)
        """
        self.cam_matrix = np.array([[pars[2],0.,pars[0]],[0.,pars[3],pars[1]],[0.,0.,1.]])
        self.dist_pars  = np.array(pars[4:])

    def showCheckerboard(self) :
        """
        Plot a checkerboard image before and after application of the warp
        """
        plotCheckerboards(self.checkerboard,self.getWarpedLayer(self.checkerboard))

    #helper function to convert a raw file name and a layer into a camwarped single layer filename
    def _getWarpedLayerFilename(self,rawname,layer) :
        return (rawname.split(os.path.sep)[-1]).split(".")[0]+f".camWarp_layer{(layer+1):02d}"

#helper function to read the binary dump of a raw im3 file 
def im3readraw(f) :
    with open(f,mode='rb') as fp : #read as binary
        content = np.fromfile(fp,dtype=np.uint16)
    return content

#helper function to write an array of uint16s as an im3 file
def im3writeraw(outname,a) :
    with open(outname,mode='wb') as fp : #write as binary
        a.tofile(fp)
