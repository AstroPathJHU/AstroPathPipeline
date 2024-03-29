#imports
import pathlib, math, cv2, functools, methodtools, more_itertools
import numpy as np, seaborn as sns, matplotlib.pyplot as plt
from ...shared.astropath_logging import printlogger
from ...utilities.config import CONST as UNIV_CONST
from ...utilities.img_file_io import get_raw_as_hwl, get_raw_as_hw, write_image_to_file
from ...utilities.miscplotting import save_figure_in_dir
from .utilities import correct_image_layer_with_warp_fields
from .config import CONST

#################### SOME CACHED FILE-SCOPE HELPER FUNCTIONS ####################

#cached radial distortion amount
@functools.lru_cache()
def radialDistortAmountAtCoords(coord_x,coord_y,fx,fy,k1,k2,k3) :
    r2 = coord_x**2 + coord_y**2
    return (k1*r2 + k2*(r2**2) + k3*(r2**3))*math.sqrt((fx*coord_x)**2 + (fy*coord_y)**2)

#cached tangential distortion amount
@functools.lru_cache()
def tangentialDistortAmountAtCoords(coord_x,coord_y,fx,fy,p1,p2) :
    r2 = coord_x**2+coord_y**2
    dx = fx*(2.*p1*coord_x*coord_y + p2*(r2 + 2.*(coord_x**2)))
    dy = fy*(2.*p2*coord_x*coord_y + p1*(2.*(coord_y**2) + r2))
    return math.sqrt((dx)**2 + (dy)**2)

#cached radial distortion amount jacobian
@functools.lru_cache()
def radialDistortAmountAtCoordsJacobian(coord_x,coord_y,fx,fy,k1,k2,k3) :
    r2 = coord_x**2 + coord_y**2
    A = math.sqrt((fx*coord_x)**2 + (fy*coord_y)**2)
    B = k1*(r2) + k2*(r2**2) + k3*(r2**3)
    C = 2*k1 + 4*k2*r2 + 6*k3*r2**2
    dfdcx = ((-1.*A*coord_x)/(fx))*C - (B/A)*fx*coord_x
    dfdcy = ((-1.*A*coord_y)/(fy))*C - (B/A)*fy*coord_y
    dfdfx = ((-1.*A*(coord_x**2))/(fx))*C
    dfdfy = ((-1.*A*(coord_y**2))/(fy))*C
    dfdk1 = A*(r2)
    dfdk2 = A*(r2**2)
    dfdk3 = A*(r2**3)
    return [dfdcx,dfdcy,dfdfx,dfdfy,dfdk1,dfdk2,dfdk3]

#cached tangential distortion amount jacobian
@functools.lru_cache()
def tangentialDistortAmountAtCoordsJacobian(coord_x,coord_y,fx,fy,p1,p2) :
    r2 = coord_x**2+coord_y**2
    dx = fx*(2.*p1*coord_x*coord_y + p2*(r2 + 2.*(coord_x**2)))
    dy = fy*(2.*p2*coord_x*coord_y + p1*(2.*(coord_y**2) + r2))
    F = math.sqrt(dx**2 + dy**2)
    dfdcx = (1./F)*(-2.*dx*p1*coord_y - 4.*dx*p2*coord_x - 2.*dx*(fx**2)*p2*coord_x - 2.*dy*(fy/fx)*p2*coord_y - 2.*dy*fx*fy*p1*coord_x)
    dfdcy = (1./F)*(-2.*dy*p2*coord_x - 4.*dy*p1*coord_y - 2.*dy*(fy**2)*p1*coord_y - 2.*dx*(fx/fy)*p1*coord_x - 2.*dx*fy*fx*p2*coord_y)
    dfdfx = (1./F)*(dx*(p2*r2 - 2.*p2*(coord_x**2)) - 2.*dy*(fy/fx)*p2*coord_x*coord_y)
    dfdfy = (1./F)*(dy*(p1*r2 - 2.*p1*(coord_y**2)) - 2.*dx*(fx/fy)*p1*coord_y*coord_x)
    dfdp1 = (1./F)*(2*dx*fx*coord_x*coord_y + 2*dy*fy*(coord_y**2) + dy*fy*r2)
    dfdp2 = (1./F)*(2*dy*fy*coord_x*coord_y + 2*dx*fx*(coord_x**2) + dx*fx*r2)
    return [dfdcx,dfdcy,dfdfx,dfdfy,dfdp1,dfdp2]

#################### MAIN WARP CLASS ####################
class Warp :
    """
    Main superclass for applying warping to images
    """

    #################### PUBLIC FUNCTIONS ####################

    def __init__(self,n,m) :
        """
        Initializes a general warp to apply to images of a certain size
        n               = image width (pixels)
        m               = image height (pixels)
        """
        self.n = n
        self.m = m
        self._checkerboard = self.__makeCheckerboard()

    def getHWLFromRaw(self,fname,nlayers=35) :
        """
        Function to return a '.Data.dat' binary file as an array of dimensions (height,width,nlayers) or (m,n,nlayers)
        """
        return get_raw_as_hwl(fname,self.m,self.n,nlayers)

    def getSingleLayerImage(self,fname) :
        """
        Function to read a file that contains one layer of an image into an array with the Warp's dimensions 
        """
        return get_raw_as_hw(fname,self.m,self.n)

    def writeSingleLayerImage(self,im,outfname) :
        """
        Function to write out an image as a properly transformed and flattened vector of uints 
        """
        write_image_to_file(im,outfname)

    #################### PRIVATE HELPER FUNCTIONS ####################

    #helper function to plot unwarped and warped checkerboard images next to one another
    def _plotCheckerboards(self,warped_board) :
        f,(ax1,ax2) = plt.subplots(1,2)
        f.set_size_inches(20.,5.)
        ax1.imshow(self._checkerboard,cmap='gray')
        ax1.set_title('original')
        ax2.imshow(warped_board,cmap='gray')
        ax2.set_title('warped')
        plt.show()

    #Helper function to create and return a checkerboard image of the appropriate size for visualizing warp effects
    def __makeCheckerboard(self) :
        #find a good size to use for the squares
        square_sizes=range(1,176)
        square_pixels = max([s for s in square_sizes if self.m%s==0]+[s for s in square_sizes if self.n%s==0])
        #make an initial black image
        data = np.zeros((self.m,self.n),dtype=np.uint16)+0.5
        #make the white squares
        for i in range(data.shape[0]) :
            for j in range(data.shape[1]) :
                if math.floor(i/square_pixels)%2==math.floor(j/square_pixels)%2 :
                    data[i][j]=1
        return data

class PolyFieldWarp(Warp) :
    """
    Subclass for applying warping to images based on a polynomial fit to datapoints of 
    warp factors vs. (scaled) distance or distance^2
    """

    #################### PUBLIC FUNCTIONS ####################

    def __init__(self,n=1344,m=1004,xc=584,yc=600,max_warp=1.85,pdegree=3,psq=False,interpolation=cv2.INTER_LINEAR,
                 plot_fit=False) :
        """
        Initializes the warp_field based on a polynomial fit to scaled radial distance or scaled radial distance squared
        Fit range and warp parameters are hardcoded except for maximum warp at furthest location
        xc            = principal center point x coordinate
        yc            = principal center point y coordinate
        max_warp      = warping factor for furthest-from-center point in fit
        pdegree       = degree of polynomial fit to use
        psq           = if True, fit to a polynomial in r^2 instead of in r
        interpolation = openCV interpolation parameter 
                        (see https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html)
        plot_fit      = if True, show plot of polynomial fit
        """
        super().__init__(n,m)
        self.xc=xc
        self.yc=yc
        self.r_warps, self.x_warps, self.y_warps = self.__getWarpFields(xc,yc,max_warp,pdegree,psq,plot_fit)
        self.interp=interpolation

    def warpAndWriteImage(self,infname,nlayers=35,layers=[1]) :
        """
        Read in an image, warp layer-by-layer with remap, and save each warped layer as its own new file
        infname = name of image file to split, warp, and re-save
        nlayers = number of layers in original image file that's opened
        layers  = list (of integers) of layers to split out, warp, and save (index starting from 1)
        """
        #get the reshaped image from the raw file
        img_to_warp = self.getHWLFromRaw(infname,nlayers)
        #remap each layer
        for i in layers :
            self.warpAndWriteLayer(img_to_warp[:,:,i-1],i,infname)

    def warpAndWriteLayer(self,layer,layernumber,rawfilename) :
        """
        Quickly warps a single inputted image layer array with the current parameters and save it
        """
        self.writeImageLayer(self.getWarpedLayer(layer),rawfilename,layernumber)

    def getWarpedLayer(self,layer) :
        """
        Warps and returns a single inputted image layer array
        """
        return correct_image_layer_with_warp_fields(layer,self.x_warps,self.y_warps,self.interp)

    def warpLayerInPlace(self,layer,dest) :
        """
        Warps a single inputted image layer into the provided destination
        """
        return correct_image_layer_with_warp_fields(layer,self.x_warps,self.y_warps,self.interp,dest)

    def writeImageLayer(self,im,rawfilename,layernumber) :
        """
        Write out a single given image layer
        """
        self.writeSingleLayerImage(im,self.__getWarpedLayerFilename(rawfilename,layernumber))

    #################### VISUALIZATION FUNCTIONS ####################

    def writeOutWarpFields(self,file_stem) :
        """
        Write out .bin files of the dx and dy warping fields and also make an image showing them 
        file_stem = the unique identifier to add to the .bin filenames
        """
        write_image_to_file(self.x_warps,f'dx_warp_field_{file_stem}.bin',dtype=CONST.OUTPUT_FIELD_DTYPE)
        write_image_to_file(self.y_warps,f'dy_warp_field_{file_stem}.bin',dtype=CONST.OUTPUT_FIELD_DTYPE)
        f,ax = plt.subplots(1,3,figsize=(3*6.4,(self.m/self.n)*6.4))
        pos = ax[0].imshow(self.r_warps)
        ax[0].scatter(self.xc,self.yc,marker='*',color='yellow')
        ax[0].set_title('total warp')
        f.colorbar(pos,ax=ax[0])
        pos = ax[1].imshow(self.x_warps)
        ax[1].scatter(self.xc,self.yc,marker='*',color='yellow')
        ax[1].set_title('dx warp')
        f.colorbar(pos,ax=ax[1])
        pos = ax[2].imshow(self.y_warps)
        ax[2].scatter(self.xc,self.yc,marker='*',color='yellow')
        ax[2].set_title('dy warp')
        f.colorbar(pos,ax=ax[2])
        fn = f'{CONST.WARP_FIELD_FIGURE_NAME}_{file_stem}.png'
        save_figure_in_dir(plt,fn)

    def showCheckerboard(self) :
        """
        Plot a checkerboard image before and after application of the warp
        """
        self._plotCheckerboards(self.getWarpedLayer(self._checkerboard))

    #################### PRIVATE HELPER FUNCTIONS ####################
    
    def __getWarpFields(self,xc,yc,max_warp,pdegree,psq,plot_fit) :
        """
        make and return r_warps (field of warp factors) and x/y_warps (two fields of warp gradient dx/dy)
        """
        #define distance fields
        grid = np.mgrid[0:self.m,0:self.n]
        rescale=500. #Alex's parameter
        #rescale = math.floor(min([xc,abs(self.n-xc),yc,abs(self.m-yc)])) #scale radius to be tangential to nearest edge
        x=(grid[1]-xc)/rescale #scaled x displacement from center
        y=(grid[0]-yc)/rescale #scaled y displacement from center
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
        x_warps = np.zeros_like(r_warps); y_warps = np.zeros_like(r_warps)
        x_warps[r!=0] = r_warps[r!=0]*(x[r!=0]/r[r!=0]); y_warps[r!=0] = r_warps[r!=0]*(y[r!=0]/r[r!=0])
        r_warps = r_warps.astype(CONST.OUTPUT_FIELD_DTYPE)
        x_warps = (-x_warps).astype(CONST.OUTPUT_FIELD_DTYPE) #signs flipped so they behave like cv2.undistort
        y_warps = (-y_warps).astype(CONST.OUTPUT_FIELD_DTYPE) #signs flipped so they behave like cv2.undistort
        return r_warps, x_warps, y_warps

    #Alex's helper function to fit a polynomial to the warping given a max amount of warp
    def __polyFit(self,max_warp,deg,squared=False,plot=False) :
        #define warping as a function of distance or distance^2
        r_points = np.array([0.0,0.2,0.4,0.8,1.4])
        if squared :
            r_points = r_points**2
        warp_amt = np.array([0.0,0.0,0.0,0.2,max_warp])
        #fit polynomial 
        coeffs = np.polyfit(r_points,warp_amt,deg)
        #make plot if requested
        if plot :
            self.__plotPolyFit(r_points,warp_amt,coeffs,squared)
        #return coefficients
        return coeffs

    #helper function to plot a curve fit to data
    def __plotPolyFit(self,x,y,c,squared,npoints=50) :
        plt.plot(x,y,".",label="Datapoints")
        fit_xs = np.linspace(x[0],x[-1],npoints)
        fit_ys = np.polynomial.polynomial.polyval(fit_xs,np.flip(c))
        plt.plot(fit_xs,fit_ys,label="Fit")
        deg=len(c)-1
        if squared :
            plt.title(f"Fit with degree {deg} squared polynomial")
        else :
            plt.title(f"Fit with degree {deg} polynomial")
        if squared : 
            plt.xlabel("scaled squared radial distance r^2") 
        else : 
            plt.xlabel("scaled radial distance r")
        plt.ylabel("warp amount")
        ftext="warp="
        for i,coeff in enumerate(np.flip(c)) :
            ftext+=f"{coeff:.04f}"
            if i!=0 :
                ftext += "*r"
                if squared and i==1 :
                    ftext+='^2'
                if i!=1 :
                    if squared :
                        ftext += f"^{2*i}"
                    else :
                        ftext += f"^{i}"
            if i!=deg :
                ftext+=" + "
        plt.text(x[0],0.5*(fit_ys[-1]-fit_ys[0]),ftext)
        plt.legend()
        plt.show()

    #helper function to convert a raw file name and a layer into a fieldwarped single layer filename
    def __getWarpedLayerFilename(self,rawname,layer) :
        return (str((((pathlib.Path(rawname)).absolute()).parts)[-1])).split(".")[0]+f".fieldWarp_layer{(layer):02d}"

class CameraWarp(Warp) :
    """
    Subclass for applying warping to images based on a camera matrix and distortion parameters
    """

    #################### PUBLIC FUNCTIONS ####################

    def __init__(self,n=1344,m=1004,cx=None,cy=None,fx=40000.,fy=40000.,
                 k1=0.,k2=0.,k3=0.,p1=0.,p2=0.,k4=None,k5=None,k6=None) :
        """
        Initialize a camera matrix and vector of distortion parameters for a camera warp transformation
        See https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html 
        for explanations of parameters/functions
        Parameters k3, k4, k5, and k6 are all optional, but if k4 is defined k5 and k6 must be as well.
        cx         = principal center point x coordinate
        cy         = principal center point y coordinate
        fx         = x focal length (pixels)
        fy         = y focal length (pixels)
        k1, k2, k3 = coeffcients for radial distortion by third-order polynomial in r^2 (numerator)
        p1, p2     = tangential distortion parameters
        k4, k5, k6 = coeffcients for radial distortion by third-order polynomial in r^2 (denominator)
        """
        super().__init__(n,m)
        if cx is None :
            self.cx=n/2.
        else :
            self.cx=cx
        if cy is None :
            self.cy=m/2.
        else :
            self.cy=cy
        self.fx=fx; self.fy=fy
        self.k1=k1; self.k2=k2
        self.p1=p1; self.p2=p2
        self.k3=k3
        self.k4=k4; self.k5=k5; self.k6=k6
        self.__cam_matrix = None 
        self.__dist_pars  = None
        self.__calculateWarpObjects()

    def parValueFromName(self,pname) :
        """
        Given a parameter name string, return the currently-set value of the parameter
        """
        if pname=='cx' :
            return self.cx
        elif pname=='cy' :
            return self.cy
        elif pname=='fx' :
            return self.fx
        elif pname=='fy' :
            return self.fy
        elif pname=='k1' :
            return self.k1
        elif pname=='k2' :
            return self.k2
        elif pname=='k3' :
            return self.k3
        elif pname=='p1' :
            return self.p1
        elif pname=='p2' :
            return self.p2
        elif pname=='k4' :
            return self.k4
        elif pname=='k5' :
            return self.k5
        elif pname=='k6' :
            return self.k6
        else :
            raise ValueError(f'ERROR: parameter name {pname} not recognized!')

    def warpAndWriteImage(self,infname,nlayers=35,layers=[1]) :
        """
        Read in an image, warp layer-by-layer with undistort, and save each warped layer as its own new file
        infname       = name of image file to split, warp, and re-save
        nlayers       = number of layers in original image file that's opened
        layers        = list (of integers) of layers to split out, warp, and save (index starting from 1)
        """
        #get the reshaped image from the raw file
        img_to_warp = self.getHWLFromRaw(infname,nlayers)
        #undistort each layer
        for i in layers :
            self.warpAndWriteLayer(img_to_warp[:,:,i-1],i,infname)

    def warpAndWriteLayer(self,layer,layernumber,rawfilename) :
        """
        Quickly warps a single inputted image layer array with the current parameters and save it
        """
        self.writeImageLayer(self.getWarpedLayer(layer),rawfilename,layernumber)

    def getWarpedLayer(self,layerimg) :
        """
        Quickly warps and returns a single inputted image layer array
        """
        #return the result of undistort
        return cv2.undistort(layerimg,self.__cam_matrix,self.__dist_pars)

    def warpLayerInPlace(self,layerimg,dest) :
        """
        Quickly warps a single inputted image layer into the provided destination
        """
        cv2.undistort(layerimg,self.__cam_matrix,self.__dist_pars,dest)

    def writeImageLayer(self,im,rawfilename,layernumber) :
        """
        Write out a single given image layer
        """
        self.writeSingleLayerImage(im,self.__getWarpedLayerFilename(rawfilename,layernumber))

    #################### PUBLIC UTILITY FUNCTIONS ####################

    @methodtools.lru_cache()
    def getCoordsFromPixel(self,pixel_x,pixel_y) :
        """
        Convert a pixel to an x/y coordinate with units of x/y focal lengths
        """
        return (pixel_x-self.cx)/self.fx, (pixel_y-self.cy)/self.fy

    def radialDistortAmountAtPixel(self,pixel_x,pixel_y) :
        """
        Return the amount of radial warp (in pixels) at the given pixel
        """
        x, y = self.getCoordsFromPixel(pixel_x,pixel_y)
        return radialDistortAmountAtCoords(x,y,self.fx,self.fy,self.k1,self.k2,self.k3)

    def tangentialDistortAmountAtPixel(self,pixel_x,pixel_y) :
        """
        Return the amount of tangential warp (in pixels) at the given pixel
        """
        x, y = self.getCoordsFromPixel(pixel_x,pixel_y)
        return tangentialDistortAmountAtCoords(x,y,self.fx,self.fy,self.p1,self.p2)

    #################### FUNCTIONS FOR USE WITH MINIMIZATION ####################

    # !!!!!! For the time being, these functions don't correctly describe dependence on k4, k5, or k6 !!!!!!

    def updateParams(self,pars) :
        """
        Update the camera matrix and distortion parameters for a new transformation on the same images
        pars = list of transformation parameters in order cx, cy, fx, fy, k1, k2, k3, p1, p2[, k4, k5, k6 (optional)]
        """
        self.cx=pars[0]; self.cy=pars[1]
        self.fx=pars[2]; self.fy=pars[3]
        self.k1=pars[4]; self.k2=pars[5]; self.k3=pars[6]
        self.p1=pars[7]; self.p2=pars[8]
        if len(pars)>9 :
            self.k4=pars[9]; self.k5=pars[10]; self.k6=pars[11]
        self.__calculateWarpObjects()

    def maxRadialDistortAmount(self,pars) :
        """
        Return the maximum amount of radial distortion (in pixels) observed with the given parameters
        """
        if pars is None :
            cx, cy, fx, fy, k1, k2, k3 = self.cx, self.cy, self.fx, self.fy, self.k1, self.k2, self.k3
        else :
            cx,cy,fx,fy,k1,k2,k3,_,_ = (*pars,)
        x, y = self._getMaxDistanceCoords(cx,cy)
        return radialDistortAmountAtCoords(x,y,fx,fy,k1,k2,k3)

    def maxTangentialDistortAmount(self,pars) :
        """
        Return the maximum amount of tangential distortion (in pixels) observed with the given parameters
        """
        if pars is None :
            cx, cy, fx, fy, p1, p2 = self.cx, self.cy, self.fx, self.fy, self.p1, self.p2
        else :
            cx,cy,fx,fy,_,_,_,p1,p2 = (*pars,)
        x, y = self._getMaxDistanceCoords(cx,cy)
        return tangentialDistortAmountAtCoords(x,y,fx,fy,p1,p2)

    def maxRadialDistortAmountJacobian(self,pars) :
        """
        Return the Jacobian vector of the maxRadialDistortAmount function (used in minimization)
        """
        if pars is None :
            cx, cy, fx, fy, k1, k2, k3 = self.cx, self.cy, self.fx, self.fy, self.k1, self.k2, self.k3
        else :
            cx,cy,fx,fy,k1,k2,k3,_,_ = (*pars,)
        x, y = self._getMaxDistanceCoords(cx,cy)
        cxcyfxfyk1k2k3_dependence = radialDistortAmountAtCoordsJacobian(x,y,fx,fy,k1,k2,k3)
        retvec = cxcyfxfyk1k2k3_dependence #add cx, cy, fx, fy, k1, k2, and k3 dependency
        retvec+=[0.,0.] # no dependence on p1/p2
        return retvec 

    def maxTangentialDistortAmountJacobian(self,pars) :
        """
        Return the Jacobian vector of the maxTangentialDistortAmount function (used in minimization)
        """
        if pars is None :
            cx, cy, fx, fy, p1, p2 = self.cx, self.cy, self.fx, self.fy, self.p1, self.p2
        else :
            cx,cy,fx,fy,_,_,_,p1,p2 = (*pars,)
        x, y = self._getMaxDistanceCoords(cx,cy)
        cxcyfxfyp1p2_dependence = tangentialDistortAmountAtCoordsJacobian(x,y,fx,fy,p1,p2)
        retvec =cxcyfxfyp1p2_dependence[:4] #add dependence on cx, cy, fy, and fy
        retvec+=[0.,0.,0.] # no dependence on k1/k2/k3
        retvec+=cxcyfxfyp1p2_dependence[4:] #add dependence on p1 and p2
        return retvec

    def _getMaxDistanceCoords(self,cx,cy) :
        """
        Get the x/y coordinate-space location of the image corner that is furthest from the principal point
        """
        x = 0 if cx>(self.n-1)/2 else self.n-1
        y = 0 if cy>(self.m-1)/2 else self.m-1
        return self.getCoordsFromPixel(x,y)

    #################### VISUALIZATION FUNCTIONS ####################

    def paramString(self) :
        parnames = ['cx',   'cy',   'fx',   'fy',   'k1',   'k2',   'k3',   'p1',   'p2',   'k4',   'k5',   'k6']
        parvals  = [self.cx,self.cy,self.fx,self.fy,self.k1,self.k2,self.k3,self.p1,self.p2,self.k4,self.k5,self.k6]
        s=''
        for n,v in more_itertools.zip_equal(parnames,parvals) :
            if v is not None :
                s+=f'{n}={v:.3f}, '
        return s[:-2]

    def printParams(self, logger=None) :
        """
        Print the current warp parameters in a nice string
        """
        if logger is None: logger = printlogger("warp")
        logger.info(self.paramString())

    def getWarpFields(self) :
        """
        Get the total, dx, and dy warp amount fields
        """
        map_x, map_y = cv2.initUndistortRectifyMap(self.__cam_matrix,self.__dist_pars,None,self.__cam_matrix,
                                                   (self.n,self.m),cv2.CV_32FC1)
        grid = np.mgrid[0:self.m,0:self.n]
        xpos, ypos = grid[1], grid[0]
        x_warps = xpos-map_x
        y_warps = ypos-map_y
        r_warps = (np.sqrt(x_warps**2+y_warps**2)).astype(CONST.OUTPUT_FIELD_DTYPE)
        x_warps = (x_warps).astype(CONST.OUTPUT_FIELD_DTYPE)
        y_warps = (y_warps).astype(CONST.OUTPUT_FIELD_DTYPE)
        return r_warps, x_warps, y_warps

    def writeOutWarpFields(self,file_stem,save_fields=True) :
        """
        Write out .bin files of the dx and dy warping fields and also make an image showing them 
        file_stem   = the unique identifier to add onto the warp field .bin file names
        save_fields = true if the actual fields themselves should be saved in addition to the plots thereof
        """
        r_warps, x_warps, y_warps = self.getWarpFields()
        if save_fields :
            write_image_to_file(x_warps,f'{UNIV_CONST.X_WARP_BIN_FILENAME}_{file_stem}.bin',
                                dtype=CONST.OUTPUT_FIELD_DTYPE)
            write_image_to_file(y_warps,f'{UNIV_CONST.Y_WARP_BIN_FILENAME}_{file_stem}.bin',
                                dtype=CONST.OUTPUT_FIELD_DTYPE)
        f,ax = plt.subplots(1,3,figsize=(3*6.4,(self.m/self.n)*6.4))
        pos = ax[0].imshow(r_warps)
        ax[0].scatter(self.cx,self.cy,marker='*',color='yellow')
        ax[0].set_title('total warp')
        f.colorbar(pos,ax=ax[0])
        pos = ax[1].imshow(x_warps)
        ax[1].scatter(self.cx,self.cy,marker='*',color='yellow')
        ax[1].set_title('dx warp')
        f.colorbar(pos,ax=ax[1])
        pos = ax[2].imshow(y_warps)
        ax[2].scatter(self.cx,self.cy,marker='*',color='yellow')
        ax[2].set_title('dy warp')
        f.colorbar(pos,ax=ax[2])
        fn = f'{CONST.WARP_FIELD_FIGURE_NAME}_{file_stem}.png'
        save_figure_in_dir(plt,fn)

    def showCheckerboard(self) :
        """
        Plot a checkerboard image before and after application of the warp
        """
        self._plotCheckerboards(self.getWarpedLayer(self._checkerboard))

    def makeWarpAmountFigure(self,npoints=50) :
        """
        Plots the radial and tangential warping fields and the curve of the radial warping dependences
        """
        max_x, max_y = self._getMaxDistanceCoords(self.cx,self.cy)
        xvals = np.linspace(0.,max_x,npoints)
        yvals = np.linspace(0.,max_y,npoints)
        max_r = math.sqrt(max_x**2+max_y**2)
        xaxis_points = np.linspace(0.,max_r,npoints)
        yaxis_points = np.array([radialDistortAmountAtCoords(x,y,self.fx,self.fy,self.k1,self.k2,self.k3) for x,y in zip(xvals,yvals)])
        rad_heat_map = np.zeros((self.m,self.n))
        tan_heat_map = np.zeros((self.m,self.n))
        for i in range(self.m) :
            for j in range(self.n) :
                rad_heat_map[i,j] = self.radialDistortAmountAtPixel(j,i)
                tan_heat_map[i,j] = self.tangentialDistortAmountAtPixel(j,i)                
        f,(ax1,ax2,ax3) = plt.subplots(1,3,figsize=(3*6.4,(self.m/self.n)*6.4))
        rhm=sns.heatmap(rad_heat_map,ax=ax1)
        ax1.scatter(self.cx,self.cy,marker='*',color='yellow')
        rhm.set_title('radial warp components',fontsize=14)
        ax2.plot(xaxis_points,yaxis_points)
        ax2.set_xlabel('distance from center (focal lengths)',fontsize=14)
        ax2.set_ylabel('radial warp amount (pixels)',fontsize=14)
        txt = f'warp=({self.k1:.02f}*r^2+{self.k2:.02f}*r^4'
        if self.k3 is not None :
            txt+=f'+{self.k3:.02f}*r^6'
        txt+= f')\n*sqrt(({self.fx:.0f}*x)^2+({self.fy:.0f}*y)^2)'
        ax2.text(xaxis_points[0],0.5*(yaxis_points[-1]-yaxis_points[0]),txt)#,fontsize='14')
        thm=sns.heatmap(tan_heat_map,ax=ax3)
        ax3.scatter(self.cx,self.cy,marker='*',color='yellow')
        thm.set_title('tangential warp components',fontsize=14)
        fn = 'warp_amounts.png'
        save_figure_in_dir(plt,fn)

    #################### PRIVATE HELPER FUNCTIONS ####################

    #helper function to make or update the camera matrix and the vector of distortion parameters
    def __calculateWarpObjects(self) :
        #make the camera matrix
        self.__cam_matrix = np.array([[self.fx,0.,self.cx],[0.,self.fy,self.cy],[0.,0.,1.]])
        #make the vector of distortion parameters
        dist_list = [self.k1,self.k2,self.p1,self.p2]
        if self.k3 is not None:
            dist_list.append(self.k3)
            if self.k4 is not None :
                dist_list.append(self.k4)
                dist_list.append(self.k5)
                dist_list.append(self.k6)
        self.__dist_pars=np.array(dist_list)

    #helper function to convert a raw file name and a layer into a camwarped single layer filename
    def __getWarpedLayerFilename(self,rawname,layer) :
        return (str(((pathlib.Path(rawname).absolute()).parts)[-1])).split(".")[0]+f".camWarp_layer{(layer):02d}"
