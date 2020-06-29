#imports
from ..utilities.img_file_io import getRawAsHWL, getRawAsHW, writeImageToFile
import numpy as np
import os, math, cv2
import matplotlib.pyplot as plt, seaborn as sns

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
        self._checkerboard = self.__makeCheckerboard()

    #################### COMMON FUNCTIONS ####################

    def getHWLFromRaw(self,fname,nlayers=35) :
        """
        Function to return a '.Data.dat' binary file as an array of dimensions (height,width,nlayers) or (m,n,nlayers)
        """
        return getRawAsHWL(fname,self.m,self.n,nlayers)

    def getSingleLayerImage(self,fname) :
        """
        Function to read a file that contains one layer of an image into an array with the Warp's dimensions 
        """
        return getRawAsHW(fname,self.m,self.n)

    def writeSingleLayerImage(self,im,outfname) :
        """
        Function to write out an image as a properly transformed and flattened vector of uints 
        """
        writeImageToFile(im,outfname)

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
        self.xc=xc
        self.yc=yc
        self.r_warps, self.x_warps, self.y_warps = self.__getWarpFields(xc,yc,max_warp,pdegree,psq,plot_fit)
        self.interp=interpolation
        #plot warp fields if requested
        if plot_warpfields : self.plotWarpFields()

    #################### COMMON FUNCTIONS ####################

    def warpAndWriteImage(self,infname,nlayers=35,layers=[1]) :
        """
        Read in an image, warp layer-by-layer with remap, and save each warped layer as its own new file in the current directory
        infname       = name of image file to split, warp, and re-save
        nlayers       = number of layers in original image file that's opened
        layers        = list (of integers) of layers to split out, warp, and save (index starting from 1)
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
        Quickly warps and returns a single inputted image layer array
        """
        map_x, map_y = self.__getMapMatrices()
        return cv2.remap(layer,map_x,map_y,self.interp)

    def warpLayerInPlace(self,layerimg,dest) :
        """
        Quickly warps a single inputted image layer into the provided destination
        """
        map_x, map_y = self.__getMapMatrices()
        return cv2.remap(layerimg,map_x,map_y,self.interp,dest)

    def writeImageLayer(self,im,rawfilename,layernumber) :
        """
        Write out a single given image layer
        """
        self.writeSingleLayerImage(im,self.__getWarpedLayerFilename(rawfilename,layernumber))

    #################### VISUALIZATION FUNCTIONS ####################

    def plotWarpFields(self) :
        """
        Plot three heatmaps of the r-, x-, and y-dependent warping fields
        """
        f,(ax1,ax2,ax3) = plt.subplots(1,3)
        f.set_size_inches(20.,5.)
        #plot radial field as a heatmap
        g1 = sns.heatmap(self.r_warps,ax=ax1)
        ax1.scatter(self.xc,self.yc,marker='*',color='yellow')
        g1.set_title('radially-dependent warping shifts')
        #plot x and y shifts
        g2 = sns.heatmap(self.x_warps,ax=ax2)
        ax2.scatter(self.xc,self.yc,marker='*',color='yellow')
        g2.set_title('warping shift x components')
        g3 = sns.heatmap(self.y_warps,ax=ax3)
        ax3.scatter(self.xc,self.yc,marker='*',color='yellow')
        g3.set_title('warping shift y components')
        plt.show()

    def showCheckerboard(self) :
        """
        Plot a checkerboard image before and after application of the warp
        """
        self._plotCheckerboards(self.getWarpedLayer(self._checkerboard))

    #################### PRIVATE HELPER FUNCTIONS ####################
    
    #helper function to make and return r_warps (field of warp factors) and x/y_warps (two fields of warp gradient dx/dy)
    def __getWarpFields(self,xc,yc,max_warp,pdegree,psq,plot_fit) :
        #define distance fields
        grid = np.mgrid[1:self.m+1,1:self.n+1]
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
        return r_warps, r_warps*x, r_warps*y

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

    #helper function to calculate and return the map matrices for remap
    def __getMapMatrices(self) :
        grid = np.mgrid[1:self.m+1,1:self.n+1]
        xpos, ypos = grid[1], grid[0]
        map_x = (xpos-self.x_warps).astype(np.float32) 
        map_y = (ypos-self.y_warps).astype(np.float32) #maybe use maps from convertMaps() instead later on?
        return map_x, map_y

    #helper function to convert a raw file name and a layer into a fieldwarped single layer filename
    def __getWarpedLayerFilename(self,rawname,layer) :
        return (rawname.split(os.path.sep)[-1]).split(".")[0]+f".fieldWarp_layer{(layer):02d}"

class CameraWarp(Warp) :
    """
    Subclass for applying warping to images based on a camera matrix and distortion parameters
    """
    def __init__(self,n=1344,m=1004,cx=None,cy=None,fx=40000.,fy=40000.,k1=0.,k2=0.,p1=0.,p2=0.,k3=0.,k4=None,k5=None,k6=None) :
        """
        Initialize a camera matrix and vector of distortion parameters for a camera warp transformation
        See https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html for explanations of parameters/functions
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

    #################### COMMON FUNCTIONS ####################

    def warpAndWriteImage(self,infname,nlayers=35,layers=[1]) :
        """
        Read in an image, warp layer-by-layer with undistort, and save each warped layer as its own new file in the current directory
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
        return self._radialDistortAmountAtCoords(x,y)

    def tangentialDistortAmountAtPixel(self,pixel_x,pixel_y) :
        """
        Return the amount of tangential warp (in pixels) at the given pixel
        """
        x, y = self.getCoordsFromPixel(pixel_x,pixel_y)
        return self._tangentialDistortAmountAtCoords(x,y)

    #################### FUNCTIONS FOR USE WITH MINIMIZATION ####################

    # !!!!!! For the time being, these functions don't correctly describe dependence on k4, k5, or k6 !!!!!!

    def updateParams(self,pars) :
        """
        Update the camera matrix and distortion parameters for a new transformation on the same images
        pars = list of transformation parameters in order cx, cy, fx, fy, [dist_vec] (len 4-8, depending)
        """
        self.cx=pars[0]; self.cy=pars[1]
        self.fx=pars[2]; self.fy=pars[3]
        self.k1=pars[4]; self.k2=pars[5]
        self.p1=pars[6]; self.p2=pars[7]
        if len(pars)>8 :
            self.k3=pars[8]
            if len(pars)>9 :
                self.k4=pars[9]; self.k5=pars[10]; self.k6=pars[11]
        self.__calculateWarpObjects()

    def maxRadialDistortAmount(self,pars) :
        """
        Return the maximum amount of radial distortion (in pixels) observed with the given parameters
        """
        x, y = self._getMaxDistanceCoords(pars)
        return self._radialDistortAmountAtCoords(x,y,pars)

    def maxTangentialDistortAmount(self,pars) :
        """
        Return the maximum amount of tangential distortion (in pixels) observed with the given parameters
        """
        x, y = self._getMaxDistanceCoords(pars)
        return self._tangentialDistortAmountAtCoords(x,y,pars)

    def maxRadialDistortAmountJacobian(self,pars) :
        """
        Return the Jacobian vector of the maxRadialDistortAmount function (used in minimization)
        """
        x, y = self._getMaxDistanceCoords(pars)
        fxfyk1k2k3_dependence = self._radialDistortAmountAtCoordsJacobian(x,y,pars)
        retvec =[0.,0.] # no dependence on cx/cy
        retvec+=fxfyk1k2k3_dependence[:-1] #add fx, fy, k1, and k2 dependency
        retvec+=[0.,0.] # no dependence on p1/p2
        retvec+=[fxfyk1k2k3_dependence[-1]] #add k3 dependence
        return retvec 

    def maxTangentialDistortAmountJacobian(self,pars) :
        """
        Return the Jacobian vector of the maxTangentialDistortAmount function (used in minimization)
        """
        x, y = self._getMaxDistanceCoords(pars)
        fxfyp1p2_dependence = self._tangentialDistortAmountAtCoordsJacobian(x,y,pars)
        retvec =[0.,0.] # no dependence on cx/cy
        retvec+=fxfyp1p2_dependence[:2]
        retvec+=[0.,0.] # no dependence on k1/k2
        retvec+=fxfyp1p2_dependence[2:]
        retvec+=[0.] # no dependence on k3
        return retvec

    def _getMaxDistanceCoords(self,pars=None) :
        """
        Get the x/y coordinate-space location of the image corner that is furthest from the principal point
        pars = parameter list to use for this function evaluation only (if None, use current warp parameters)
        """
        cx,cy,_,_,_,_,_,_,_ = self.__getEvalPars(pars)
        x = 0 if cx>(self.n-1)/2 else self.n-1
        y = 0 if cy>(self.m-1)/2 else self.m-1
        return self.getCoordsFromPixel(x,y)

    def _radialDistortAmountAtCoords(self,coord_x,coord_y,pars=None) :
        """
        Return the amount of radial warp (in pixels) at the given coordinate-space location
        pars = parameter list to use for this function evaluation only (if None, use current warp parameters)
        """
        _,_,fx,fy,k1,k2,_,_,k3 = self.__getEvalPars(pars)
        r = math.sqrt(coord_x**2+coord_y**2)
        return (k1*(r**2) + k2*(r**4) + k3*(r**6))*math.sqrt((fx*coord_x)**2 + (fy*coord_y)**2)

    def _tangentialDistortAmountAtCoords(self,coord_x,coord_y,pars=None) :
        """
        Return the amount of tangential warp (in pixels) at the given coordinate-space location
        pars = parameter list to use for this function evaluation only (if None, use current warp parameters)
        """
        _,_,fx,fy,_,_,p1,p2,_ = self.__getEvalPars(pars)
        r = math.sqrt(coord_x**2+coord_y**2)
        dx = 2.*fx*p1*coord_x*coord_y + 2.*fx*p2*(coord_x**2) + fx*p2*(r**2)
        dy = 2.*fy*p2*coord_x*coord_y + 2.*fy*p1*(coord_y**2) + fy*p1*(r**2)
        return math.sqrt((dx)**2 + (dy)**2)

    def _radialDistortAmountAtCoordsJacobian(self,coord_x,coord_y,pars=None) :
        """
        Return the Jacobian vector of the _radialDistortAmountAtCoords function (used in minimization)
        pars = parameter list to use for this function evaluation only (if None, use current warp parameters)
        """
        _,_,fx,fy,k1,k2,_,_,k3 = self.__getEvalPars(pars)
        r = math.sqrt(coord_x**2+coord_y**2)
        A = math.sqrt((fx*coord_x)**2 + (fy*coord_y)**2)
        B = k1*(r**2) + k2*(r**4) + k3*(r**6)
        dfdfx = (B/A)*fx*(coord_x**2)
        dfdfy = (B/A)*fy*(coord_y**2)
        dfdk1 = A*(r**2)
        dfdk2 = A*(r**4)
        dfdk3 = A*(r**6)
        return [dfdfx,dfdfy,dfdk1,dfdk2,dfdk3]

    def _tangentialDistortAmountAtCoordsJacobian(self,coord_x,coord_y,pars=None) :
        """
        Return the Jacobian of the _tangentialDistortAmountAtCoords function (used in minimization)
        pars = parameter list to use for this function evaluation only (if None, use current warp parameters)
        """
        _,_,fx,fy,_,_,p1,p2,_ = self.__getEvalPars(pars)
        r = math.sqrt(coord_x**2+coord_y**2)
        dx = 2.*fx*p1*coord_x*coord_y + 2.*fx*p2*(coord_x**2) + fx*p2*(r**2)
        dy = 2.*fy*p2*coord_x*coord_y + 2.*fy*p1*(coord_y**2) + fy*p1*(r**2)
        dfdfx = (dx/fx)/(math.sqrt(1+(dy/dx)**2))
        dfdfy = (dy/fy)/(math.sqrt(1+(dx/dy)**2))
        dfdp1 = (2*dx*fx*coord_x*coord_y + 2*dy*fy*(coord_y**2) + dy*fy*(r**2))/(math.sqrt(dx**2+dy**2))
        dfdp2 = (2*dy*fy*coord_x*coord_y + 2*dx*fx*(coord_x**2) + dx*fx*(r**2))/(math.sqrt(dx**2+dy**2))
        return [dfdfx,dfdfy,dfdp1,dfdp2]

    #################### VISUALIZATION FUNCTIONS ####################

    def paramString(self) :
        parnames=['cx','cy','fx','fy','k1','k2','k3','p1','p2','k4','k5','k6']
        parvals=[self.cx,self.cy,self.fx,self.fy,self.k1,self.k2,self.k3,self.p1,self.p2,self.k4,self.k5,self.k6]
        s=''
        for n,v in zip(parnames,parvals) :
            if v is not None :
                s+=f'{n}={v:.3f}, '
        return s[:-2]

    def printParams(self) :
        """
        Print the current warp parameters in a nice string
        """
        print(self.paramString())

    def showCheckerboard(self) :
        """
        Plot a checkerboard image before and after application of the warp
        """
        self._plotCheckerboards(self.getWarpedLayer(self._checkerboard))

    def makeWarpAmountFigure(self,show=False,npoints=50) :
        max_x, max_y = self._getMaxDistanceCoords()
        xvals = np.linspace(0.,max_x,npoints)
        yvals = np.linspace(0.,max_y,npoints)
        max_r = math.sqrt(max_x**2+max_y**2)
        xaxis_points = np.linspace(0.,max_r,npoints)
        yaxis_points = np.array([self._radialDistortAmountAtCoords(x,y) for x,y in zip(xvals,yvals)])
        rad_heat_map = np.zeros((self.m,self.n))
        tan_heat_map = np.zeros((self.m,self.n))
        for i in range(self.m) :
            for j in range(self.n) :
                rad_heat_map[i,j] = self.radialDistortAmountAtPixel(j,i)
                tan_heat_map[i,j] = self.tangentialDistortAmountAtPixel(j,i)                
        f,(ax1,ax2,ax3) = plt.subplots(1,3)
        f.set_size_inches(20.,5.)
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
        plt.savefig('warp_amounts.png')
        if show :
            plt.show()
        plt.close()

    def writeParameterTextFile(self,par_mask=None,init_its=None,polish_its=None,init_min_runtime=None,polish_min_runtime=None,rawcost=None,bestcost=None) :
        fn = 'warping_parameters.txt'
        max_r_x, max_r_y = self._getMaxDistanceCoords()
        pars=[self.cx,self.cy,self.fx,self.fy,self.k1,self.k2,self.p1,self.p2,self.k3]
        if par_mask==None :
            par_mask=[True for p in pars]
        to_write = {
            'n':str(self.n),
            'm':str(self.m),
            'cx':str(self.cx)+('' if par_mask[0] else ' (fixed)'),
            'cy':str(self.cy)+('' if par_mask[1] else ' (fixed)'),
            'fx':str(self.fx)+('' if par_mask[2] else ' (fixed)'),
            'fy':str(self.fy)+('' if par_mask[3] else ' (fixed)'),
            'k1':str(self.k1)+('' if par_mask[4] else ' (fixed)'),
            'k2':str(self.k2)+('' if par_mask[5] else ' (fixed)'),
            'k3':str(self.k3)+('' if par_mask[8] else ' (fixed)'),
            'p1':str(self.p1)+('' if par_mask[6] else ' (fixed)'),
            'p2':str(self.p2)+('' if par_mask[7] else ' (fixed)'),
            'max_r_x_coord':str(max_r_x),
            'max_r_y_coord':str(max_r_y),
            'max_r':str(math.sqrt((max_r_x)**2+(max_r_y)**2)),
            'max_radial_warp':str(self.maxRadialDistortAmount(pars)),
            'max_tangential_warp':str(self.maxTangentialDistortAmount(pars)),
        }
        if init_its is not None : to_write['initial_fit_iterations'] = str(init_its)
        if polish_its is not None : to_write['polishing_fit_iterations'] = str(polish_its)
        if init_min_runtime is not None : to_write['initial_minimization_time'] = str(init_min_runtime)
        if polish_min_runtime is not None : to_write['polish_minimization_time'] = str(polish_min_runtime)
        if rawcost is not None : to_write['raw_cost'] = str(rawcost)
        if bestcost is not None : to_write['best_cost'] = str(bestcost)
        if rawcost is not None and bestcost is not None : to_write['cost_reduction'] = f'{(100*(1.-bestcost/rawcost))}%'
        max_key_width = 0; max_value_width = 0
        for k,v in to_write.items() :
            if len(k)>max_key_width :
                max_key_width=len(k)
            if len(v)>max_value_width :
                max_value_width=len(v)
        with open(fn,'w') as fp :
            for k,v in to_write.items() :
                fp.write(f'{k:<{max_key_width+3}}{v:<{max_value_width}}\n')

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
        return (rawname.split(os.path.sep)[-1]).split(".")[0]+f".camWarp_layer{(layer):02d}"

    #helper function to return a tuple of parameters for single function evaluations
    def __getEvalPars(self,pars) :
        if pars is None :
            return self.cx, self.cy, self.fx, self.fy, self.k1, self.k2, self.p1, self.p2, self.k3
        else :
            return (*pars, )
