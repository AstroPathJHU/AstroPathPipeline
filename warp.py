#imports
import numpy as np
import os, math, cv2
import matplotlib.pyplot as plt, seaborn as sns

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
        self._checkerboard = self.__makeCheckerboard()

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

    def warpImage(self,infname,nlayers=35,layers=[1]) :
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
            self.warpLayer(img_to_warp[:,:,i],i,infname)

    def warpLayer(self,layer,layernumber,rawfilename) :
        """
        Quickly warp a single inputted image layer array with the current parameters and save it
        """
        self.writeSingleLayerImage(self.getWarpedLayer(layer),self.__getWarpedLayerFilename(rawfilename,layernumber))

    def getWarpedLayer(self,layer) :
        """
        Quickly warps and returns a single inputted image layer array
        """
        map_x, map_y = self.__getMapMatrices()
        return cv2.remap(layer,map_x,map_y,self.interp)

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
        plt.plot(fit_xs,fit_ys,label=f"Fit")
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
    def __init__(self,n=1344,m=1004,cx=672.5,cy=502.,fx=40000.,fy=40000.,k1=0.,k2=0.,p1=0.,p2=0.,k3=None,k4=None,k5=None,k6=None) :
        """
        Initialize a camera matrix and vector of distortion parameters for a camera warp transformation
        See https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html for explanations of parameters/functions
        Parameters k3, k4, k5, and k6 are all optional, but if k4 is defined k5 and k6 must be as well.
        cx         = principal center point x coordinate
        cy         = principal center point y coordinate
        fx         = x focal length (pixels)
        fy         = y focal length (pixels)
        k1, k2, k3 = radial distortion by third-order polynomial in r^2 (numerator)
        k4, k5, k6 = radial distortion by third-order polynomial in r^2 (denominator)
        p1, p2     = tangential distortion parameters
        """
        super().__init__(n,m)
        self.cx=cx; self.cy=cy; self.fx=fx; self.fy=fy
        self.k1=k1; self.k2=k2
        self.p1=p1; self.p2=p2
        self.k3=k3
        self.k4=k4; self.k5=k5; self.k6=k6
        self.__cam_matrix = None 
        self.__dist_pars  = None
        self.__calculateWarpObjects()

    def warpImage(self,infname,nlayers=35,layers=[1]) :
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
            self.warpLayer(img_to_warp[:,:,i],i,infname)

    def warpLayer(self,layer,layernumber,rawfilename) :
        """
        Quickly warps a single inputted image layer array with the current parameters and save it
        """
        self.writeSingleLayerImage(self.getWarpedLayer(layer),self.__getWarpedLayerFilename(rawfilename,layernumber))

    def getWarpedLayer(self,layerimg) :
        """
        Quickly warps and returns a single inputted image layer array
        """
        #return the result of undistort
        return cv2.undistort(layerimg,self.__cam_matrix,self.__dist_pars)

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

    def getCoordsFromPixel(self,pixel_x,pixel_y) :
        """
        Convert a pixel to an x/y coordinate with units of x/y focal lengths
        """
        return (pixel_x-self.cx)/self.fx, (pixel_y-self.cy)/self.fy

    def getMaxDistanceCoords(self) :
        """
        Get the x/y coordinate-space location of the image corner that is furthest from the principal point
        """
        corner_point_xy_pixels = [(0,0),(self.n-1,0),(0,self.m-1),(self.n-1,self.m-1)]
        corner_point_xy_coords = [self.getCoordsFromPixel(x,y) for x,y in corner_point_xy_pixels]
        distances = [math.sqrt(x**2+y**2) for x,y in corner_point_xy_coords]
        r = max(distances)
        return corner_point_xy_coords[distances.index(r)]

    def radDistortAmountAtCoords(self,coord_x,coord_y) :
        """
        Return the amount of radial warp (in pixels) at the given coordinate-space location
        """
        r = math.sqrt(coord_x**2+coord_y**2)
        return (self.k1*(r**2) + self.k2*(r**4))*math.sqrt((self.fx*coord_x)**2 + (self.fy*coord_y)**2)

    def radDistortAmountAtPixel(self,pixel_x,pixel_y) :
        """
        Return the amount of radial warp (in pixels) at the given pixel
        """
        x, y = self.getCoordsFromPixel(pixel_x,pixel_y)
        return self.radDistortAmountAtCoords(x,y)

    def printParams(self) :
        """
        Print the current warp parameters in a nice string
        """
        parnames=['cx','cy','fx','fy','k1','k2','p1','p2','k3','k4','k5','k6']
        parvals=[self.cx,self.cy,self.fx,self.fy,self.k1,self.k2,self.p1,self.p2,self.k3,self.k4,self.k5,self.k6]
        s=''
        for n,v in zip(parnames,parvals) :
            if v is not None :
                s+=f'{n}={v:.3f}, '
        print(s[:-2])

    def showCheckerboard(self) :
        """
        Plot a checkerboard image before and after application of the warp
        """
        self._plotCheckerboards(self.getWarpedLayer(self._checkerboard))

    def plotRadialWarpAmount(self,npoints=50) :
        max_x, max_y = self.getMaxDistanceCoords()
        xvals = np.linspace(0.,max_x,npoints)
        yvals = np.linspace(0.,max_y,npoints)
        max_r = math.sqrt(max_x**2+max_y**2)
        xaxis_points = np.linspace(0.,max_r,npoints)
        yaxis_points = np.array([self.radDistortAmountAtCoords(x,y) for x,y in zip(xvals,yvals)])
        heat_map = np.zeros((self.m,self.n))
        for i in range(heat_map.shape[0]) :
            for j in range(heat_map.shape[1]) :
                heat_map[i,j] = self.radDistortAmountAtPixel(j,i)
        f,(ax1,ax2,ax3) = plt.subplots(1,3)
        f.set_size_inches(20.,5.)
        hm=sns.heatmap(heat_map,ax=ax1)
        ax1.scatter(self.cx,self.cy,marker='*',color='yellow')
        hm.set_title('radial warp components',fontsize=14)
        ax2.plot(xaxis_points,yaxis_points)
        ax2.set_xlabel('distance from center (focal lengths)',fontsize=14)
        ax2.set_ylabel('radial warp amount (pixels)',fontsize=14)
        txt = f'warp=({self.k1:.04f}*r^2+{self.k2:.04f}*r^4)\n'
        txt+= f'*sqrt(({self.fx:.0f}*x)^2+({self.fy:.0f}*y)^2)'
        ax2.text(xaxis_points[0],0.5*(yaxis_points[-1]-yaxis_points[0]),txt,fontsize='14')
        plt.show()

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

#helper function to read the binary dump of a raw im3 file 
def im3readraw(f) :
    with open(f,mode='rb') as fp : #read as binary
        content = np.fromfile(fp,dtype=np.uint16)
    return content

#helper function to write an array of uint16s as an im3 file
def im3writeraw(outname,a) :
    with open(outname,mode='wb') as fp : #write as binary
        a.tofile(fp)
