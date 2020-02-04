#imports
import numpy as np
from plotting import *
import cv2

class WarpingError(Exception) :
    """
    Class for errors encountered during warping
    """
    pass

#Alex's function to fit a polynomial to the warping given a max amount of warp
def polyFit(max_warp,deg,squared=False,plot=False) :
    #define warping as a function of distance
    r_points = np.array([0.0,0.2,0.4,0.8,1.4])
    if squared :
        r_points = r_points**2
    warp_amt = np.array([0.0,0.0,0.0,0.2,max_warp])
    #fit polynomial (*****may want to do this with scipy.optimize instead*****)
    coeffs = np.polyfit(r_points,warp_amt,deg)
    #make plot if requested
    if plot :
        plotPolyFit(r_points,warp_amt,coeffs,squared)
    #return coefficients
    return coeffs

#Alex's function to define a warp field
def makeWarp(n=1344,m=1004,xc=584,yc=600,max_warp=1.85,pdegree=3,psq=False,plot_fit=False,plot_warpfields=False) :
    #define distance fields
    grid = np.mgrid[1:m+1,1:n+1]
    x=(grid[1]-xc)/500. #scaled x displacement from center
    y=(grid[0]-yc)/500. #scaled y displacement from center
    r=np.sqrt(x**2+y**2)                                      #scaled total distance from center
    #fit polynomial to data
    coeffs = polyFit(max_warp,pdegree,psq,plot=plot_fit)
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
    #return double layer field of x and y shifts
    return d_warps

#helper function to read the binary dump of a raw im3 file 
def im3readraw(f) :
    with open(f,mode='rb') as fp : #read as binary
        content = np.fromfile(fp,dtype=np.uint16)
    return content

#helper function to write an array of uint16s as an im3 file
def im3writeraw(outname,a) :
    with open(outname,mode='wb') as fp : #write as binary
        a.tofile(fp)

#function to apply a warp field to one image and write it as a new file
def warpImageWithField(warp,infname,file_ext=r".raw",nlayers=35,interpolation=cv2.INTER_LINEAR,showimgs=False) :
    #get the .raw file as a vector of uint16s
    img = im3readraw(infname)
    #reshape it to match the warping field
    try :
        img_a = np.reshape(img,(nlayers,)+warp.shape[:2][::-1],order="F") #dim. rev. due to MATLAB/python order mismatch
    except ValueError :
        msg = f"ERROR: Raw image file shape ({nlayers} layers, {len(img)} total bytes) is mismatched to"
        msg+= f" warp field dimensions (shape={warp.shape})!"
        raise WarpingError(msg)
    #flip x and y dimensions to match warp field, move layers to z-axis
    img_to_warp = np.transpose(img_a,(2,1,0))
    #use the warp matrix to calculate the map matrices for remap
    m,n = warp.shape[:2]
    grid = np.mgrid[1:m+1,1:n+1]
    xpos, ypos = grid[1], grid[0]
    map_x = (xpos-warp[:,:,0]).astype(np.float32); map_y = (ypos-warp[:,:,1]).astype(np.float32) #maybe use maps from convertMaps() instead later on?
    #get the remapped image
    img_warped = np.zeros(img_to_warp.shape)
    for i in range(img_to_warp.shape[-1]) :
        img_warped[:,:,i] = cv2.remap(img_to_warp[:,:,i],map_x,map_y,interpolation)
    if showimgs :
        plotWarpingComparison(infname.rstrip(file_ext),img_to_warp,img_warped,layers=[1,2,3])
    #transpose the resulting image back to its original dimensions
    img_transposed = np.transpose(img_warped,(2,1,0))
    #write out image flattened in fortran order
    outfname = infname.replace(file_ext,'.warpTEST')
    im3writeraw(outfname,img_transposed.flatten(order="F").astype(np.uint16))