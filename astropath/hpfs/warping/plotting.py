#imports
from .warp import PolyFieldWarp, CameraWarp
from .utilities import warp_logger
from .config import CONST
from ...utilities.misc import save_figure_in_dir
import numpy as np, matplotlib.pyplot as plt
from sklearn.decomposition import PCA

#helper function to normalize a list of given raw values by subtracting the mean and dividing by the standard deviations
#returns the list of standardized values, plus the mean and standard deviation
def standardizeValues(rawvals,weights=None,plot=False) :
    if weights is None :
        m = np.mean(rawvals); s = np.std(rawvals)
    else :
        m = 0; sw = 0; sw2 = 0;
        for v,w in zip(rawvals,weights) :
            m+=(w*v); sw+=w; sw2+=(w**2)
        m/=sw
        s = np.sqrt(((np.std(rawvals)**2)*sw2)/(sw**2))
    if plot :
        f,ax = plt.subplots()
        ax.hist(rawvals,bins=20,label='raw values')
        ax.plot([m,m],[0.8*y for y in list(ax.get_ylim())],label=f'mean={m:.2f}')
        ax.plot([m+s,m+s],[0.8*y for y in list(ax.get_ylim())],color='g',label=f'+/- std.dev.={s:.2f}')
        ax.plot([m-s,m-s],[0.8*y for y in list(ax.get_ylim())],color='g',label=f'+/- std.dev.={s:.2f}')
        ax.legend(loc='best')
        plt.show()
    return list([(rv-m)/s for rv in rawvals]), m, s

#makes a plot of the principal points in a list of results; shaded by cost reduction
#also prints the mean and weighted mean
def principalPointPlot(all_results,save_stem=None) :
    mean_cx = np.mean([r.cx for r in all_results if r.cost_reduction>0.]); cx_err = np.std([r.cx for r in all_results if r.cost_reduction>0.])
    mean_cy = np.mean([r.cy for r in all_results if r.cost_reduction>0.]); cy_err = np.std([r.cy for r in all_results if r.cost_reduction>0.])
    weighted_mean_cx = 0.; weighted_mean_cy = 0.; sw = 0.; sw2 = 0.
    for r in all_results :
        w = r.cost_reduction
        if w<0. :
            continue
        weighted_mean_cx+=w*r.cx
        weighted_mean_cy+=w*r.cy
        sw+=w
        sw2+=w**2
    weighted_mean_cx/=sw; w_mean_cx_err = np.sqrt(((cx_err**2)*sw2)/(sw**2))
    weighted_mean_cy/=sw; w_mean_cy_err = np.sqrt(((cy_err**2)*sw2)/(sw**2))
    txt_lines = []
    txt_lines.append(f'Mean center point cx = {mean_cx} +/- {cx_err}')
    txt_lines.append(f'Mean center point cy = {mean_cy} +/- {cy_err}')
    txt_lines.append(f'Weighted mean center point cx = {weighted_mean_cx} +/- {w_mean_cx_err}')
    txt_lines.append(f'Weighted mean center point cy = {weighted_mean_cy} +/- {w_mean_cy_err}')
    if save_stem is not None :
        fn = f'{save_stem}_mean_principal_point.txt'
        with open(fn,'w') as fp :
            for tl in txt_lines :
                fp.write(f'{tl}\n')
    else :
        for tl in txt_lines :
            print(tl)
    f,ax=plt.subplots()
    pos = ax.scatter([r.cx for r in all_results],[r.cy for r in all_results],c=[r.cost_reduction for r in all_results])
    ax.errorbar(mean_cx,mean_cy,yerr=cy_err,xerr=cx_err,marker='x',color='tab:red',label='mean')
    ax.errorbar(weighted_mean_cx,weighted_mean_cy,yerr=w_mean_cy_err,xerr=w_mean_cx_err,marker='x',color='tab:blue',label='weighted mean')
    ax.set_title('Center point locations with cost redux')
    ax.set_xlabel('cx point')
    ax.set_ylabel('cy point')
    ax.legend(loc='best')
    f.colorbar(pos,ax=ax)
    if save_stem is not None :
        fn = f'{save_stem}_principal_point_plot.png'
        save_figure_in_dir(plt,fn)
    else :
        plt.show()

#makes plots of the maximum amounts of radial warping, the cost reduction vs. the amount of max. radial warping, 
#and the principal points locations shaded by max amount of radial warping
def radWarpAmtPlots(all_results,save_stem=None) :
    vs = np.array([r.max_rad_warp for r in all_results])
    weighted_mean = np.sum([r.max_rad_warp*r.cost_reduction for r in all_results if r.cost_reduction>0])/np.sum([r.cost_reduction for r in all_results if r.cost_reduction>0])
    txt_lines = []
    txt_lines.append(f'Mean at {np.mean(vs)}')
    txt_lines.append(f'Weighted mean at {weighted_mean}')
    if save_stem is not None :
        fn = f'{save_stem}_mean_radial_warp_amount.txt'
        with open(fn,'w') as fp :
            for tl in txt_lines :
                fp.write(f'{tl}\n')
    else :
        for tl in txt_lines :
            print(tl)
    f,ax=plt.subplots(1,3,figsize=(3*6.4,4.6))
    ax[0].hist(vs,bins=25,label='all')
    ax[0].plot([np.mean(vs),np.mean(vs)],[0.8*y for y in ax[0].get_ylim()],label='mean')
    ax[0].plot([weighted_mean,weighted_mean],[0.8*y for y in ax[0].get_ylim()],label='weighted mean')
    ax[0].set_title('max. rad. warps')
    ax[0].set_xlabel('max radial warp amount')
    ax[0].set_ylabel('count')
    ax[0].legend(loc='best')
    ax[1].scatter(vs,[r.cost_reduction for r in all_results])
    ax[1].set_title('cost reduction vs. max radial warp')
    ax[1].set_xlabel('max radial warp')
    ax[1].set_ylabel('cost reduction')
    pos = ax[2].scatter([r.cx for r in all_results],[r.cy for r in all_results],c=[r.max_rad_warp for r in all_results])
    mean_cx = np.mean([r.cx for r in all_results if r.cost_reduction>0.]); cx_err = np.std([r.cx for r in all_results if r.cost_reduction>0.])
    mean_cy = np.mean([r.cy for r in all_results if r.cost_reduction>0.]); cy_err = np.std([r.cy for r in all_results if r.cost_reduction>0.])
    weighted_mean_cx = 0.; weighted_mean_cy = 0.; sw = 0.; sw2 = 0.
    for r in all_results :
        w = r.cost_reduction
        if w<=0. :
            continue
        weighted_mean_cx+=w*r.cx
        weighted_mean_cy+=w*r.cy
        sw+=w
        sw2+=w**2
    weighted_mean_cx/=sw; w_mean_cx_err = np.sqrt(((cx_err**2)*sw2)/(sw**2))
    weighted_mean_cy/=sw; w_mean_cy_err = np.sqrt(((cy_err**2)*sw2)/(sw**2))
    ax[2].errorbar(mean_cx,mean_cy,yerr=cy_err,xerr=cx_err,marker='x',color='tab:red',label='mean')
    ax[2].errorbar(weighted_mean_cx,weighted_mean_cy,yerr=w_mean_cy_err,xerr=w_mean_cx_err,marker='x',color='tab:blue',label='weighted mean')
    ax[2].set_title('Center point locations colored by max radial warp')
    ax[2].set_xlabel('cx point')
    ax[2].set_ylabel('cy point')
    ax[2].legend(loc='best')
    f.colorbar(pos,ax=ax[2])
    if save_stem is not None :
        fn = f'{save_stem}_radial_warp_amount_plots.png'
        save_figure_in_dir(plt,fn)
    else :
        plt.show()

#makes plots of the radial warping parameters and the cost reductions vs each
def radWarpParPlots(all_results,save_stem=None) :
    f,ax=plt.subplots()
    pos = ax.scatter([r.k1 for r in all_results],[r.k2 for r in all_results],c=[r.k3 for r in all_results])
    ax.set_title('radial warping parameters (color=k3)')
    ax.set_xlabel('k1')
    ax.set_ylabel('k2')
    f.colorbar(pos,ax=ax)
    if save_stem is not None :
        fn = f'{save_stem}_all_radial_warp_parameters_plot.png'
        save_figure_in_dir(plt,fn)
    else :
        plt.show()
    f,ax=plt.subplots(1,3,figsize=(3*6.4,4.6))
    ax[0].scatter([r.k1 for r in all_results],[r.cost_reduction for r in all_results])
    ax[0].set_title('cost reduction vs. k1')
    ax[0].set_ylabel('cost reduction')
    ax[0].set_xlabel('k1')
    ax[1].scatter([r.k2 for r in all_results],[r.cost_reduction for r in all_results])
    ax[1].set_title('cost reduction vs. k2')
    ax[1].set_ylabel('cost reduction')
    ax[1].set_xlabel('k2')
    ax[2].scatter([r.k3 for r in all_results],[r.cost_reduction for r in all_results])
    ax[2].set_title('cost reduction vs. k3')
    ax[2].set_ylabel('cost reduction')
    ax[2].set_xlabel('k3')
    if save_stem is not None :
        fn = f'{save_stem}_cost_redux_vs_radial_warp_parameters_plots.png'
        save_figure_in_dir(plt,fn)
    else :
        plt.show()

#plots the radial warping parameters in standardized units and the first and second PCA components thereof
def radWarpPCAPlots(all_results,weighted=False,save_stem=None) :
    #get rid of any results with no reduction in cost
    all_results = [r for r in all_results if r.cost_reduction>0]
    #plot the standardized radial warping parameters
    sk1s, k1m, k1std = standardizeValues(np.array([r.k1 for r in all_results]),np.array([r.cost_reduction for r in all_results]) if weighted else None,False)
    sk2s, k2m, k2std = standardizeValues(np.array([r.k2 for r in all_results]),np.array([r.cost_reduction for r in all_results]) if weighted else None,False)
    sk3s, k3m, k3std = standardizeValues(np.array([r.k3 for r in all_results]),np.array([r.cost_reduction for r in all_results]) if weighted else None,False)
    txt_lines = []
    txt_lines.append(f'k1 {"weighted " if weighted else ""}mean = {k1m} ; {"weighted " if weighted else ""}std. dev. = {k1std}')
    txt_lines.append(f'k2 {"weighted " if weighted else ""}mean = {k2m} ; {"weighted " if weighted else ""}std. dev. = {k2std}')
    txt_lines.append(f'k3 {"weighted " if weighted else ""}mean = {k3m} ; {"weighted " if weighted else ""}std. dev. = {k3std}')
    f,ax = plt.subplots()
    pos = ax.scatter(sk1s,sk2s,c=sk3s)
    ax.set_title(f'{"weighted " if weighted else ""}standardized rad. warp parameters')
    ax.set_xlabel(f'{"weighted " if weighted else ""}standardized k1 parameter')
    ax.set_ylabel(f'{"weighted " if weighted else ""}standardized k2 parameter')
    f.colorbar(pos, ax=ax)
    if save_stem is not None :
        fn = f'{save_stem}_all_{"weighted_" if weighted else ""}standardized_radial_warp_parameters_plot.png'
        save_figure_in_dir(plt,fn)
    else :
        plt.show()
    #do the principal component analysis
    standardized_parameters = np.array([sk1s,sk2s,sk3s]).transpose(1,0)
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(standardized_parameters)
    txt_lines.append(f'{"weighted " if weighted else ""}PCA components: {pca.components_}')
    if save_stem is not None :
        fn = f'{save_stem}_radial_warp_{"weighted_" if weighted else ""}PCA_results.txt'
        with open(fn,'w') as fp :
            for tl in txt_lines :
                fp.write(f'{tl}\n')
    else :
        for tl in txt_lines :
            print(tl)
    pc1s = pcs[:,0]
    pc2s = pcs[:,1]
    f,ax = plt.subplots()
    pos = ax.scatter(pc1s,pc2s,c=[r.cost_reduction for r in all_results])
    ax.set_title(f'rad. warp {"weighted " if weighted else ""}PCs (color=cost redux)')
    ax.set_xlabel('first principal component')
    ax.set_ylabel('second principal component')
    f.colorbar(pos,ax=ax)
    if save_stem is not None :
        fn = f'{save_stem}_cost_redux_vs_radial_warp_{"weighted_" if weighted else ""}PCA_components.png'
        save_figure_in_dir(plt,fn)
    else :
        plt.show()

######## Several helper functions to translate lists of warp results into various overall warp fields and their variations ########
def getListsOfWarpFields(all_results) :
    all_results = [r for r in all_results if r.cost_reduction>0.]
    all_warps = []
    for r in all_results :
        all_warps.append(CameraWarp(n=r.n, m=r.m, cx=r.cx, cy=r.cy, fx=r.fx, fy=r.fy, k1=r.k1, k2=r.k2, k3=r.k3, p1=r.p1, p2=r.p2))
    all_drs = []; all_dxs = []; all_dys= []
    for w in all_warps :
        dr, dx, dy = w.getWarpFields()
        all_drs.append(dr); all_dxs.append(dx); all_dys.append(dy)
    return all_drs, all_dxs, all_dys

def getMeanWarpFields(all_results) :
    all_results = [r for r in all_results if r.cost_reduction>0.]
    all_drs, all_dxs, all_dys = getListsOfWarpFields(all_results)
    mean_dr = np.mean(list(all_drs),axis=0)
    mean_dx = np.mean(list(all_dxs),axis=0)
    mean_dy = np.mean(list(all_dys),axis=0)
    return mean_dr, mean_dx, mean_dy

def getWeightedMeanWarpFields(all_results) :
    all_results = [r for r in all_results if r.cost_reduction>0.]
    all_drs, all_dxs, all_dys = getListsOfWarpFields(all_results)
    weighted_mean_dr = np.sum([dr*r.cost_reduction for dr,r in zip(all_drs,all_results)],axis=0)/np.sum([r.cost_reduction for r in all_results])
    weighted_mean_dx = np.sum([dx*r.cost_reduction for dx,r in zip(all_dxs,all_results)],axis=0)/np.sum([r.cost_reduction for r in all_results])
    weighted_mean_dy = np.sum([dy*r.cost_reduction for dy,r in zip(all_dys,all_results)],axis=0)/np.sum([r.cost_reduction for r in all_results])
    return weighted_mean_dr, weighted_mean_dx, weighted_mean_dy

def getWarpFieldStdDevs(all_results) :
    all_results = [r for r in all_results if r.cost_reduction>0.]
    all_drs, all_dxs, all_dys = getListsOfWarpFields(all_results)
    dr_stddev = np.std(list(all_drs),axis=0)
    dx_stddev = np.std(list(all_dxs),axis=0)
    dy_stddev = np.std(list(all_dys),axis=0)
    return dr_stddev, dx_stddev, dy_stddev

def getWarpFieldWeightedStdErr(all_results) :
    all_results = [r for r in all_results if r.cost_reduction>0.]
    all_drs, all_dxs, all_dys = getListsOfWarpFields(all_results)
    dr_stddev = np.std(list(all_drs),axis=0)
    dx_stddev = np.std(list(all_dxs),axis=0)
    dy_stddev = np.std(list(all_dys),axis=0)
    sw = np.sum([r.cost_reduction for r in all_results]); sw2 = np.sum([r.cost_reduction**2 for r in all_results])
    dr_w_stderr = np.sqrt(((dr_stddev**2)*sw2)/(sw**2))
    dx_w_stderr = np.sqrt(((dx_stddev**2)*sw2)/(sw**2))
    dy_w_stderr = np.sqrt(((dy_stddev**2)*sw2)/(sw**2))
    return dr_w_stderr, dx_w_stderr, dy_w_stderr

###################################################################################################################################

#plots the mean, weighted mean, and standard deviation dr, dx, and dy warp fields from a list of WarpFitResults
def warpFieldVariationPlots(all_results,save_stem=None) :
    mean_dr, mean_dx, mean_dy = getMeanWarpFields(all_results)
    f,ax = plt.subplots(4,3,figsize=(3*6.4,4*(mean_dr.shape[0]/mean_dr.shape[1])*6.4))
    pos = ax[0][0].imshow(mean_dr)
    ax[0][0].set_title('mean dr')
    f.colorbar(pos,ax=ax[0][0])
    pos = ax[0][1].imshow(mean_dx)
    ax[0][1].set_title('mean dx')
    f.colorbar(pos,ax=ax[0][1])
    pos = ax[0][2].imshow(mean_dy)
    ax[0][2].set_title('mean dy')
    f.colorbar(pos,ax=ax[0][2])
    dr_stddev, dx_stddev, dy_stddev = getWarpFieldStdDevs(all_results)
    pos = ax[1][0].imshow(dr_stddev)
    ax[1][0].set_title('dr std. dev.')
    f.colorbar(pos,ax=ax[1][0])
    pos = ax[1][1].imshow(dx_stddev)
    ax[1][1].set_title('dx std. dev')
    f.colorbar(pos,ax=ax[1][1])
    pos = ax[1][2].imshow(dy_stddev)
    ax[1][2].set_title('dy std. dev')
    f.colorbar(pos,ax=ax[1][2])
    weighted_mean_dr, weighted_mean_dx, weighted_mean_dy = getWeightedMeanWarpFields(all_results)
    pos = ax[2][0].imshow(weighted_mean_dr)
    ax[2][0].set_title('weighted mean dr')
    f.colorbar(pos,ax=ax[2][0])
    pos = ax[2][1].imshow(weighted_mean_dx)
    ax[2][1].set_title('weighted mean dx')
    f.colorbar(pos,ax=ax[2][1])
    pos = ax[2][2].imshow(weighted_mean_dy)
    ax[2][2].set_title('weighted mean dy')
    f.colorbar(pos,ax=ax[2][2])
    weighted_dr_stderr, weighted_dx_stderr, weighted_dy_stderr = getWarpFieldWeightedStdErr(all_results)
    pos = ax[3][0].imshow(weighted_dr_stderr)
    ax[3][0].set_title('weighted dr std. err.')
    f.colorbar(pos,ax=ax[3][0])
    pos = ax[3][1].imshow(weighted_dx_stderr)
    ax[3][1].set_title('weighted dx std. err.')
    f.colorbar(pos,ax=ax[3][1])
    pos = ax[3][2].imshow(weighted_dy_stderr)
    ax[3][2].set_title('weighted dy std. err.')
    f.colorbar(pos,ax=ax[3][2])
    if save_stem is not None :
        fn = f'{save_stem}_warp_field_variation_plots.png'
        save_figure_in_dir(plt,fn)
    else :
        plt.show()

#plots the total weighted mean dr, dx, and dy fields minus the original polynomial fit warping fields for a list of WarpFitResults
def compareWithAlexWarp(all_results) :
    alex_warp = PolyFieldWarp()
    alex_dr = alex_warp.r_warps
    alex_dx = alex_warp.x_warps
    alex_dy = alex_warp.y_warps
    wm_dr_total, wm_dx_total, wm_dy_total = getWeightedMeanWarpFields(all_results)
    f,ax=plt.subplots(1,3,figsize=(3*6.4,(wm_dr_total.shape[0]/wm_dr_total.shape[1])*6.4))
    pos = ax[0].imshow(wm_dr_total-alex_dr)
    ax[0].set_title("total weighted mean - Alex's warp field dr")
    f.colorbar(pos,ax=ax[0])
    pos = ax[1].imshow(wm_dx_total-alex_dx)
    ax[1].set_title("total weighted mean - Alex's warp field dx")
    f.colorbar(pos,ax=ax[1])
    pos = ax[2].imshow(wm_dy_total-alex_dy)
    ax[2].set_title("total weighted mean - Alex's warp field dy")
    f.colorbar(pos,ax=ax[2])
    plt.show()

###################################################################################################################################

#file-scope dict of the opposite overlap correspondences
OPPOSITE_OVERLAP_TAGS = {1:9,2:8,3:7,4:6,6:4,7:3,8:2,9:1}

#little utility class to help with making the octet overlap comparison images
class OctetComparisonVisualization :

    def __init__(self,overlaps,shifted,name_stem,opposite=False) :
        """
        overlaps  = list of 8 AlignmentOverlap objects to use in building the figure
        shifted   = whether the figure should be built using the shifted overlap images
        name_stem = name to use for the title and filename of the figure
        """
        self.overlaps = overlaps
        self.shifted = shifted
        self.name_stem = name_stem
        self.opposite = opposite
        self.outer_clip = self.overlaps[0].nclip
        self.shift_clip = self.outer_clip+2
        self.normalize = CONST.OVERLAY_NORMALIZE
        if self.opposite :
            self.p1_im = self.overlaps[0].images[1]/self.normalize
        else :
            self.p1_im = self.overlaps[0].images[0]/self.normalize
        self.whole_image = np.zeros((self.p1_im.shape[0],self.p1_im.shape[1],3),dtype=self.p1_im.dtype)
        self.images_stacked_mask = np.zeros(self.whole_image.shape,dtype=np.uint8)
        self.overlay_dicts = {}
        for olap in self.overlaps :
            if self.opposite:
                self.overlay_dicts[OPPOSITE_OVERLAP_TAGS[olap.tag]] = {'image':olap.getimage(self.normalize,self.shifted),
                                                                       'dx':olap.result.dx/2.,'dy':olap.result.dy/2.}
            else :
                self.overlay_dicts[olap.tag] = {'image':olap.getimage(self.normalize,self.shifted),
                                                'dx':-olap.result.dx/2.,'dy':-olap.result.dy/2.}

    def stackOverlays(self) :
        """
        Stack the overlay images into the whole image
        returns a list of tuples of (p1, code) for any overlaps that couldn't be stacked into the whole image
        """
        failed_p1s_codes = []
        #add each overlay to the total image
        for code in self.overlay_dicts.keys() :
            ret = self.__addSingleOverlap(code)
            if ret is not True :
                failed_p1s_codes.append(ret)
        #divide the total image by how many overlays are contributing at each point
        self.whole_image[self.images_stacked_mask!=0]/=self.images_stacked_mask[self.images_stacked_mask!=0]
        #fill in the holes with the p1 image in the appropriate color
        if self.opposite :
            fill_p1 = np.array([np.zeros_like(self.p1_im),self.p1_im,0.5*self.p1_im]).transpose(1,2,0)
        else :
            fill_p1 = np.array([self.p1_im,np.zeros_like(self.p1_im),0.5*self.p1_im]).transpose(1,2,0)
        self.whole_image=np.where(self.whole_image==0,fill_p1,self.whole_image)
        return failed_p1s_codes

    def writeOutFigure(self) :
        """
        Write out a .png of the total octet overlay image
        """ 
        f,ax = plt.subplots(figsize=(CONST.OCTET_OVERLAP_COMPARISON_FIGURE_WIDTH,
                                     np.rint((self.whole_image.shape[0]/self.whole_image.shape[1])*CONST.OCTET_OVERLAP_COMPARISON_FIGURE_WIDTH)))
        ax.imshow(np.clip(self.whole_image,0.,1.))
        ax.set_title(self.name_stem.replace('_',' '))
        savename = f'{self.name_stem}.png'
        save_figure_in_dir(plt,savename)

    #helper function to add a single overlap's set of overlays to the total image
    def __addSingleOverlap(self,code) :
        #figure out the total image x and y start and end points
        tix_1 = 0; tix_2 = 0; tiy_1 = 0; tiy_2 = 0
        #x positions
        if code in [3,6,9] : #left column
            tix_1 = self.outer_clip
            if self.shifted :
                tix_1+=self.shift_clip
            tix_2 = tix_1+self.overlay_dicts[code]['image'].shape[1]
        elif code in [2,8] : #center column
            tix_1 = self.outer_clip
            tix_2 = self.p1_im.shape[1]-self.outer_clip
            if self.shifted :
                tix_1+=self.shift_clip
                tix_2-=self.shift_clip
        elif code in [1,4,7] : #right column
            tix_2 = self.p1_im.shape[1]-self.outer_clip
            if self.shifted :
                tix_2-=self.shift_clip
            tix_1 = tix_2-self.overlay_dicts[code]['image'].shape[1]
        #y positions
        if code in [7,8,9] : #top row
            tiy_1 = self.outer_clip
            if self.shifted :
                tiy_1+=self.shift_clip
            tiy_2 = tiy_1+self.overlay_dicts[code]['image'].shape[0]
        elif code in [4,6] : #center row
            tiy_1 = self.outer_clip
            tiy_2 = self.p1_im.shape[0]-self.outer_clip
            if self.shifted :
                tiy_1+=self.shift_clip
                tiy_2-=self.shift_clip
        elif code in [1,2,3] : #bottom column
            tiy_2 = self.p1_im.shape[0]-self.outer_clip
            if self.shifted :
                tiy_2-=self.shift_clip
            tiy_1 = tiy_2-self.overlay_dicts[code]['image'].shape[0]
        #figure out the alignment adjustment if necessary
        dx = self.overlay_dicts[code]['dx'] if self.shifted else 0
        dy = self.overlay_dicts[code]['dy'] if self.shifted else 0
        tix_1+=dx; tix_2+=dx
        tiy_1+=dy; tiy_2+=dy
        tix_1=int(np.rint(tix_1)); tix_2=int(np.rint(tix_2)); tiy_1=int(np.rint(tiy_1)); tiy_2=int(np.rint(tiy_2))
        #add the overlay to the total image and increment the mask
        try :
            self.whole_image[tiy_1:tiy_2,tix_1:tix_2,:]+=self.overlay_dicts[code]['image']
            self.images_stacked_mask[tiy_1:tiy_2,tix_1:tix_2,:]+=1
            return True
        except Exception as e :
            fp1 = self.overlaps[0].p1
            msg=f'WARNING: overlap with p1={fp1} and code {code} could not be stacked into octet overlay comparison'
            msg+=f' and will be plotted separately. Exception: {e}'
            warp_logger.warn(msg)
            return tuple((fp1,code))
