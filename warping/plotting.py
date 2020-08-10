#imports
from .warp import PolyFieldWarp, CameraWarp
import numpy as np, matplotlib.pyplot as plt
from sklearn.decomposition import PCA

#helper function to normalize a list of given raw values by subtracting the mean and dividing by the standard deviations
#returns the list of standardized values, plus the mean and standard deviation
def standardizeValues(rawvals,plot=False) :
    m = np.mean(rawvals); s = np.std(rawvals)
    print(f'mean = {m}, std. dev. = {s}')
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
def principalPointPlot(all_results) :
    mean_cx = np.mean([r.cx for r in all_results])
    mean_cy = np.mean([r.cy for r in all_results])
    print(f'Mean center point at ({mean_cx}, {mean_cy})')
    weighted_mean_cx = np.sum([r.cx*r.cost_reduction for r in all_results])/np.sum([r.cost_reduction for r in all_results])
    weighted_mean_cy = np.sum([r.cy*r.cost_reduction for r in all_results])/np.sum([r.cost_reduction for r in all_results])
    print(f'Weighted mean center point at ({weighted_mean_cx}, {weighted_mean_cy})')
    f,ax=plt.subplots()
    pos = ax.scatter([r.cx for r in all_results],[r.cy for r in all_results],c=[r.cost_reduction for r in all_results])
    ax.scatter(mean_cx,mean_cy,marker='x',color='tab:red',label='mean')
    ax.scatter(weighted_mean_cx,weighted_mean_cy,marker='x',color='tab:blue',label='weighted mean')
    ax.set_title('Center point locations with cost redux')
    ax.set_xlabel('cx point')
    ax.set_ylabel('cy point')
    ax.legend(loc='best')
    f.colorbar(pos,ax=ax)
    plt.show()

#makes plots of the maximum amounts of radial warping, the cost reduction vs. the amount of max. radial warping, 
#and the principal points locations shaded by max amount of radial warping
def radWarpAmtPlots(all_results) :
    vs = np.array([r.max_rad_warp for r in all_results])
    print(f'Mean at {np.mean(vs)}')
    weighted_mean = np.sum([r.max_rad_warp*r.cost_reduction for r in all_results])/np.sum([r.cost_reduction for r in all_results])
    print(f'Weighted mean at {weighted_mean}')
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
    mean_cx = np.mean([r.cx for r in all_results])
    mean_cy = np.mean([r.cy for r in all_results])
    weighted_mean_cx = np.sum([r.cx*r.cost_reduction for r in all_results])/np.sum([r.cost_reduction for r in all_results])
    weighted_mean_cy = np.sum([r.cy*r.cost_reduction for r in all_results])/np.sum([r.cost_reduction for r in all_results])
    ax[2].scatter(mean_cx,mean_cy,marker='x',color='tab:red',label='mean')
    ax[2].scatter(weighted_mean_cx,weighted_mean_cy,marker='x',color='tab:blue',label='weighted mean')
    ax[2].set_title('Center point locations colored by max radial warp')
    ax[2].set_xlabel('cx point')
    ax[2].set_ylabel('cy point')
    ax[2].legend(loc='best')
    f.colorbar(pos,ax=ax[2])
    plt.show()

#makes plots of the radial warping parameters and the cost reductions vs each
def radWarpParPlots(all_results) :
    f,ax=plt.subplots()
    pos = ax.scatter([r.k1 for r in all_results],[r.k2 for r in all_results],c=[r.k3 for r in all_results])
    ax.set_title('radial warping parameters (color=k3)')
    ax.set_xlabel('k1')
    ax.set_ylabel('k2')
    f.colorbar(pos,ax=ax)
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
    plt.show()

#plots the radial warping parameters in standardized units and the first and second PCA components thereof
def radWarpPCAPlots(all_results) :
    #plot the standardized radial warping parameters
    sk1s, k1m, k1std = standardizeValues(np.array([r.k1 for r in all_results]),False)
    sk2s, k2m, k2std = standardizeValues(np.array([r.k2 for r in all_results]),False)
    sk3s, k3m, k3std = standardizeValues(np.array([r.k3 for r in all_results]),False)
    f,ax = plt.subplots()
    pos = ax.scatter(sk1s,sk2s,c=sk3s)
    ax.set_title('standardized rad. warp parameters')
    ax.set_xlabel('standardized k1 parameter')
    ax.set_ylabel('standardized k2 parameter')
    f.colorbar(pos, ax=ax)
    plt.show()
    #do the principal component analysis
    standardized_parameters = np.array([sk1s,sk2s,sk3s]).transpose(1,0)
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(standardized_parameters)
    print(f'PCA components: {pca.components_}')
    pc1s = pcs[:,0]
    pc2s = pcs[:,1]
    f,ax = plt.subplots()
    pos = ax.scatter(pc1s,pc2s,c=[r.cost_reduction for r in all_results])
    ax.set_title('rad. warp PCs (color=cost redux)')
    ax.set_xlabel('first principal component')
    ax.set_ylabel('second principal component')
    f.colorbar(pos,ax=ax)
    plt.show()

######## Several helper functions to translate lists of warp results into various overall warp fields and their variations ########
def getListsOfWarpFields(all_results) :
    all_warps = []
    for r in all_results :
        all_warps.append(CameraWarp(cx=r.cx, cy=r.cy, k1=r.k1, k2=r.k2, k3=r.k3))
    all_drs = []; all_dxs = []; all_dys= []
    for w in all_warps :
        dr, dx, dy = w.getWarpFields()
        all_drs.append(dr); all_dxs.append(dx); all_dys.append(dy)
    return all_drs, all_dxs, all_dys

def getMeanWarpFields(all_results) :
    all_drs, all_dxs, all_dys = getListsOfWarpFields(all_results)
    mean_dr = np.mean(list(all_drs),axis=0)
    mean_dx = np.mean(list(all_dxs),axis=0)
    mean_dy = np.mean(list(all_dys),axis=0)
    return mean_dr, mean_dx, mean_dy

def getWeightedMeanWarpFields(all_results) :
    all_drs, all_dxs, all_dys = getListsOfWarpFields(all_results)
    weighted_mean_dr = np.sum([dr*r.cost_reduction for dr,r in zip(all_drs,all_results)],axis=0)/np.sum([r.cost_reduction for r in all_results])
    weighted_mean_dx = np.sum([dx*r.cost_reduction for dx,r in zip(all_dxs,all_results)],axis=0)/np.sum([r.cost_reduction for r in all_results])
    weighted_mean_dy = np.sum([dy*r.cost_reduction for dy,r in zip(all_dys,all_results)],axis=0)/np.sum([r.cost_reduction for r in all_results])
    return weighted_mean_dr, weighted_mean_dx, weighted_mean_dy

def getWarpFieldStdDevs(all_results) :
    all_drs, all_dxs, all_dys = getListsOfWarpFields(all_results)
    dr_stddev = np.std(list(all_drs),axis=0)
    dx_stddev = np.std(list(all_dxs),axis=0)
    dy_stddev = np.std(list(all_dys),axis=0)
    return dr_stddev, dx_stddev, dy_stddev
###################################################################################################################################

#plots the mean, weighted mean, and standard deviation dr, dx, and dy warp fields from a list of WarpFitResults
def warpFieldVariationPlots(all_results) :
    mean_dr, mean_dx, mean_dy = getMeanWarpFields(all_results)
    f,ax = plt.subplots(1,3,figsize=(3*6.4,(mean_dr.shape[0]/mean_dr.shape[1])*6.4))
    pos = ax[0].imshow(mean_dr)
    ax[0].set_title('mean dr')
    f.colorbar(pos,ax=ax[0])
    pos = ax[1].imshow(mean_dx)
    ax[1].set_title('mean dx')
    f.colorbar(pos,ax=ax[1])
    pos = ax[2].imshow(mean_dy)
    ax[2].set_title('mean dy')
    f.colorbar(pos,ax=ax[2])
    plt.show()
    weighted_mean_dr, weighted_mean_dx, weighted_mean_dy = getWeightedMeanWarpFields(all_results)
    f,ax = plt.subplots(1,3,figsize=(3*6.4,(weighted_mean_dr.shape[0]/weighted_mean_dr.shape[1])*6.4))
    pos = ax[0].imshow(weighted_mean_dr)
    ax[0].set_title('weighted mean dr')
    f.colorbar(pos,ax=ax[0])
    pos = ax[1].imshow(weighted_mean_dx)
    ax[1].set_title('weighted mean dx')
    f.colorbar(pos,ax=ax[1])
    pos = ax[2].imshow(weighted_mean_dy)
    ax[2].set_title('weighted mean dy')
    f.colorbar(pos,ax=ax[2])
    plt.show()
    dr_stddev, dx_stddev, dy_stddev = getWarpFieldStdDevs(all_results)
    f,ax = plt.subplots(1,3,figsize=(3*6.4,(dr_stddev.shape[0]/dr_stddev.shape[1])*6.4))
    pos = ax[0].imshow(dr_stddev)
    ax[0].set_title('dr std. dev.')
    f.colorbar(pos,ax=ax[0])
    pos = ax[1].imshow(dx_stddev)
    ax[1].set_title('dx std. dev')
    f.colorbar(pos,ax=ax[1])
    pos = ax[2].imshow(dy_stddev)
    ax[2].set_title('dy std. dev')
    f.colorbar(pos,ax=ax[2])
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
