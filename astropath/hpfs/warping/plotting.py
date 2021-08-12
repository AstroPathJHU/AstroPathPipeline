#imports
import numpy as np
import matplotlib.pyplot as plt
from ...utilities.misc import save_figure_in_dir

def principal_point_plot(all_results,save_stem=None,save_dir=None) :
    """
    makes a plot of the principal points in a list of results shaded by cost reduction
    """
    mean_cx = np.mean([r.cx for r in all_results if r.cost_reduction>0.]) 
    cx_err = np.std([r.cx for r in all_results if r.cost_reduction>0.])
    mean_cy = np.mean([r.cy for r in all_results if r.cost_reduction>0.])
    cy_err = np.std([r.cy for r in all_results if r.cost_reduction>0.])
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
    f,ax=plt.subplots()
    pos = ax.scatter([r.cx for r in all_results],[r.cy for r in all_results],c=[r.cost_reduction for r in all_results])
    meanlabel = f'mean=({mean_cx:.1f}'+r'$\pm$'+f'{cx_err:.1f} , {mean_cy:.1f}'+r'$\pm$'+f'{cy_err:.1f})'
    wmeanlabel = f'mean=({weighted_mean_cx:.1f}'+r'$\pm$'+f'{w_mean_cx_err:.1f} , {weighted_mean_cy:.1f}'
    wmeanlabel+= r'$\pm$'+f'{w_mean_cy_err:.1f})'
    ax.errorbar(mean_cx,mean_cy,yerr=cy_err,xerr=cx_err,marker='x',color='tab:red',label=meanlabel)
    ax.errorbar(weighted_mean_cx,weighted_mean_cy,yerr=w_mean_cy_err,xerr=w_mean_cx_err,
                marker='x',color='tab:blue',label=wmeanlabel)
    ax.set_title('Center point locations with cost redux')
    ax.set_xlabel('cx point')
    ax.set_ylabel('cy point')
    ax.legend(loc='best')
    f.colorbar(pos,ax=ax)
    if save_stem is not None :
        fn = f'{save_stem}_principal_point_plot.png'
        save_figure_in_dir(plt,fn,save_dir)
    else :
        plt.show()

def rad_warp_amt_plots(all_results,save_stem=None,save_dir=None) :
    """
    makes plots of the maximum amounts of radial warping, the cost reduction vs. the amount of max. radial warping, 
    and the principal points locations shaded by max amount of radial warping
    """
    vs = np.array([r.max_rad_warp for r in all_results])
    good_results = [r for r in all_results if r.cost_reduction>0]
    sum_weights = np.sum([r.cost_reduction for r in good_results])
    weighted_mean = np.sum([r.max_rad_warp*r.cost_reduction for r in good_results])/sum_weights
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
    mean_cx = np.mean([r.cx for r in good_results]); cx_err = np.std([r.cx for r in good_results])
    mean_cy = np.mean([r.cy for r in good_results]); cy_err = np.std([r.cy for r in good_results])
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
    ax[2].errorbar(weighted_mean_cx,weighted_mean_cy,yerr=w_mean_cy_err,xerr=w_mean_cx_err,
                    marker='x',color='tab:blue',label='weighted mean')
    ax[2].set_title('Center point locations colored by max radial warp')
    ax[2].set_xlabel('cx point')
    ax[2].set_ylabel('cy point')
    ax[2].legend(loc='best')
    f.colorbar(pos,ax=ax[2])
    if save_stem is not None :
        fn = f'{save_stem}_radial_warp_amount_plots.png'
        save_figure_in_dir(plt,fn,save_dir)
    else :
        plt.show()

#makes plots of the radial warping parameters and the cost reductions vs each
def rad_warp_par_plots(all_results,save_stem=None,save_dir=None) :
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
        save_figure_in_dir(plt,fn,save_dir)
    else :
        plt.show()

#makes a plot of the cost reduction vs. # of fit iterations
def fit_iteration_plot(all_results,save_stem=None,save_dir=None) :
    f,ax = plt.subplots()
    xs = [r.fit_its for r in all_results]
    ys = [r.cost_reduction for r in all_results]
    ax.scatter(xs,ys)
    ax.set_title('cost reduction vs. fit iterations')
    ax.set_xlabel('number of fit iterations')
    ax.set_ylabel('fractional reduction in fit cost')
    if save_stem is not None :
        fn = f'{save_stem}_cost_redux_vs_fit_iterations_plot.png'
        save_figure_in_dir(plt,fn,save_dir)
    else :
        plt.show()

######## Several helper functions to translate lists of warp results ########
########    into various overall warp fields and their variations    ########
def get_lists_of_warp_fields(all_results) :
    all_results = [r for r in all_results if r.cost_reduction>0.]
    all_warps = []
    for r in all_results :
        all_warps.append(CameraWarp(n=r.n,m=r.m,cx=r.cx,cy=r.cy,fx=r.fx,fy=r.fy,
                                    k1=r.k1,k2=r.k2,k3=r.k3,p1=r.p1,p2=r.p2))
    all_drs = []; all_dxs = []; all_dys= []
    for w in all_warps :
        dr, dx, dy = w.getWarpFields()
        all_drs.append(dr); all_dxs.append(dx); all_dys.append(dy)
    return all_drs, all_dxs, all_dys

def get_mean_warp_fields(all_results) :
    all_results = [r for r in all_results if r.cost_reduction>0.]
    all_drs, all_dxs, all_dys = get_lists_of_warp_fields(all_results)
    mean_dr = np.mean(list(all_drs),axis=0)
    mean_dx = np.mean(list(all_dxs),axis=0)
    mean_dy = np.mean(list(all_dys),axis=0)
    return mean_dr, mean_dx, mean_dy

def get_weighted_mean_warp_fields(all_results) :
    all_results = [r for r in all_results if r.cost_reduction>0.]
    all_drs, all_dxs, all_dys = get_lists_of_warp_fields(all_results)
    sum_weights = np.sum([r.cost_reduction for r in all_results])
    weighted_mean_dr = np.sum([dr*r.cost_reduction for dr,r in zip(all_drs,all_results)],axis=0)/sum_weights
    weighted_mean_dx = np.sum([dx*r.cost_reduction for dx,r in zip(all_dxs,all_results)],axis=0)/sum_weights
    weighted_mean_dy = np.sum([dy*r.cost_reduction for dy,r in zip(all_dys,all_results)],axis=0)/sum_weights
    return weighted_mean_dr, weighted_mean_dx, weighted_mean_dy

def get_warp_field_std_devs(all_results) :
    all_results = [r for r in all_results if r.cost_reduction>0.]
    all_drs, all_dxs, all_dys = get_lists_of_warp_fields(all_results)
    dr_stddev = np.std(list(all_drs),axis=0)
    dx_stddev = np.std(list(all_dxs),axis=0)
    dy_stddev = np.std(list(all_dys),axis=0)
    return dr_stddev, dx_stddev, dy_stddev

def get_warp_field_weighted_std_err(all_results) :
    all_results = [r for r in all_results if r.cost_reduction>0.]
    all_drs, all_dxs, all_dys = get_lists_of_warp_fields(all_results)
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
def warp_field_variation_plots(all_results,save_stem=None,save_dir=None) :
    mean_dr, mean_dx, mean_dy = get_mean_warp_fields(all_results)
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
    dr_stddev, dx_stddev, dy_stddev = get_warp_field_std_devs(all_results)
    pos = ax[1][0].imshow(dr_stddev)
    ax[1][0].set_title('dr std. dev.')
    f.colorbar(pos,ax=ax[1][0])
    pos = ax[1][1].imshow(dx_stddev)
    ax[1][1].set_title('dx std. dev')
    f.colorbar(pos,ax=ax[1][1])
    pos = ax[1][2].imshow(dy_stddev)
    ax[1][2].set_title('dy std. dev')
    f.colorbar(pos,ax=ax[1][2])
    weighted_mean_dr, weighted_mean_dx, weighted_mean_dy = get_weighted_mean_warp_fields(all_results)
    pos = ax[2][0].imshow(weighted_mean_dr)
    ax[2][0].set_title('weighted mean dr')
    f.colorbar(pos,ax=ax[2][0])
    pos = ax[2][1].imshow(weighted_mean_dx)
    ax[2][1].set_title('weighted mean dx')
    f.colorbar(pos,ax=ax[2][1])
    pos = ax[2][2].imshow(weighted_mean_dy)
    ax[2][2].set_title('weighted mean dy')
    f.colorbar(pos,ax=ax[2][2])
    weighted_dr_stderr, weighted_dx_stderr, weighted_dy_stderr = get_warp_field_weighted_std_err(all_results)
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
        save_figure_in_dir(plt,fn,save_dir)
    else :
        plt.show()
