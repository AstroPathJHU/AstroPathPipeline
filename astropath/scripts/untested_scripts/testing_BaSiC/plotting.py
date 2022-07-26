#imports
import numpy as np, matplotlib as mpl, matplotlib.pyplot as plt
from astropath.utilities.miscplotting import save_figure_in_dir
from astropath.utilities.img_file_io import smooth_image_worker
from .utilities import timestamp

def illumination_variation_plots(samp,uncorr_mi,mi_corr_mi,basic_corr_mi,central=False,save_dirpath=None) :
    fn = 'smoothed_mean_image_pixel_intensities'
    if central :
        fn+='_central'
    fn+='.png'
    if (save_dirpath/fn).is_file() :
        print(f'{timestamp()} illumination variation plot already exists at {save_dirpath/fn}')
        return
    sm_uncorr_mi = smooth_image_worker(uncorr_mi,100,gpu=True)
    sm_mi_corr_mi = smooth_image_worker(mi_corr_mi,100,gpu=True)
    sm_basic_corr_mi = smooth_image_worker(basic_corr_mi,100,gpu=True)
    #clip the outer edges off if the plots are for the central region only
    if central :
        yclip = int(sm_uncorr_mi.shape[0]*0.1)
        xclip = int(sm_uncorr_mi.shape[1]*0.1)
        sm_uncorr_mi = sm_uncorr_mi[yclip:-yclip,xclip:-xclip,:]
        sm_mi_corr_mi = sm_mi_corr_mi[yclip:-yclip,xclip:-xclip,:]
    #figure out the number of layers and filter breaks
    nlayers=sm_uncorr_mi.shape[-1]
    last_filter_layers = [lg[1] for lg in list(samp.layer_groups.values())[:-1]] 
    #keep track of the uncorrected and corrected images' minimum and maximum (and 5/95%ile) pixel intensities 
    u_low_pixel_intensities=[]; u_high_pixel_intensities=[]
    mi_c_low_pixel_intensities=[]; mi_c_high_pixel_intensities=[]
    basic_c_low_pixel_intensities=[]; basic_c_high_pixel_intensities=[]
    u_std_devs=[]; mi_c_std_devs=[]; basic_c_std_devs=[]
    plt.figure(figsize=(16.8,(27./64.)*16.8))
    xaxis_vals = list(range(1,nlayers+1))
    #iterate over the layers
    for layer_i in range(nlayers) :
        #find the min, max, and 5/95%ile pixel intensities for this uncorrected and corrected image layer
        sorted_u_layer = np.sort((sm_uncorr_mi[:,:,layer_i]).flatten())/np.mean(sm_uncorr_mi[:,:,layer_i])
        u_low_pixel_intensities.append(sorted_u_layer[int(0.05*len(sorted_u_layer))])
        u_stddev = np.std(sorted_u_layer); u_std_devs.append(u_stddev)
        u_high_pixel_intensities.append(sorted_u_layer[int(0.95*len(sorted_u_layer))])
        sorted_mi_c_layer = np.sort((sm_mi_corr_mi[:,:,layer_i]).flatten())
        sorted_mi_c_layer/=np.mean(sm_mi_corr_mi[:,:,layer_i])
        mi_c_low_pixel_intensities.append(sorted_mi_c_layer[int(0.05*len(sorted_mi_c_layer))])
        mi_c_stddev = np.std(sorted_mi_c_layer); mi_c_std_devs.append(mi_c_stddev)
        mi_c_high_pixel_intensities.append(sorted_mi_c_layer[int(0.95*len(sorted_mi_c_layer))])
        sorted_basic_c_layer = np.sort((sm_basic_corr_mi[:,:,layer_i]).flatten())
        sorted_basic_c_layer/=np.mean(sm_basic_corr_mi[:,:,layer_i])
        basic_c_low_pixel_intensities.append(sorted_basic_c_layer[int(0.05*len(sorted_basic_c_layer))])
        basic_c_stddev = np.std(sorted_basic_c_layer); basic_c_std_devs.append(basic_c_stddev)
        basic_c_high_pixel_intensities.append(sorted_basic_c_layer[int(0.95*len(sorted_basic_c_layer))])
        if layer_i==0 :
            plt.fill([layer_i+0.5,layer_i+0.5,layer_i+1.5,layer_i+1.5],
                     [1.-u_stddev,1.+u_stddev,1.+u_stddev,1.-u_stddev],
                     'mediumseagreen',alpha=0.5,label='uncorrected std. dev.')
            plt.fill([layer_i+0.5,layer_i+0.5,layer_i+1.5,layer_i+1.5],
                     [1.-mi_c_stddev,1.+mi_c_stddev,1.+mi_c_stddev,1.-mi_c_stddev],
                     'goldenrod',alpha=0.5,label='meanimage corrected std. dev.')
            plt.fill([layer_i+0.5,layer_i+0.5,layer_i+1.5,layer_i+1.5],
                     [1.-basic_c_stddev,1.+basic_c_stddev,1.+basic_c_stddev,1.-basic_c_stddev],
                     'darkorchid',alpha=0.5,label='BaSiC corrected std. dev.')
        else :
            plt.fill([layer_i+0.5,layer_i+0.5,layer_i+1.5,layer_i+1.5],
                     [1.-u_stddev,1.+u_stddev,1.+u_stddev,1.-u_stddev],'mediumseagreen',alpha=0.5)
            plt.fill([layer_i+0.5,layer_i+0.5,layer_i+1.5,layer_i+1.5],
                     [1.-mi_c_stddev,1.+mi_c_stddev,1.+mi_c_stddev,1.-mi_c_stddev],'goldenrod',alpha=0.5)
            plt.fill([layer_i+0.5,layer_i+0.5,layer_i+1.5,layer_i+1.5],
                     [1.-basic_c_stddev,1.+basic_c_stddev,1.+basic_c_stddev,1.-basic_c_stddev],'darkorchid',alpha=0.5)
    #plot the relative intensity plots together, with the broadband filter breaks
    plt.plot([xaxis_vals[0],xaxis_vals[-1]],[1.0,1.0],color='darkgreen',linestyle='dashed',label='mean')
    totalmin=min(min(u_low_pixel_intensities),min(mi_c_low_pixel_intensities),min(basic_c_low_pixel_intensities))
    totalmax=max(max(u_high_pixel_intensities),max(mi_c_high_pixel_intensities),max(basic_c_high_pixel_intensities))
    for i in range(len(last_filter_layers)+1) :
        f_i = 0 if i==0 else last_filter_layers[i-1]
        l_i = xaxis_vals[-1] if i==len(last_filter_layers) else last_filter_layers[i]
        if i==0 :
            plt.plot(xaxis_vals[f_i:l_i],u_low_pixel_intensities[f_i:l_i],
                     color='darkred',marker='v',label=r'uncorrected 5th %ile')
            plt.plot(xaxis_vals[f_i:l_i],u_high_pixel_intensities[f_i:l_i],
                     color='darkred',marker='^',label=r'uncorrected 95th %ile')
            plt.plot(xaxis_vals[f_i:l_i],mi_c_low_pixel_intensities[f_i:l_i],
                     color='darkblue',marker='v',label=r'meanimage corrected 5th %ile')
            plt.plot(xaxis_vals[f_i:l_i],mi_c_high_pixel_intensities[f_i:l_i],
                     color='darkblue',marker='^',label=r'meanimage corrected 95th %ile')
            plt.plot(xaxis_vals[f_i:l_i],basic_c_low_pixel_intensities[f_i:l_i],
                     color='darkorange',marker='v',label=r'BaSiC corrected 5th %ile')
            plt.plot(xaxis_vals[f_i:l_i],basic_c_high_pixel_intensities[f_i:l_i],
                     color='darkorange',marker='^',label=r'BaSiC corrected 95th %ile')
            plt.plot([l_i+0.5,l_i+0.5],[totalmin-0.01,totalmax+0.01],
                     color='black',linewidth=2,linestyle='dotted',label='broadband filter changeover')
        else :
            plt.plot(xaxis_vals[f_i:l_i],u_low_pixel_intensities[f_i:l_i],color='darkred',marker='v')
            plt.plot(xaxis_vals[f_i:l_i],u_high_pixel_intensities[f_i:l_i],color='darkred',marker='^')
            plt.plot(xaxis_vals[f_i:l_i],mi_c_low_pixel_intensities[f_i:l_i],color='darkblue',marker='v')
            plt.plot(xaxis_vals[f_i:l_i],mi_c_high_pixel_intensities[f_i:l_i],color='darkblue',marker='^')
            plt.plot(xaxis_vals[f_i:l_i],basic_c_low_pixel_intensities[f_i:l_i],color='darkorange',marker='v')
            plt.plot(xaxis_vals[f_i:l_i],basic_c_high_pixel_intensities[f_i:l_i],color='darkorange',marker='^')
            if i!=len(last_filter_layers) :
                plt.plot([l_i+0.5,l_i+0.5],[totalmin-0.01,totalmax+0.01],color='black',linewidth=2,linestyle='dotted')
    plt.title('uncorrected/corrected smoothed mean image relative pixel intensities',fontsize=14)
    plt.xlabel('layer number',fontsize=14)
    #fix the range on the x-axis to accommodate the legend
    plt.xlim(0,nlayers+(10 if nlayers==35 else 12))
    plt.ylabel('pixel intensity relative to layer mean',fontsize=14)
    plt.legend(loc='lower right')
    save_figure_in_dir(plt,fn,save_dirpath)

def overlap_mse_reduction_plots(overlap_comparisons_by_layer_n,save_dirpath) :
    layer_dir = save_dirpath/'mse_reduction_layer_plots'
    if not layer_dir.is_dir() :
        layer_dir.mkdir(parents=True)
    for layer_n in overlap_comparisons_by_layer_n.keys() :
        fn = f'overlap_mse_reduction_comparisons_layer_{layer_n}.png'
        if (layer_dir/fn).is_file() :
            continue
        overlap_comparisons = overlap_comparisons_by_layer_n[layer_n]
        f,ax = plt.subplots(4,4,figsize=(4*6.4,4*4.6))
        sum_weights = np.sum(np.array([oc.npix for oc in overlap_comparisons]))
        orig_rel_residuals = [oc.orig_mse_diff/oc.orig_mse1 for oc in overlap_comparisons]
        stddev_orig_rel_residuals = np.std(np.array(orig_rel_residuals))
        w_mean_orig_rel_residuals = np.sum(np.array([oc.npix*(oc.orig_mse_diff/oc.orig_mse1) for oc in overlap_comparisons]))/sum_weights
        meanimage_rel_residuals = [oc.meanimage_mse_diff/oc.meanimage_mse1 for oc in overlap_comparisons]
        stddev_meanimage_rel_residuals = np.std(np.array(meanimage_rel_residuals))
        w_mean_meanimage_rel_residuals = np.sum(np.array([oc.npix*(oc.meanimage_mse_diff/oc.meanimage_mse1) for oc in overlap_comparisons]))/sum_weights
        basic_rel_residuals = [oc.basic_mse_diff/oc.basic_mse1 for oc in overlap_comparisons]
        stddev_basic_rel_residuals = np.std(np.array(basic_rel_residuals))
        w_mean_basic_rel_residuals = np.sum(np.array([oc.npix*(oc.basic_mse_diff/oc.basic_mse1) for oc in overlap_comparisons]))/sum_weights
        alignment_x_diffs = [oc.basic_dx-oc.meanimage_dx for oc in overlap_comparisons]
        alignment_y_diffs = [oc.basic_dy-oc.meanimage_dy for oc in overlap_comparisons]
        ax[0][0].hist(orig_rel_residuals,bins=80)
        ax[0][0].set_title('uncorrected relative residuals')
        ax[0][0].axvline(w_mean_orig_rel_residuals,label=f'w. mean = {w_mean_orig_rel_residuals:.4f}+/-{stddev_orig_rel_residuals:.4f}',color='r',linewidth=2)
        ax[0][0].legend()
        ax[0][1].hist(meanimage_rel_residuals,bins=80)
        ax[0][1].set_title('meanimage relative residuals')
        ax[0][1].axvline(w_mean_meanimage_rel_residuals,label=f'w. mean = {w_mean_meanimage_rel_residuals:.4f}+/-{stddev_meanimage_rel_residuals:.4f}',color='r',linewidth=2)
        ax[0][1].legend()
        ax[0][2].hist(basic_rel_residuals,bins=80)
        ax[0][2].set_title('BaSiC relative residuals')
        ax[0][2].axvline(w_mean_basic_rel_residuals,label=f'w. mean = {w_mean_basic_rel_residuals:.4f}+/-{stddev_basic_rel_residuals:.4f}',color='r',linewidth=2)
        ax[0][2].legend()
        ax[0][3].hist2d(alignment_x_diffs,alignment_y_diffs,bins=60,norm=mpl.colors.LogNorm())
        ax[0][3].set_title('alignment x/y differences')
        for ti,tag_pair in enumerate([(1,9),(2,8),(3,7),(4,6)]) :
            tag_comparisons = [oc for oc in overlap_comparisons if oc.tag in tag_pair]
            weights = [oc.npix for oc in tag_comparisons]
            sum_weights = np.sum(np.array(weights))
            rel_residual_diffs = [(oc.basic_mse_diff/oc.basic_mse1)-(oc.meanimage_mse_diff/oc.meanimage_mse1) for oc in tag_comparisons]
            stddev_rel_residual_diffs = np.std(np.array(rel_residual_diffs))
            w_mean_rel_residual_diffs = np.sum(np.array([weight*rel_resid_diff for weight,rel_resid_diff in zip(weights,rel_residual_diffs)]))/sum_weights
            meanimage_rel_residual_reduxes = [1.-(oc.meanimage_mse_diff/oc.meanimage_mse1)/(oc.orig_mse_diff/oc.orig_mse1) for oc in tag_comparisons]
            basic_rel_residual_reduxes = [1.-(oc.basic_mse_diff/oc.basic_mse1)/(oc.orig_mse_diff/oc.orig_mse1) for oc in tag_comparisons]
            rel_residual_redux_diffs = [brrr-mirrr for brrr,mirrr in zip(basic_rel_residual_reduxes,meanimage_rel_residual_reduxes)]
            stddev_rel_residual_redux_diffs = np.std(np.array(rel_residual_redux_diffs))
            w_mean_rel_residual_redux_diffs = np.sum(np.array([weight*rel_resid_redux_diff for weight,rel_resid_redux_diff in zip(weights,rel_residual_redux_diffs)]))/sum_weights
            ax[1][ti].hist(rel_residual_diffs,bins=80)
            ax[1][ti].set_title(f'BaSiC-meanimage rel. residuals, tag={tag_pair}')
            ax[1][ti].axvline(w_mean_rel_residual_diffs,label=f'w. mean = {w_mean_rel_residual_diffs:.4f}+/-{stddev_rel_residual_diffs:.4f}',color='r',linewidth=2)
            ax[1][ti].legend()
            ax[2][ti].hist2d(meanimage_rel_residual_reduxes,basic_rel_residual_reduxes,bins=60,norm=mpl.colors.LogNorm())
            ax[2][ti].set_title(f'BaSiC vs. meanimage rel. residual reductions, tag={tag_pair}')
            ax[2][ti].set_xlabel('meanimage rel. residual reductions')
            ax[2][ti].set_ylabel('BaSiC rel. residual reductions')
            ax[2][ti].axvline(0.,color='k',linewidth=2)
            ax[2][ti].axhline(0.,color='k',linewidth=2)
            ax[3][ti].hist(rel_residual_redux_diffs,bins=80)
            ax[3][ti].set_title(f'BaSiC-meanimage rel. residual reductions, tag={tag_pair}')
            ax[3][ti].axvline(w_mean_rel_residual_redux_diffs,label=f'w. mean = {w_mean_rel_residual_redux_diffs:.4f}+/-{stddev_rel_residual_redux_diffs:.4f}',color='r',linewidth=2)
            ax[3][ti].legend()
        plt.tight_layout()
        save_figure_in_dir(plt,fn,layer_dir)
        plt.close()

def overlap_mse_reduction_comparison_plot(samp,overlap_comparisons_by_layer_n,save_dirpath) :
    rel_mse_redux_diff_means = []
    rel_mse_redux_diff_stds = []
    for layer_n in overlap_comparisons_by_layer_n.keys() :
        rel_mse_redux_diff_means.append([])
        rel_mse_redux_diff_stds.append([])
        overlap_comparisons = overlap_comparisons_by_layer_n[layer_n]
        for tag_pair in [(1,9),(2,8),(3,7),(4,6)] :
            tag_comparisons = [oc for oc in overlap_comparisons if oc.tag in tag_pair]
            weights = [oc.npix for oc in tag_comparisons]
            sum_weights = np.sum(np.array(weights))
            meanimage_rel_mse_reduxes = [1.-(oc.meanimage_mse_diff/oc.meanimage_mse1)/(oc.orig_mse_diff/oc.orig_mse1) for oc in tag_comparisons]
            basic_rel_mse_reduxes = [1.-(oc.basic_mse_diff/oc.basic_mse1)/(oc.orig_mse_diff/oc.orig_mse1) for oc in tag_comparisons]
            rel_mse_redux_diffs = [brrr-mirrr for brrr,mirrr in zip(basic_rel_mse_reduxes,meanimage_rel_mse_reduxes)]
            stddev_rel_mse_redux_diffs = np.std(np.array(rel_mse_redux_diffs))
            w_mean_rel_mse_redux_diffs = np.sum(np.array([weight*rel_resid_redux_diff for weight,rel_resid_redux_diff in zip(weights,rel_mse_redux_diffs)]))/sum_weights
            rel_mse_redux_diff_means[-1].append(w_mean_rel_mse_redux_diffs)
            rel_mse_redux_diff_stds[-1].append(stddev_rel_mse_redux_diffs)
    rel_mse_redux_diff_means = (np.array(rel_mse_redux_diff_means)).T
    rel_mse_redux_diff_stds = (np.array(rel_mse_redux_diff_stds)).T
    xaxis_vals = np.array(list(range(1,len(overlap_comparisons_by_layer_n.keys())+1)))
    f,ax = plt.subplots(figsize=(10.,4.6))
    ax.axhline(0.0,color='gray',linestyle='dotted')
    last_filter_layers = [lg[1] for lg in list(samp.layer_groups.values())[:-1]] 
    for i in range(len(last_filter_layers)+1) :
        f_i = 0 if i==0 else last_filter_layers[i-1]
        l_i = xaxis_vals[-1] if i==len(last_filter_layers) else last_filter_layers[i]
        if i==0 :
            ax.errorbar(xaxis_vals[f_i:l_i],rel_mse_redux_diff_means[0,f_i:l_i],
                        yerr=rel_mse_redux_diff_stds[0,f_i:l_i],
                        color='darkblue',marker='^',alpha=0.85,label='tag 1/9')
            ax.errorbar(xaxis_vals[f_i:l_i]+0.1,rel_mse_redux_diff_means[1,f_i:l_i],
                        yerr=rel_mse_redux_diff_stds[1,f_i:l_i],
                        color='darkorange',marker='v',alpha=0.85,label='tag 2/8')
            ax.errorbar(xaxis_vals[f_i:l_i]+0.2,rel_mse_redux_diff_means[2,f_i:l_i],
                        yerr=rel_mse_redux_diff_stds[2,f_i:l_i],
                        color='darkgreen',marker='<',alpha=0.85,label='tag 3/7')
            ax.errorbar(xaxis_vals[f_i:l_i]+0.3,rel_mse_redux_diff_means[3,f_i:l_i],
                        yerr=rel_mse_redux_diff_stds[3,f_i:l_i],
                        color='darkmagenta',marker='>',alpha=0.85,label='tag 4/6')
            ax.axvline(l_i+0.5,color='black',linewidth=2,linestyle='dotted',label='broadband filter changeover')
        else :
            ax.errorbar(xaxis_vals[f_i:l_i],rel_mse_redux_diff_means[0,f_i:l_i],
                        yerr=rel_mse_redux_diff_stds[0,f_i:l_i],
                        color='darkblue',marker='^',alpha=0.85)
            ax.errorbar(xaxis_vals[f_i:l_i]+0.1,rel_mse_redux_diff_means[1,f_i:l_i],
                        yerr=rel_mse_redux_diff_stds[1,f_i:l_i],
                        color='darkorange',marker='v',alpha=0.85)
            ax.errorbar(xaxis_vals[f_i:l_i]+0.2,rel_mse_redux_diff_means[2,f_i:l_i],
                        yerr=rel_mse_redux_diff_stds[2,f_i:l_i],
                        color='darkgreen',marker='<',alpha=0.85)
            ax.errorbar(xaxis_vals[f_i:l_i]+0.3,rel_mse_redux_diff_means[3,f_i:l_i],
                        yerr=rel_mse_redux_diff_stds[3,f_i:l_i],
                        color='darkmagenta',marker='>',alpha=0.85)
            if i!=len(last_filter_layers) :
                ax.axvline(l_i+0.5,color='black',linewidth=2,linestyle='dotted')
    ax.set_xticks(xaxis_vals)
    ax.set_xticklabels(xaxis_vals)
    ax.set_xlabel('image layer')
    ax.set_ylabel('BaSiC-meanimage relative overlap MSE reductions')
    ax.legend(loc='best')
    save_figure_in_dir(plt,'mse_reduction_differences.png',save_dirpath)
    plt.close()

def overlap_mse_reduction_comparison_box_plot(samp,overlap_comparisons_by_layer_n,save_dirpath) :
    data_vals = []
    for layer_n in overlap_comparisons_by_layer_n.keys() :
        overlap_comparisons = overlap_comparisons_by_layer_n[layer_n]
        meanimage_rel_mse_reduxes = [1.-(oc.meanimage_mse_diff/oc.meanimage_mse1)/(oc.orig_mse_diff/oc.orig_mse1) for oc in overlap_comparisons]
        basic_rel_mse_reduxes = [1.-(oc.basic_mse_diff/oc.basic_mse1)/(oc.orig_mse_diff/oc.orig_mse1) for oc in overlap_comparisons]
        rel_mse_redux_diffs = [brrr-mirrr for brrr,mirrr in zip(basic_rel_mse_reduxes,meanimage_rel_mse_reduxes)]
        data_vals.append(rel_mse_redux_diffs)
    xaxis_vals = np.array(list(range(1,len(overlap_comparisons_by_layer_n.keys())+1)))
    f,ax = plt.subplots(figsize=(14.,6.4))
    ax.axhline(0.0,color='darkblue',linestyle='dashed')
    ax.boxplot(data_vals,
               notch=True,
               whis=(5,95),
               bootstrap=3000,
               labels=xaxis_vals,
               sym='',
               )
    last_filter_layers = [lg[1] for lg in list(samp.layer_groups.values())[:-1]] 
    for i in range(len(last_filter_layers)+1) :
        l_i = xaxis_vals[-1] if i==len(last_filter_layers) else last_filter_layers[i]
        if i!=len(last_filter_layers) :
            ax.axvline(l_i+0.5,color='black',linewidth=2,linestyle='dotted')
    ax.set_xticks(xaxis_vals)
    ax.set_xticklabels(xaxis_vals,rotation=45)
    ax.set_xlabel('image layer')
    ax.set_ylabel('BaSiC-meanimage relative overlap MSE reductions')
    ax.set_yscale('symlog')
    #ax.set_ylim(ax.get_ylim()[0],5e-2)
    ax.set_yticks([-10,-1,-0.1,0.0,0.1])
    ax.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.get_yaxis().set_minor_locator(mpl.ticker.FixedLocator([-20,-8,-6,-4,-2,-0.8,-0.6,-0.4,-0.2,-0.08,-0.06,-0.04,-0.02,0.02,0.04,0.06,0.08,0.12,0.14,0.16,0.18,0.22,0.24,0.26,0.28]))
    save_figure_in_dir(plt,'mse_reduction_differences_box_plot.png',save_dirpath)
    plt.close()
