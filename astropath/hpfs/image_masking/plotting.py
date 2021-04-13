#imports
from ...utilities.misc import cd, cropAndOverwriteImage
import numpy as np, matplotlib.pyplot as plt

#helper function to write out a sheet of masking information plots for an image
def doMaskingPlotsForImage(image_key,tissue_mask,plot_dict_lists,compressed_full_mask,savedir=None) :
    #figure out how many rows/columns will be in the sheet and set up the plots
    n_rows = len(plot_dict_lists)+1
    n_cols = max(n_rows,len(plot_dict_lists[0]))
    for pdi in range(1,len(plot_dict_lists)) :
        if len(plot_dict_lists[pdi]) > n_cols :
            n_cols = len(plot_dict_lists[pdi])
    f,ax = plt.subplots(n_rows,n_cols,figsize=(n_cols*6.4,n_rows*tissue_mask.shape[0]/tissue_mask.shape[1]*6.4))
    #add the masking plots for each layer group
    for row,plot_dicts in enumerate(plot_dict_lists) :
        for col,pd in enumerate(plot_dicts) :
            dkeys = pd.keys()
            #imshow plots
            if 'image' in dkeys :
                #edit the overlay based on the full mask
                if 'title' in dkeys and 'overlay (clipped)' in pd['title'] :
                    pd['image'][:,:,1][compressed_full_mask[:,:,row+1]>1]=pd['image'][:,:,0][compressed_full_mask[:,:,row+1]>1]
                    pd['image'][:,:,0][(compressed_full_mask[:,:,row+1]>1) & (pd['image'][:,:,2]!=0)]=0
                    pd['image'][:,:,2][(compressed_full_mask[:,:,row+1]>1) & (pd['image'][:,:,2]!=0)]=0
                imshowkwargs = {}
                possible_keys = ['cmap','vmin','vmax']
                for pk in possible_keys :
                    if pk in dkeys :
                        imshowkwargs[pk]=pd[pk]
                pos = ax[row][col].imshow(pd['image'],**imshowkwargs)
                f.colorbar(pos,ax=ax[row][col])
                if 'title' in dkeys :
                    title_text = pd['title'].replace('IMAGE',image_key)
                    ax[row][col].set_title(title_text)
            #histogram plots
            elif 'hist' in dkeys :
                binsarg=100
                logarg=False
                if 'bins' in dkeys :
                    binsarg=pd['bins']
                if 'log_scale' in dkeys :
                    logarg=pd['log_scale']
                ax[row][col].hist(pd['hist'],binsarg,log=logarg)
                if 'xlabel' in dkeys :
                    xlabel_text = pd['xlabel'].replace('IMAGE',image_key)
                    ax[row][col].set_xlabel(xlabel_text)
                if 'line_at' in dkeys :
                    ax[row][col].plot([pd['line_at'],pd['line_at']],
                                    [0.8*y for y in ax[row][col].get_ylim()],
                                    linewidth=2,color='tab:red',label=pd['line_at'])
                    ax[row][col].legend(loc='best')
            #bar plots
            elif 'bar' in dkeys :
                ax[row][col].bar(pd['bins'][:-1],pd['bar'],width=1.0)
                if 'xlabel' in dkeys :
                    xlabel_text = pd['xlabel'].replace('IMAGE',image_key)
                    ax[row][col].set_xlabel(xlabel_text)
                if 'line_at' in dkeys :
                    ax[row][col].plot([pd['line_at'],pd['line_at']],
                                    [0.8*y for y in ax[row][col].get_ylim()],
                                    linewidth=2,color='tab:red',label=pd['line_at'])
                    ax[row][col].legend(loc='best')
            #remove axes from any extra slots
            if col==len(plot_dicts)-1 and col<n_cols-1 :
                for ci in range(col+1,n_cols) :
                    ax[row][ci].axis('off')
    #add the plots of the full mask layer groups
    enumerated_mask_max = np.max(compressed_full_mask)
    for lgi in range(compressed_full_mask.shape[-1]-1) :
        pos = ax[n_rows-1][lgi].imshow(compressed_full_mask[:,:,lgi+1],vmin=0.,vmax=enumerated_mask_max,cmap='rainbow')
        f.colorbar(pos,ax=ax[n_rows-1][lgi])
        ax[n_rows-1][lgi].set_title(f'full mask, layer group {lgi+1}')
    #empty the other unused axes in the last row
    for ci in range(n_rows-1,n_cols) :
        ax[n_rows-1][ci].axis('off')
    #show/save the plot
    if savedir is None :
        plt.show()
    else :
        with cd(savedir) :
            fn = f'{image_key}_masking_plots.png'
            plt.savefig(fn); plt.close(); cropAndOverwriteImage(fn)