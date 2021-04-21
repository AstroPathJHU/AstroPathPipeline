#imports
from astropath.utilities.img_file_io import getImageHWLFromXMLFile, getRawAsHWL, smoothImageWithUncertaintyWorker
from astropath.utilities.dataclasses import MyDataClass
from astropath.utilities.tableio import writetable
from astropath.utilities.misc import cd, split_csv_to_list, cropAndOverwriteImage
from astropath.utilities.config import CONST as UNIV_CONST
from argparse import ArgumentParser
import numpy as np, matplotlib.pyplot as plt
import pathlib

#little helper dataclass to organize numerical entries
class TableEntry(MyDataClass) :
    slide_ID_1               : str
    slide_ID_2               : str
    layer_n                  : int
    delta_over_sigma_std_dev : float

#helper function to check the arguments
def checkArgs(args) :
    #root dir must exist
    rdp = pathlib.Path(args.root_dir)
    if not rdp.is_dir() :
        raise ValueError(f'ERROR: root directory {args.root_dir} does not exist!')
    #meanimages and standard errors must exist for all slides
    for sid in args.slides :
        mifp = pathlib.Path(args.root_dir) / sid / 'im3' / 'meanimage' / f'{sid}-mean_image.bin'
        if not mifp.is_file() :
            print(f'WARNING: Mean image file does not exist for slide {sid}, will skip this slide!')
            args.slides.remove(sid)
            continue
        semifp = pathlib.Path(args.root_dir) / sid / 'im3' / 'meanimage' / f'{sid}-std_error_of_mean_image.bin'
        if not mifp.is_file() :
            print(f'WARNING: Mean image file does not exist for slide {sid}, will skip this slide!')
            args.slides.remove(sid)
            continue
    if len(args.slides)<1 :
        print(f'ERROR: no valid slides remain!')
    #create the working directory if it doesn't already exist
    wdp = pathlib.Path(args.workingdir)
    if not wdp.is_dir() :
        pathlib.Path.mkdir(wdp)

#helper function to normalize an image layer by its weighted mean 
def normalizeImageLayer(mil,semil) :
    weights = 1./(semil**2)
    weighted_mil = weights*mil
    sum_weighted_mil = np.sum(weighted_mil)
    sum_weights = np.sum(weights)
    mil_mean = sum_weighted_mil/sum_weights
    #mil_mean = np.average(mil,weights=weights)
    return mil/mil_mean, semil/mil_mean

#helper function to get the standard deviation of the delta/sigma distribution for every layer of two meanimages compared to one another
def get_delta_over_sigma_std_devs_by_layer(dims,mi1,semi1,mi2,semi2) :
    delta_over_sigma_std_devs = []
    layers = UNIV_CONST.BRIGHTEST_LAYERS_35 if dims[-1]==35 else UNIV_CONST.BRIGHTEST_LAYERS_43
    for ln in range(1,dims[-1]+1) :
        if ln not in layers :
            delta_over_sigma_std_devs.append(0.)
            continue
        print(f'\tDoing layer {ln}...')
        mil1 = mi1[:,:,ln-1]; semil1 = semi1[:,:,ln-1]
        mil2 = mi2[:,:,ln-1]; semil2 = semi2[:,:,ln-1]
        mil1max=np.max(mil1); mil1min=np.min(mil1)
        mil2max=np.max(mil2); mil2min=np.min(mil2)
        semil1min = np.min(semil1); semil2min = np.min(semil2)
        if mil1max==mil1min or mil2max==mil2min or semil1min==0. or semil2min==0. :
            delta_over_sigma_std_devs.append(0.)
            continue
        #normalize the image layers by the mean of the mean image layer
        mil1,semil1 = normalizeImageLayer(mil1,semil1)
        mil2,semil2 = normalizeImageLayer(mil2,semil2)
        #make the delta/sigma image
        delta_over_sigma = (mil1-mil2)/(np.sqrt(semil1**2+semil2**2))
        std_dev = np.std(delta_over_sigma)
        delta_over_sigma_std_devs.append(std_dev)
    return delta_over_sigma_std_devs

#helper function to make the consistency check grid plot
def consistency_check_grid_plot(slide_ids,root_dir,workingdir) :
    #get the dimensions of the images
    dims = getImageHWLFromXMLFile(root_dir,slide_ids[0])
    #start up an array to hold all of the necessary values and a list of table entries
    layers = UNIV_CONST.BRIGHTEST_LAYERS_35 if dims[-1]==35 else UNIV_CONST.BRIGHTEST_LAYERS_43
    table_entries = []
    dos_std_dev_plot_values = np.zeros((len(slide_ids),len(slide_ids),dims[-1]))
    #for each possible pair of slide ids, find the standard deviation in each image layer of the delta/sigma
    pairs_done = set()
    for is1,sid1 in enumerate(slide_ids) :
        mi1   = getRawAsHWL((pathlib.Path(root_dir) / sid1 / 'im3' / 'meanimage' / f'{sid1}-mean_image.bin'),
                            *(dims),np.float64)
        semi1 = getRawAsHWL((pathlib.Path(root_dir) / sid1 / 'im3' / 'meanimage' / f'{sid1}-std_error_of_mean_image.bin'),
                            *(dims),np.float64)
        for is2,sid2 in enumerate(slide_ids) :
            if sid2==sid1 :
                dos_std_dev_plot_values[is1,is2,:] = 0.
                continue 
            elif (sid2,sid1) in pairs_done :
                for li in range(dims[-1]) :
                    dos_std_dev_plot_values[is1,is2,li] = dos_std_dev_plot_values[is2,is1,li]
                continue
            print(f'Finding std. devs. of delta/sigma for {sid1} vs. {sid2}...')
            mi2   = getRawAsHWL((pathlib.Path(root_dir) / sid2 / 'im3' / 'meanimage' / f'{sid2}-mean_image.bin'),
                                *(dims),np.float64)
            semi2 = getRawAsHWL((pathlib.Path(root_dir) / sid2 / 'im3' / 'meanimage' / f'{sid2}-std_error_of_mean_image.bin'),
                                *(dims),np.float64)
            dossd_list = get_delta_over_sigma_std_devs_by_layer(dims,mi1,semi1,mi2,semi2)
            dos_std_dev_plot_values[is1,is2,:] = dossd_list
            for ln in layers :
                table_entries.append(TableEntry(sid1,sid2,ln,dos_std_dev_plot_values[is1,is2,ln-1]))
            pairs_done.add((sid1,sid2))
    with cd(workingdir) :
        writetable('meanimage_comparison_table.csv',table_entries)
    #for each image layer, plot a grid of the delta/sigma comparisons
    for ln in layers : 
        print(f'Saving plot for layer {ln}...')
        fig,ax = plt.subplots(figsize=(1.*len(slide_ids),1.*len(slide_ids)))
        pos = ax.imshow(dos_std_dev_plot_values[:,:,ln-1])
        ax.set_xticks(np.arange(len(slide_ids)))
        ax.set_yticks(np.arange(len(slide_ids)))
        ax.set_xticklabels(slide_ids)
        ax.set_yticklabels(slide_ids)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")
        for i in range(len(slide_ids)):
            for j in range(len(slide_ids)):
                v = dos_std_dev_plot_values[i,j,ln-1]
                if v!=0.0 :
                    text = ax.text(j, i, f'{v:.02f}',ha="center", va="center", color="b")
                    text.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='white'))
        ax.set_title(f'mean image delta/sigma std. devs. in layer {ln}')
        fig.colorbar(pos,ax=ax)
        fig.tight_layout()
        figname = f'meanimage_comparison_layer_{ln}.png'
        with cd(workingdir) :
            plt.savefig(figname)
            cropAndOverwriteImage(figname)

#################### MAIN SCRIPT ####################

def main(args=None) :
    parser = ArgumentParser()
    parser.add_argument('--slides', type=split_csv_to_list,
                        help='comma-separated list of slides whose meanimages/flatfields should be compared')
    parser.add_argument('--root_dir', 
                        help='Path to the directory with all of the [slideID]/im3/meanimage subdirectories in it')
    parser.add_argument('--workingdir', 
                        help='Path to the working directory where results will be saved')
    args = parser.parse_args(args=args)
    #check the arguments
    checkArgs(args)
    #run the main workhorse function
    consistency_check_grid_plot(args.slides,args.root_dir,args.workingdir)
    print('Done : )')

if __name__=='__main__' :
    main()