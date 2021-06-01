#imports
from astropath.baseclasses.sample import SampleDef
from astropath.utilities.img_file_io import getImageHWLFromXMLFile, getRawAsHWL
from astropath.utilities.dataclasses import MyDataClass
from astropath.utilities.tableio import readtable,writetable
from astropath.utilities.misc import cd, split_csv_to_list, split_csv_to_list_of_floats, cropAndOverwriteImage
from astropath.utilities.config import CONST as UNIV_CONST
from argparse import ArgumentParser
import numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pathlib, math, logging

#name of the meanimage subdirectory (may be overwritten if 'flatw' flag is added)
MEANIMAGE_SUBDIR_NAME = 'meanimage'
FLATW_MEANIMAGE_SUBDIR_NAME = 'meanimage_from_fw_files'

#logger
logger = logging.getLogger("flatfield_consistency")
logger.setLevel(logging.DEBUG)
stream_handler = logging.StreamHandler()
logging_formatter = logging.Formatter("[%(asctime)s] %(message)s  [%(funcName)s]","%Y-%m-%d %H:%M:%S")
stream_handler.setFormatter(logging_formatter)
logger.addHandler(stream_handler)

#little helper dataclass to organize numerical entries
class TableEntry(MyDataClass) :
    root_dir_1               : str
    slide_ID_1               : str
    root_dir_2               : str
    slide_ID_2               : str
    layer_n                  : int
    delta_over_sigma_std_dev : float

#helper function to get a dictionary with keys=root directory paths, values = lists of slide IDs in that root directory
def get_slide_ids_by_root_dir(root_dirs,skip_slides) :
    slide_ids_by_rootdir = {}
    for root_dir in root_dirs :
        samps = readtable(pathlib.Path(root_dir)/'sampledef.csv',SampleDef)
        slide_ids_by_rootdir[pathlib.Path(root_dir)] = [s.SlideID for s in samps if ((s.isGood==1) and (s.SlideID not in skip_slides))]
    return slide_ids_by_rootdir

#helper function to return the list of slide IDs ordered according to some convention, and a list of slides to put lines after in the plot
def sort_slide_ids(slide_ids,slide_ids_by_rootdir,sort_by) :
    sampledefs = []
    for rd,sids in slide_ids_by_rootdir.items() :
        for sid in sids :
            if sid not in slide_ids :
                continue
            sampledefs.append(SampleDef(SlideID=sid,root=rd))
    if sort_by=='project_cohort_batch' :
        projects = list(set([sd.Project for sd in sampledefs]))
        cohorts = list(set([sd.Cohort for sd in sampledefs]))
        batches = list(set([sd.BatchID for sd in sampledefs]))
        projects.sort(); cohorts.sort(); batches.sort()
        ordered_sids = []
        lines_after = []
        for p in projects :
            if len(ordered_sids)>0 and ordered_sids[-1] not in lines_after :
                lines_after.append(ordered_sids[-1])
            for c in cohorts :
                if len(ordered_sids)>0 and ordered_sids[-1] not in lines_after :
                    lines_after.append(ordered_sids[-1])
                for b in batches :
                    if len(ordered_sids)>0 and ordered_sids[-1] not in lines_after :
                        lines_after.append(ordered_sids[-1])
                    for sd in sampledefs :
                        if sd.Project==p and sd.Cohort==c and sd.BatchID==b :
                            ordered_sids.append(sd.SlideID)
        return ordered_sids, lines_after[:-1]
    raise RuntimeError(f'ERROR: sort option {sort_by} is not recognized!')

#helper function to check the arguments
def checkArgs(args) :
    #if the 'flatw' flag was added, change the name of the meanimage subdirectory
    if args.flatw :
        MEANIMAGE_SUBDIR_NAME = FLATW_MEANIMAGE_SUBDIR_NAME
    #if a prior run file is given, just make sure it exists and can be read as TableEntry objects
    if args.input_file is not None :
        ifp = pathlib.Path(args.input_file)
        if not ifp.is_file() :
            raise ValueError(f'ERROR: input file {args.input_file} does not exist!')
        try :
            table_entries = readtable(ifp,TableEntry)
            table_entries = table_entries #pyflakes doesn't like me just making sure the table can be read
        except Exception as e :
            raise ValueError(f'ERROR: input file {args.input_file} could not be read as a list of table entries. Exception: {e}')
    else :
        #root dirs must exist, each with a sampledef.csv file in it
        for root_dir in args.root_dirs :
            rdp = pathlib.Path(root_dir)
            if not rdp.is_dir() :
                raise ValueError(f'ERROR: root directory {root_dir} does not exist!')
            sdfp = rdp/'sampledef.csv'
            if not sdfp.is_file() :
                raise ValueError(f'ERROR: root directory {root_dir} does not contain a sampledef.csv file!')
        slide_ids_by_rootdir = get_slide_ids_by_root_dir(args.root_dirs,args.skip_slides)
        #meanimages and standard errors must exist for all slides
        for root_dir,slide_ids in slide_ids_by_rootdir.items() :
            for sid in slide_ids :
                mifp = root_dir / sid / 'im3' / f'{MEANIMAGE_SUBDIR_NAME}' / f'{sid}-mean_image.bin'
                if not mifp.is_file() :
                    logger.warning(f'WARNING: Mean image file does not exist for slide {sid}, will skip this slide!')
                    slide_ids_by_rootdir[root_dir].remove(sid)
                    args.skip_slides.append(sid)
                    continue
                semifp = root_dir / sid / 'im3' / f'{MEANIMAGE_SUBDIR_NAME}' / f'{sid}-std_error_of_mean_image.bin'
                if not semifp.is_file() :
                    logger.warning(f'WARNING: Standard error of mean image file does not exist for slide {sid}, will skip this slide!')
                    slide_ids_by_rootdir[root_dir].remove(sid)
                    args.skip_slides.append(sid)
                    continue
            if len(slide_ids_by_rootdir[root_dir])<1 :
                logger.warning(f'WARNING: no valid slides remain from root dir {root_dir}, will skip it!')
                args.root_dirs.remove(root_dir)
        if len(args.root_dirs)<1 :
            raise RuntimeError('ERROR: no valid slides to run on!')
    #bounds have to be coherent
    if len(args.bounds)!=2 or args.bounds[0]>=args.bounds[1] :
        raise ValueError(f'ERROR: bounds argument was given as {args.bounds} but expected lower_bound,upper_bound!')
    #create the working directory if it doesn't already exist
    wdp = pathlib.Path(args.workingdir)
    if not wdp.is_dir() :
        pathlib.Path.mkdir(wdp)
    #add the log file in the working directory to the logger handlers
    file_handler = logging.FileHandler(wdp/'flatfield_consistency.log')
    file_handler.setFormatter(logging_formatter)
    logger.addHandler(file_handler)

#helper function to normalize an image layer by its weighted mean 
def normalize_image_layer(mil,semil) :
    weights = 1./(semil**2)
    weighted_mil = weights*mil
    sum_weighted_mil = np.sum(weighted_mil)
    sum_weights = np.sum(weights)
    mil_mean = sum_weighted_mil/sum_weights
    #mil_mean = np.average(mil,weights=weights)
    return mil/mil_mean, semil/mil_mean

#helper function to get the standard deviation of the delta/sigma distribution for every layer of two meanimages compared to one another
def get_delta_over_sigma_std_devs_by_layer(dims,layers,mi1,semi1,mi2,semi2) :
    delta_over_sigma_std_devs = []
    for ln in range(1,dims[-1]+1) :
        if ln not in layers :
            delta_over_sigma_std_devs.append(0.)
            continue
        mil1 = mi1[:,:,ln-1]; semil1 = semi1[:,:,ln-1]
        mil2 = mi2[:,:,ln-1]; semil2 = semi2[:,:,ln-1]
        mil1max=np.max(mil1); mil1min=np.min(mil1)
        mil2max=np.max(mil2); mil2min=np.min(mil2)
        semil1min = np.min(semil1); semil2min = np.min(semil2)
        if mil1max==mil1min or mil2max==mil2min : #or semil1min==0. or semil2min==0. :
            delta_over_sigma_std_devs.append(0.)
            continue
        #normalize the image layers by the mean of the mean image layer
        mil1,semil1 = normalize_image_layer(mil1,semil1)
        mil2,semil2 = normalize_image_layer(mil2,semil2)
        #make the delta/sigma image
        delta_over_sigma = np.zeros_like(mil1)
        delta_over_sigma[(mil1!=0) & (mil2!=0)] = (mil1[(mil1!=0) & (mil2!=0)]-mil2[(mil1!=0) & (mil2!=0)])/(np.sqrt(semil1[(mil1!=0) & (mil2!=0)]**2+semil2[(mil1!=0) & (mil2!=0)]**2))
        std_dev = np.std(delta_over_sigma[delta_over_sigma!=0])
        delta_over_sigma_std_devs.append(std_dev)
    return delta_over_sigma_std_devs

#helper function to make a single comparison plot of some type
def make_and_save_single_plot(slide_ids,values_to_plot,plot_title,figname,workingdir,lines_after,bounds) :
    #make the figure
    fig,ax = plt.subplots(figsize=(1.*len(slide_ids),1.*len(slide_ids)))
    #figure out the scaled font sizes
    scaled_label_font_size = 10.*(1.+math.log10(len(slide_ids)/5.)) if len(slide_ids)>5 else 10.
    scaled_title_font_size = 10.*(1.+math.log2(len(slide_ids)/6.)) if len(slide_ids)>5 else 10.
    #add the grid to the plot
    pos = ax.imshow(values_to_plot,vmin=bounds[0],vmax=bounds[1])
    #add other patches
    patches = []
    #black out any zero values in the plot
    for iy in range(values_to_plot.shape[0]) :
        for ix in range(values_to_plot.shape[1]) :
            if values_to_plot[iy,ix]==0. :
                patches.append(Rectangle((ix-0.5,iy-0.5),1,1,edgecolor='k',facecolor='k',fill=True))
    #add lines after certain slides
    if lines_after!=[''] and len(lines_after)>0 :
        for sid in lines_after :
            if sid not in slide_ids :
                raise RuntimeError(f'ERROR: requested to add a separator after slide {sid} but this slide will not be on the plot!')
            sindex = slide_ids.index(sid)
            patches.append(Rectangle((sindex+0.375,-0.5),0.25,len(slide_ids)+1,edgecolor='r',facecolor='r',fill=True))
            patches.append(Rectangle((-0.5,sindex+0.375),len(slide_ids)+1,0.25,edgecolor='r',facecolor='r',fill=True))
    for patch in patches :
        ax.add_patch(patch)
    #adjust some stuff on the plots
    ax.set_xticks(np.arange(len(slide_ids)))
    ax.set_yticks(np.arange(len(slide_ids)))
    ax.set_xticklabels(slide_ids,fontsize=scaled_label_font_size)
    ax.set_yticklabels(slide_ids,fontsize=scaled_label_font_size)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")
    #add the exact numerical values inside each box
    for i in range(len(slide_ids)):
        for j in range(len(slide_ids)):
            v = values_to_plot[i,j]
            if v!=0. :
                text = ax.text(j, i, f'{v:.02f}',ha="center", va="center", color="b")
                text.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='white'))
    #set the title, add the colorbar, etc.
    ax.set_title(plot_title,fontsize=1.1*scaled_title_font_size)
    cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
    cbar = fig.colorbar(pos,cax=cax)
    cbar.ax.tick_params(labelsize=scaled_title_font_size)
    #save the plot
    with cd(workingdir) :
        plt.savefig(figname)
        plt.close()
        cropAndOverwriteImage(figname)

#helper function to make the consistency check grid plot
def consistency_check_grid_plot(input_file,root_dirs,skip_slide_ids,workingdir,sort_by,lines_after,bounds,all_or_brightest,save_all_layers,flatw) :
    #set the meanimage subdirectory name
    if flatw :
        MEANIMAGE_SUBDIR_NAME = FLATW_MEANIMAGE_SUBDIR_NAME
    #make the dict of slide IDs by root dir
    if input_file is not None :
        slide_ids_by_rootdir = {}
        table_entries = readtable(pathlib.Path(input_file),TableEntry)
        for te in table_entries :
            if te.slide_ID_1 in skip_slide_ids :
                continue
            if te.root_dir_1 not in slide_ids_by_rootdir.keys() :
                slide_ids_by_rootdir[te.root_dir_1] = []
            if te.slide_ID_1 not in slide_ids_by_rootdir[te.root_dir_1] :
                slide_ids_by_rootdir[te.root_dir_1].append(te.slide_ID_1)
    else :
        slide_ids_by_rootdir = get_slide_ids_by_root_dir(root_dirs,skip_slide_ids)
    #get the dimensions of the images (and make sure they all match)
    dims = None 
    for root_dir,slide_ids in slide_ids_by_rootdir.items() :
        for sid in slide_ids :
            this_slide_dims = getImageHWLFromXMLFile(root_dir,sid)
            if dims is None :
                dims = this_slide_dims
            if this_slide_dims!=dims :
                raise RuntimeError('ERROR: dimensions of all slides used must match!')
    #make a list of all the slide IDs, removing any slides with meaningless meanimages/standard errors
    logger.info('Finding slides to use...')
    slide_ids = []
    if input_file is not None :
        logger.info(f'Will use slides from input file {input_file}')
        for root_dir,sids in slide_ids_by_rootdir.items() :
            slide_ids+=sids
    else :
        for root_dir,sids in slide_ids_by_rootdir.items() :
            si = 0
            while si in range(len(sids)) :
                sid=sids[si]
                logger.info(f'\tChecking {sid}...')
                mi   = getRawAsHWL((root_dir / sid / 'im3' / f'{MEANIMAGE_SUBDIR_NAME}' / f'{sid}-mean_image.bin'),*(dims),np.float64)
                semi = getRawAsHWL((pathlib.Path(root_dir) / sid / 'im3' / f'{MEANIMAGE_SUBDIR_NAME}' / f'{sid}-std_error_of_mean_image.bin'),*(dims),np.float64)
                if np.min(mi)==np.max(mi) or np.max(semi)==0. : #or sum([np.min(semi[:,:,li])==0. for li in range(dims[-1])])==dims[-1] :
                    logger.warning(f'WARNING: slide {sid} will be skipped because not enough images were stacked!')
                    slide_ids_by_rootdir[root_dir].remove(sid)
                else :
                    slide_ids.append(sid)
                    si+=1
    #if requested, sort the list of slide IDs
    if sort_by!='order' :
        logger.info(f'Sorting slide IDs by {sort_by}...')
        slide_ids, new_lines_after = sort_slide_ids(slide_ids,slide_ids_by_rootdir,sort_by)
        if lines_after==[''] :
            lines_after=new_lines_after
    #start up an array to hold all of the necessary values and a list of table entries
    if all_or_brightest=='all' :
        layers = list(range(1,dims[-1]+1))
    elif all_or_brightest=='brightest' :
        layers = UNIV_CONST.BRIGHTEST_LAYERS_35 if dims[-1]==35 else UNIV_CONST.BRIGHTEST_LAYERS_43
    dos_std_dev_plot_values = np.zeros((len(slide_ids),len(slide_ids),dims[-1]))
    if input_file is not None :
        logger.info('Getting plot values from input file...')
        for te in table_entries :
            if (te.slide_ID_1 not in slide_ids) or (te.slide_ID_2 not in slide_ids) or (te.layer_n not in layers) :
                continue
            is1 = slide_ids.index(te.slide_ID_1)
            is2 = slide_ids.index(te.slide_ID_2)
            dos_std_dev_plot_values[is1,is2,te.layer_n-1] = te.delta_over_sigma_std_dev
            dos_std_dev_plot_values[is2,is1,te.layer_n-1] = te.delta_over_sigma_std_dev
    else :
        #for each possible pair of slide ids, find the standard deviation in each image layer of the delta/sigma
        output_table_entries = []
        pairs_done = set()
        for is1,sid1 in enumerate(slide_ids) :
            s1rd = ([rd for rd,sids in slide_ids_by_rootdir.items() if sid1 in sids])[0]
            mi1   = getRawAsHWL((s1rd / sid1 / 'im3' / f'{MEANIMAGE_SUBDIR_NAME}' / f'{sid1}-mean_image.bin'),
                                *(dims),np.float64)
            semi1 = getRawAsHWL((s1rd / sid1 / 'im3' / f'{MEANIMAGE_SUBDIR_NAME}' / f'{sid1}-std_error_of_mean_image.bin'),
                                *(dims),np.float64)
            for is2,sid2 in enumerate(slide_ids) :
                if sid2==sid1 :
                    dos_std_dev_plot_values[is1,is2,:] = 0.
                    continue 
                elif (sid2,sid1) in pairs_done :
                    for li in range(dims[-1]) :
                        dos_std_dev_plot_values[is1,is2,li] = dos_std_dev_plot_values[is2,is1,li]
                    continue
                logger.info(f'Finding std. devs. of delta/sigma for {sid1} vs. {sid2}...')
                s2rd = ([rd for rd,sids in slide_ids_by_rootdir.items() if sid2 in sids])[0]
                mi2   = getRawAsHWL((s2rd / sid2 / 'im3' / f'{MEANIMAGE_SUBDIR_NAME}' / f'{sid2}-mean_image.bin'),
                                    *(dims),np.float64)
                semi2 = getRawAsHWL((s2rd / sid2 / 'im3' / f'{MEANIMAGE_SUBDIR_NAME}' / f'{sid2}-std_error_of_mean_image.bin'),
                                    *(dims),np.float64)
                dossd_list = get_delta_over_sigma_std_devs_by_layer(dims,layers,mi1,semi1,mi2,semi2)
                dos_std_dev_plot_values[is1,is2,:] = dossd_list
                for ln in layers :
                    output_table_entries.append(TableEntry(s1rd,sid1,s2rd,sid2,ln,dos_std_dev_plot_values[is1,is2,ln-1]))
                pairs_done.add((sid1,sid2))
        with cd(workingdir) :
            writetable('meanimage_comparison_table.csv',output_table_entries)
    #for each image layer, plot a grid of the delta/sigma comparisons
    if save_all_layers :
        for ln in layers : 
            logger.info(f'Saving plot for layer {ln}...')
            make_and_save_single_plot(slide_ids,
                                      dos_std_dev_plot_values[:,:,ln-1],
                                      f'mean image delta/sigma std. devs. in layer {ln}',
                                      f'meanimage_comparison_layer_{ln}.png',
                                      workingdir,
                                      lines_after,
                                      bounds)
    #save a plot of the average over all considered layers
    logger.info(f'Saving plot of values averaged over {all_or_brightest} layers...')
    average_values = np.zeros_like(dos_std_dev_plot_values[:,:,0])
    for i in range(len(slide_ids)) :
        for j in range(len(slide_ids)) :
            num = 0; den = 0
            for li in range(dims[-1]) :
                if dos_std_dev_plot_values[i,j,li]!=0. :
                    num+=dos_std_dev_plot_values[i,j,li]
                    den+=1
            if den==0 :
                average_values[i,j]=0.
            else :
                average_values[i,j]=num/den
    make_and_save_single_plot(slide_ids,
                              average_values,
                              f'avg. mean image delta/sigma std. devs. in {all_or_brightest} layers',
                              f'meanimage_comparison_average_over_{all_or_brightest}_layers.png',
                              workingdir,
                              lines_after,
                              bounds)

#################### MAIN SCRIPT ####################

def main(args=None) :
    parser = ArgumentParser()
    # a group for defining the slides to use either from a prior run or from a sampledef.csv file
    slide_def_group = parser.add_mutually_exclusive_group(required=True)
    slide_def_group.add_argument('--input_file', 
                        help='Path to a .csv file previously generated by this code that contains TableEntry objects (use to quickly remake plots of prior runs)')
    slide_def_group.add_argument('--root_dirs', type=split_csv_to_list, default='',
                        help='Comma-separated list of paths to directories with [slideID]/im3/meanimage subdirectories and sampledef.csv files in them')
    #other optional arguments
    parser.add_argument('--skip_slides', type=split_csv_to_list, default='',
                        help='Comma-separated list of slides to skip')
    parser.add_argument('--workingdir', 
                        help='Path to the working directory where results will be saved')
    parser.add_argument('--sort_by', choices=['order','project_cohort_batch'], default='project_cohort_batch',
                        help='how to sort the order of the slide IDs')
    parser.add_argument('--lines_after', type=split_csv_to_list, default='',
                        help="""Comma-separated list of slides to put separating lines after on the plot 
                                (default is project/cohort/batch transitions, adding this argument will overwrite those automatic options)""")
    parser.add_argument('--bounds', type=split_csv_to_list_of_floats, default='0.85,1.15',
                        help='Hard limits to the imshow scale for the plot (given as lower_bound,upper_bound')
    parser.add_argument('--all_or_brightest', choices=['all','brightest'], default='all',
                        help='Whether to make plots and sum values over all image layers or just the brightest')
    parser.add_argument('--save_all_layers', action='store_true',
                        help='Add this flag to save the plots of the individual layers, not just the average over all of them')
    parser.add_argument('--flatw', action='store_true',
                        help='Add this flag to use meanimages from "meanimage_from_fw_files" instead of "meanimage" subdirectories')
    args = parser.parse_args(args=args)
    #check the arguments
    checkArgs(args)
    #run the main workhorse function
    consistency_check_grid_plot(args.input_file,args.root_dirs,args.skip_slides,args.workingdir,
                                args.sort_by,args.lines_after,args.bounds,args.all_or_brightest,args.save_all_layers,args.flatw)
    logger.info('Done : )')

if __name__=='__main__' :
    main()