#imports
from ...utilities.misc import cd, crop_and_overwrite_image
from ...utilities.config import CONST as UNIV_CONST
import matplotlib.pyplot as plt
import pathlib

def plot_tissue_edge_rectangle_locations(all_rects,edge_rects,root_dir,slideID,save_dirpath=None) :
    """
    Make and save the plot of the edge field locations next to the qptiff for reference

    all_rects    = list of all Rectangle objects to plot
    edge_rects   = list that is a subset of the above of any Rectangles that are on the edges of the tissue
    root_dir     = path to the root directory for the cohort the slide comes from
    slideID      = ID of the slide whose Rectangle locations are being plotted
    save_dirpath = path to directory to save the plot in (if None the plot is saved in the current directory)
    """
    #some constants
    SINGLE_FIG_SIZE = (9.6,7.2)
    FONTSIZE = 13.5
    FIGURE_NAME = f'{slideID}_rectangle_locations.png'
    #make and save the plot
    edge_rect_xs = [r.x for r in edge_rects]
    edge_rect_ys = [r.y for r in edge_rects]
    bulk_rect_xs = [r.x for r in all_rects if r not in edge_rects]
    bulk_rect_ys = [r.y for r in all_rects if r not in edge_rects]
    has_qptiff = (root_dir / f'{slideID}' / f'{UNIV_CONST.DBLOAD_DIR_NAME}' / f'{slideID}{UNIV_CONST.QPTIFF_SUFFIX}').is_file()
    if has_qptiff :
        f,(ax1,ax2) = plt.subplots(1,2,figsize=(2*SINGLE_FIG_SIZE[0],SINGLE_FIG_SIZE[1]))
    else :
        f,ax1 = plt.subplots(figsize=SINGLE_FIG_SIZE)
    ax1.scatter(edge_rect_xs,edge_rect_ys,marker='o',color='r',label='edges')
    ax1.scatter(bulk_rect_xs,bulk_rect_ys,marker='o',color='b',label='bulk')
    ax1.invert_yaxis()
    ax1.set_title(f'{slideID} rectangles, ({len(edge_rect_xs)} edge and {len(bulk_rect_xs)} bulk)',fontsize=FONTSIZE)
    ax1.legend(loc='best',fontsize=FONTSIZE)
    ax1.set_xlabel('x position',fontsize=FONTSIZE)
    ax1.set_ylabel('y position',fontsize=FONTSIZE)
    if has_qptiff :
        ax2.imshow(mpimg.imread(root_dir / f'{slideID}' / f'{UNIV_CONST.DBLOAD_DIR_NAME}' / f'{slideID}{UNIV_CONST.QPTIFF_SUFFIX}'))
        ax2.set_title('reference qptiff',fontsize=FONTSIZE)
    if save_dirpath is not None :
        if not save_dirpath.is_dir() :
            save_dirpath.mkdir()
        with cd(save_dirpath) :
            plt.savefig(FIGURE_NAME)
            plt.close()
            crop_and_overwrite_image(FIGURE_NAME)
    else :
        plt.savefig(FIGURE_NAME)
        plt.close()
        crop_and_overwrite_image(FIGURE_NAME)
