#imports
from astropath_calibration.warping.alignmentset import AlignmentSetForWarping 
from astropath_calibration.alignment.alignmentset import AlignmentSetComponentTiffFromXML
from astropath_calibration.utilities.img_file_io import getMedianExposureTimesAndCorrectionOffsetsForSlide
from astropath_calibration.utilities.tableio import writetable
from astropath_calibration.utilities import units
from astropath_calibration.utilities.dataclasses import MyDataClass
from astropath_calibration.utilities.misc import cd, addCommonArgumentsToParser, cropAndOverwriteImage
from astropath_calibration.utilities.config import CONST as UNIV_CONST
from matplotlib import colors
from argparse import ArgumentParser
import numpy as np, matplotlib.pyplot as plt
import logging, os
units.setup('fast')

#constants
SEP_CUT = 0.5

#logger
logger = logging.getLogger("overexposed_hpfs")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s  [%(funcName)s]","%Y-%m-%d %H:%M:%S"))
logger.addHandler(handler)

#dataclass to hold a rectangle's mse comparison information
class OverexposedHPFInfo(MyDataClass) :
    file : str
    rect_n : int
    x : float
    y : float
    dapi_af_sep : float

#dataclass to hold an overlap's mse comparison information
class OverlapMSEComparisonInfo(MyDataClass) :
    olap_n : int
    olap_tag : int
    p1_rect_n : int 
    p2_rect_n : int 
    n_pix   : int
    p1_file : str = '' 
    p1_x : float = -1. 
    p1_y : float = -1. 
    dapi_mse1 : float = -1.
    dapi_mse2 : float = -1.
    af_mse1 : float = -1.
    af_mse2 : float = -1.

#helper function to make sure all the necessary information is available from the command line arguments
def checkArgs(args) :
    #root dir must exist
    if not os.path.isdir(args.root_dir) :
        raise ValueError(f'ERROR: root_dir argument ({args.root_dir}) does not point to a valid directory!')
    if not os.path.isdir(os.path.join(args.root_dir,args.slideID)) :
        raise ValueError(f'ERROR: root_dir {args.root_dir} does not have a subdirectory for slide {args.slideID}!')
    #create the working directory if it doesn't already exist
    if not os.path.isdir(args.workingdir) :
        os.mkdir(args.workingdir)

#helper function to make plots of the separation values and which rectangles were flagged
def plotRectangleInfo(sid,rects) :
    xs = np.array([r['x'] for r in rects.values()])
    ys = np.array([r['y'] for r in rects.values()])
    w = np.max(xs)-np.min(xs); h = np.max(ys)-np.min(ys)
    if h>w :
        f = plt.figure(figsize=(2*((1.1*w)/(1.1*h))*9.6,2*9.6))
    else :
        f = plt.figure(figsize=(2*9.6,2*9.6*((1.1*h)/(1.1*w))))
    ax1 = f.add_subplot(2,2,1)
    ax1.set_facecolor('gray')
    pos = ax1.scatter([r['x'] for r in rects.values()],[r['y'] for r in rects.values()],marker='o',
                       c=[r['rel_diff_dev']for r in rects.values()],cmap='gist_ncar')
    ax1.invert_yaxis()
    f.colorbar(pos,ax=ax1)
    ax1.set_title(f'{sid} mean overlap rel. mse diff. sep. btwn layers per pixel',fontsize=16)
    ax2 = f.add_subplot(2,2,2)
    ax2.scatter([r['x'] for r in rects.values()],
                [r['y'] for r in rects.values()],
                marker='o',
                c='gray',label='not flagged')
    flagged_rs = [r for r in rects.values() if r['rel_diff_dev']>=SEP_CUT]
    ax2.scatter([r['x'] for r in correctly_flagged_rs],
                  [r['y'] for r in correctly_flagged_rs],
                  marker='o',
                  c='tab:red',label='flagged')
    ax2.invert_yaxis()
    ax2.legend(loc='best')
    ax2.set_title(f'{sid} HPFs of interest',fontsize=16)
    ax3 = f.add_subplot(2,1,3)
    ax3.hist([r['rel_diff_dev'] for r in rects.values()],100)
    ax3.plot([SEP_CUT,SEP_CUT],
            [0.8*y for y in ax3.get_ylim()],linewidth=3,alpha=0.8,color='tab:red',label=f"cut at {SEP_CUT}")
    ax3.legend(loc='best')
    ax3.set_title(f'{sid} mean overlap rel. mse diff. sep. btwn DAPI and AF layers per pixel')
    fn = f'{sid}_hpf_plots.png'
    plt.savefig(fn)
    cropAndOverwriteImage(fn)

#helper function to return the mean separation between the overlap mse relative differences in the DAPI and AF layers
def getRectangleRelativeDifferenceSeparation(rn,this_rect_olaps) :
    sum_rel_diff_seps = 0.; sum_weights = 0.
    for olap in this_rect_olaps :
        mse1=olap.dapi_mse1; mse2=olap.dapi_mse2
        omse1=olap.af_mse1; omse2=olap.af_mse2
        if mse1==0 or omse1==0 :
            msg=f'WARNING: rectangle {rn} (x={olap.p1_x:.1f}, y={olap.p1_y:.1f}) overlap {olap.olap_tag} '
            msg+=f'has DAPI mse1={mse1:.1f}, AF mse1={omse1:.1f}; '
            msg+=f'will be skipped!'
            logger.warn(msg)
            continue
        dapi_rel_diff = (mse1-mse2)/mse1
        af_rel_diff = (omse1-omse2)/omse1
        w=olap.n_pix
        sum_rel_diff_seps+=w*(dapi_rel_diff-af_rel_diff)
        sum_weights+=w
    return sum_rel_diff_seps/sum_weights

#helper function to get the list of overlap MSE comparison info objects
def getOverlapMSEComparisonDict(rd,sid) :
    #start the dictionary of all the overlap MSE comparison info objects keyed by overlap n
    all_olap_mse_infos = {}
    #get overlap mse information from the slide's component .tiff images in the DAPI and autofluorescence layers
    for lgi,ln in enumerate((UNIV_CONST.COMP_TIFF_DAPI_LAYER,UNIV_CONST.COMP_TIFF_AF_LAYER)) :
        logger.info(f'Getting overlap mses for component .tiff image layer {ln}....')
        a = AlignmentSetComponentTiffFromXML(rd,sid,
                                             nclip=UNIV_CONST.N_CLIP,
                                             layer=ln)
        a.logger.setLevel(logging.WARN)
        a.align()
        #add this layer's information to each overlap
        for olap in a.overlaps :
            if olap.n not in all_olap_mse_infos.keys() :
                im1, im2 = olap.shifted
                n_pix = int(0.5*(im1.shape[0]+im2.shape[0]))*int(0.5*(im1.shape[1]+im2.shape[1]))
                all_olap_mse_infos[olap.n] = OverlapMSEComparisonInfo(olap.n,olap.tag,olap.p1,olap.p2,n_pix)
            if olap.result is None :
                continue
            mse1 = olap.result.mse1
            mse2 = olap.result.mse2
            if lgi==0 :
                all_olap_mse_infos[olap.n].dapi_mse1 = mse1
                all_olap_mse_infos[olap.n].dapi_mse2 = mse2
            elif lgi==1 :
                all_olap_mse_infos[olap.n].af_mse1 = mse1
                all_olap_mse_infos[olap.n].af_mse2 = mse2
        #add the information for the rectangles (only once)
        if lgi==0 :
            for rect in a.rectangles :
                for olap_n in all_olap_mse_infos.keys() :
                    if all_olap_mse_infos[olap_n].p1_rect_n==rect.n :
                        all_olap_mse_infos[olap_n].p1_file=rect.file
                        all_olap_mse_infos[olap_n].p1_x = rect.cx
                        all_olap_mse_infos[olap_n].p1_y = rect.cy
    return all_olap_mse_infos

#helper function to write out a bunch of overlaps' mse values in several different layers using raw and component tiff files
def findOverexposedHPFs(rd,sid,workingdir) :
    #get the overlap MSE comparison info objects
    overlaps = getOverlapMSEComparisonDict(rd,sid)
    #find the mean relative difference separation for every rectangle 
    rects = {}
    for rn in set([olap.p1_rect_n for olap in overlaps]) :
        this_rect_olaps = [olap for olap in overlaps if olap.p1_rect_n==rn]
        rel_diff_dev = getRectangleRelativeDifferenceSeparation(rn,this_rect_olaps)
        rects[rn] = {'file':this_rect_olaps[0].p1_file,'n':rn,'x':this_rect_olaps[0].p1_x,'y':this_rect_olaps[0].p1_y,'rel_diff_dev':rel_diff_dev}
    #iterate, removing flagged HPFs, until no new ones get added
    ii = 1
    previously_flagged_rect_ns = []
    flagged_rect_ns = [rn for rn,r in rects.items() if r['rel_diff_dev']>=SEP_CUT]
    while len(flagged_rect_ns)!=len(previously_flagged_rect_ns) :
        logger.info(f'iteration {ii}: {len(flagged_rect_ns)} HPFs flagged out of {len(rects)}')
        for frn in flagged_rect_ns :
            flagged_rect_olaps = [olap for olap in overlaps if olap.p1_rect_n==frn]
            rects_to_adjust = [rn for rn,r in rects.items() if 
                               rn in set([o.p2_rect_n for o in flagged_rect_olaps])
                               and rn not in flagged_rect_ns]
            for rn in rects_to_adjust :
                this_rect_olaps = [olap for olap in overlaps if olap.p1_rect_n==rn and olap.p2_rect_n not in flagged_rect_ns]
                if len(this_rect_olaps)!=0 :
                    rel_diff_dev = getRectangleRelativeDifferenceSeparation(rn,this_rect_olaps)
                    rects[rn]['rel_diff_dev'] = rel_diff_dev
        previously_flagged_rect_ns = flagged_rect_ns
        flagged_rect_ns = [rn for rn,r in rects.items() if r['rel_diff_dev']>=SEP_CUT]
        ii+=1
    #make and write out the list of flagged rectangle info objects
    overexposed_hpf_infos = [OverexposedHPFInfo(r['file'],r['n'],r['x'],r['y'],r['rel_diff_dev']) for r in [rects[n] for in n flagged_rect_ns]]
    logger.info(f'Found {len(overexposed_hpf_infos)} total overexposed HPFs in {sid}')
    with cd(workingdir) :
        writetable(f'{sid}_overexposed_HPFs.csv',overexposed_hpf_infos)
    #make and write out the plot of all the rectangles
    with cd(workingdir) :
        plotRectangleInfo(sid,rects)

#################### MAIN SCRIPT ####################

def main(args=None) :
    parser = ArgumentParser()
    parser.add_argument('slideID',          help='Name of the slide to use')
    parser.add_argument('root_dir',         help='Path to the Clinical_Specimen directory with info for the given slide')
    parser.add_argument('workingdir',       help='Path to the working directory (will be created if necessary)')
    args = parser.parse_args(args=args)
    #check the arguments
    checkArgs(args)
    #run alignments and check for overexposed HPFs
    findOverexposedHPFs(args.root_dir,args.slideID,args.workingdir)
    logger.info('Done')

if __name__=='__main__' :
    main()
