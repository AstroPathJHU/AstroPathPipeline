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
CUT_MEAN=0.10
CUT_SIGMA=5.0

#logger
logger = logging.getLogger("overexposed_hpfs")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s  [%(funcName)s]","%Y-%m-%d %H:%M:%S"))
logger.addHandler(handler)

#dataclass to hold a rectangle's mse comparison information
class RectangleMSEComparisonInfo(MyDataClass) :
    file : str
    n : int
    x : float
    y : float
    lg1_mean_abs_diff_dev : float = 0.
    lg1_mean_frac_diff_dev : float = 0.
    lg1_mean_frac_diff_dev_err : float = 0.
    lg2_mean_abs_diff_dev : float = 0.
    lg2_mean_frac_diff_dev : float = 0.
    lg2_mean_frac_diff_dev_err : float = 0.
    lg3_mean_abs_diff_dev : float = 0.
    lg3_mean_frac_diff_dev : float = 0.
    lg3_mean_frac_diff_dev_err : float = 0.
    lg4_mean_abs_diff_dev : float = 0.
    lg4_mean_frac_diff_dev : float = 0.
    lg4_mean_frac_diff_dev_err : float = 0.

#dataclass to hold an overlap's mse comparison information
class OverlapMSEComparisonInfo(MyDataClass) :
    olap_n : int
    olap_tag : int
    p1_rect_n : int 
    p2_rect_n : int
    p1_file : str = ''
    p1_x : float = -1.
    p1_y : float = -1.
    raw_dapi_mse1 : float = -1.
    raw_dapi_mse2 : float = -1.
    raw_lg2_mse1 : float = -1.
    raw_lg2_mse2 : float = -1.
    raw_lg3_mse1 : float = -1.
    raw_lg3_mse2 : float = -1.
    raw_lg4_mse1 : float = -1.
    raw_lg4_mse2 : float = -1.
    raw_lg5_mse1 : float = -1.
    raw_lg5_mse2 : float = -1.
    comp_tiff_dapi_mse1 : float = -1.
    comp_tiff_dapi_mse2 : float = -1.
    comp_tiff_af_mse1 : float = -1.
    comp_tiff_af_mse2 : float = -1.

#helper function to make sure all the necessary information is available from the command line arguments
def checkArgs(args) :
    #rawfile_top_dir/[slideID] must exist
    rawfile_dir = os.path.join(args.rawfile_top_dir,args.slideID)
    if not os.path.isdir(rawfile_dir) :
        raise ValueError(f'ERROR: rawfile directory {rawfile_dir} does not exist!')
    #root dir must exist
    if not os.path.isdir(args.root_dir) :
        raise ValueError(f'ERROR: root_dir argument ({args.root_dir}) does not point to a valid directory!')
    #images must be corrected for exposure time, and exposure time correction file must exist
    if (args.skip_exposure_time_correction) :   
        raise ValueError('ERROR: exposure time offset file must be provided.')
    if not os.path.isfile(args.exposure_time_offset_file) :
        raise ValueError(f'ERROR: exposure_time_offset_file {args.exposure_time_offset_file} does not exist!')
    #create the working directory if it doesn't already exist
    if not os.path.isdir(args.workingdir) :
        os.mkdir(args.workingdir)

#find/plot overexposed HPFs in a single image layer by comparing differences in overlap region mean squared fluxes
def findOverexposedHPFsForSlide(rtd,rd,sid,etof,workingdir) :
    #get the correction details
    med_ets, offsets = getMedianExposureTimesAndCorrectionOffsetsForSlide(rd,sid,etof)
    #align the slide's raw images in each of the brightest layers
    asets = []
    for ln in UNIV_CONST.BRIGHTEST_LAYERS_35 :
        logger.info(f'Getting overlap mses for image layer {ln}....')
        a = AlignmentSetForWarping(rd,rtd,sid,
                                   med_et=med_ets[ln-1],offset=offsets[ln-1],flatfield=None,nclip=UNIV_CONST.N_CLIP,readlayerfile=False,
                                   layer=ln,filetype='raw')
        a.logger.setLevel(logging.WARN)
        a.align(mseonly=True)
        asets.append(a)
        #plot the absolute and fractional mse differences for every overlap in this layer
        mse_diffs = []
        frac_mse_diffs = []
        for olap in a.overlaps :
            if olap.result is None :
                continue
            mse1 = olap.result.mse1
            mse2 = olap.result.mse2
            mse_diffs.append(mse2-mse1)
            frac_mse_diffs.append((mse2-mse1)/(0.5*(mse1+mse2)))
        f,ax=plt.subplots(1,3,figsize=(3*6.4,6.4))
        ax[0].hist(mse_diffs,bins=60,log=True)
        ax[0].set_title(f'{sid} absolute overlap mse differences layer {ln}')
        ax[0].set_xlabel('mse2-mse1')
        ax[1].hist(frac_mse_diffs,bins=60,log=True)
        ax[1].set_title(f'{sid} fractional overlap mse differences layer {ln}')
        ax[1].set_xlabel('(mse2-mse1)/(0.5*(mse2+mse1))')
        pos = ax[2].hist2d(frac_mse_diffs,mse_diffs,bins=60,norm=colors.LogNorm(),cmap='gray')
        f.colorbar(pos[3],ax=ax[2])
        ax[2].set_title(f'{sid} fractional vs. absolute mse differences layer {ln}')
        ax[2].set_xlabel('mse2-mse1')
        ax[2].set_ylabel('(mse2-mse1)/(0.5*(mse2+mse1))')
        fn = f'{sid}_overlap_mse_diffs_layer_{ln}.png'
        with cd(workingdir) :
            plt.savefig(fn)
            cropAndOverwriteImage(fn)
    #start a dictionary for all of the RectangleMSEComparisonInfo objects
    rectangle_info_objs = {}
    #start up the other sheet of plots
    xs = np.array([r.cx for r in asets[UNIV_CONST.DAPI_LAYER_GROUP_INDEX_35].rectangles])
    ys = np.array([r.cy for r in asets[UNIV_CONST.DAPI_LAYER_GROUP_INDEX_35].rectangles])
    w = np.max(xs)-np.min(xs)
    h = np.max(ys)-np.min(ys)
    if h>w :
        f,ax = plt.subplots(len(UNIV_CONST.BRIGHTEST_LAYERS_35)-1,5,figsize=(5*((1.1*w)/(1.1*h))*9.6,(len(UNIV_CONST.BRIGHTEST_LAYERS_35)-1)*9.6))
    else :
        f,ax = plt.subplots(len(UNIV_CONST.BRIGHTEST_LAYERS_35)-1,5,figsize=(5*9.6,(len(UNIV_CONST.BRIGHTEST_LAYERS_35)-1)*9.6*((1.1*h)/(1.1*w))))
    #for the DAPI layer compared to each other layer
    for lgi,ln in enumerate(UNIV_CONST.BRIGHTEST_LAYERS_35) :
        if lgi==UNIV_CONST.DAPI_LAYER_GROUP_INDEX_35 :
            continue
        logger.info(f'comparing overlap mse differences in layers {UNIV_CONST.BRIGHTEST_LAYERS_35[UNIV_CONST.DAPI_LAYER_GROUP_INDEX_35]} and {ln}')
        #plot the mean absolute and fractional mse difference ratios (neglecting corner overlaps and the max differences) 
        #for each HPF and show which would be flagged with the cut applied
        to_plot = []
        for r in asets[UNIV_CONST.DAPI_LAYER_GROUP_INDEX_35].rectangles :
            this_rect_olaps_1 = [o for o in asets[UNIV_CONST.DAPI_LAYER_GROUP_INDEX_35].overlaps if o.p1==r.n and o.tag not in (1,3,7,9) and o.result is not None]
            this_rect_olaps_2 = [o for o in asets[lgi].overlaps if o.p1==r.n and o.tag not in (1,3,7,9) and o.result is not None]
            if len(this_rect_olaps_1)<2 or len(this_rect_olaps_2)<2 or len(this_rect_olaps_1)!=len(this_rect_olaps_2) :
                continue
            abs_diff_devs = []
            frac_diff_devs = []
            for o1,o2 in zip(this_rect_olaps_1,this_rect_olaps_2) :
                r11 = o1.result.mse1
                r12 = o1.result.mse2
                r21 = o2.result.mse1
                r22 = o2.result.mse2
                #abs_diff_devs.append(abs(abs(r12-r11)-abs(r22-r21)))
                #frac_diff_devs.append(abs(abs((r12-r11)/(0.5*(r12+r11)))-abs((r22-r21)/(0.5*(r22+r21)))))
                abs_diff_devs.append(abs((r12-r11)-(r22-r21)))
                frac_diff_devs.append(abs(((r12-r11)/(0.5*(r12+r11)))-((r22-r21)/(0.5*(r22+r21)))))
            abs_diff_devs = np.array(abs_diff_devs)
            frac_diff_devs = np.array(frac_diff_devs)
            mean_abs_diff_dev = np.mean(abs_diff_devs)
            mean_frac_diff_dev = np.mean(frac_diff_devs)
            mean_frac_diff_dev_err = np.std(frac_diff_devs)/np.sqrt(len(frac_diff_devs)) if len(frac_diff_devs)>1 else 0
            if r.n not in rectangle_info_objs.keys() :
                rectangle_info_objs[r.n] = RectangleMSEComparisonInfo(r.file,r.n,r.cx,r.cy)
            if lgi==1 :
                rectangle_info_objs[r.n].lg1_mean_abs_diff_dev      = mean_abs_diff_dev
                rectangle_info_objs[r.n].lg1_mean_frac_diff_dev     = mean_frac_diff_dev
                rectangle_info_objs[r.n].lg1_mean_frac_diff_dev_err = mean_frac_diff_dev_err
            elif lgi==2 :
                rectangle_info_objs[r.n].lg2_mean_abs_diff_dev      = mean_abs_diff_dev
                rectangle_info_objs[r.n].lg2_mean_frac_diff_dev     = mean_frac_diff_dev
                rectangle_info_objs[r.n].lg2_mean_frac_diff_dev_err = mean_frac_diff_dev_err
            elif lgi==3 :
                rectangle_info_objs[r.n].lg3_mean_abs_diff_dev      = mean_abs_diff_dev
                rectangle_info_objs[r.n].lg3_mean_frac_diff_dev     = mean_frac_diff_dev
                rectangle_info_objs[r.n].lg3_mean_frac_diff_dev_err = mean_frac_diff_dev_err
            elif lgi==4 :
                rectangle_info_objs[r.n].lg4_mean_abs_diff_dev      = mean_abs_diff_dev
                rectangle_info_objs[r.n].lg4_mean_frac_diff_dev     = mean_frac_diff_dev
                rectangle_info_objs[r.n].lg4_mean_frac_diff_dev_err = mean_frac_diff_dev_err
            to_plot.append({'x':r.cx,
                            'y':r.cy,
                            'n':r.n,
                            'f':r.file,
                            'abs':mean_abs_diff_dev,
                            'frac':mean_frac_diff_dev,
                            'frac_err':mean_frac_diff_dev_err,
                            'sigma_diff':(mean_frac_diff_dev-CUT_MEAN)/mean_frac_diff_dev_err if mean_frac_diff_dev_err!=0 else 0.})
        for p in to_plot :
            if (p['frac_err']!=0. and p['sigma_diff']>=CUT_SIGMA) or (p['frac_err']==0. and p['frac']>=CUT_MEAN) :
                msg=f"rectangle # {p['n']} (file {p['f']}) flagged using layer {ln} with mean abs. "
                msg+=f"diff dev. {p['abs']:.2f}, mean frac. diff dev {p['frac']:.6f} +/- {p['frac_err']:.6f} "
                msg+=f"({p['sigma_diff']} sigma from mean frac. diff. dev. cut at {CUT_MEAN})"
                logger.info(msg)
        pos = ax[lgi-1][0].scatter([p['x'] for p in to_plot],
                                   [p['y'] for p in to_plot],
                                   marker='o',
                                   c=[p['abs'] for p in to_plot],cmap='plasma')
        ax[lgi-1][0].invert_yaxis()
        f.colorbar(pos,ax=ax[lgi-1][0])
        ax[lgi-1][0].set_title(f'{sid} HPF abs. mse diff. devs btwn layers {UNIV_CONST.BRIGHTEST_LAYERS_35[UNIV_CONST.DAPI_LAYER_GROUP_INDEX_35]} and {ln}',fontsize=16)
        pos = ax[lgi-1][1].scatter([p['x'] for p in to_plot],
                                   [p['y'] for p in to_plot],
                                   marker='o',
                                   c=[p['frac'] for p in to_plot],cmap='plasma')
        ax[lgi-1][1].invert_yaxis()
        f.colorbar(pos,ax=ax[lgi-1][1])
        ax[lgi-1][1].set_title(f'{sid} HPF frac. mse diff. devs btwn layers {UNIV_CONST.BRIGHTEST_LAYERS_35[UNIV_CONST.DAPI_LAYER_GROUP_INDEX_35]} and {ln}',fontsize=16)
        pos = ax[lgi-1][2].scatter([p['x'] for p in to_plot],
                                   [p['y'] for p in to_plot],
                                   marker='o',
                                   c=[p['frac_err'] for p in to_plot],cmap='plasma')
        ax[lgi-1][2].invert_yaxis()
        f.colorbar(pos,ax=ax[lgi-1][2])
        ax[lgi-1][2].set_title(f'{sid} HPF frac. mse diff. dev. err. btwn layers {UNIV_CONST.BRIGHTEST_LAYERS_35[UNIV_CONST.DAPI_LAYER_GROUP_INDEX_35]} and {ln}',fontsize=16)
        pos = ax[lgi-1][3].scatter([p['x'] for p in to_plot],
                                   [p['y'] for p in to_plot],
                                   marker='o',
                                   c=[p['sigma_diff'] for p in to_plot],cmap='plasma')
        ax[lgi-1][3].invert_yaxis()
        f.colorbar(pos,ax=ax[lgi-1][3])
        ax[lgi-1][3].set_title(f'{sid} HPF frac. mse diff. dev. sigma from cut btwn layers {UNIV_CONST.BRIGHTEST_LAYERS_35[UNIV_CONST.DAPI_LAYER_GROUP_INDEX_35]} and {ln}',fontsize=16)
        ax[lgi-1][4].scatter([p['x'] for p in to_plot if (p['frac_err']!=0. and p['sigma_diff']<CUT_SIGMA) or (p['frac_err']==0. and p['frac']<CUT_MEAN)],
                             [p['y'] for p in to_plot if (p['frac_err']!=0. and p['sigma_diff']<CUT_SIGMA) or (p['frac_err']==0. and p['frac']<CUT_MEAN)],
                             marker='o',
                             c='gray',
                             label='not flagged')
        ax[lgi-1][4].scatter([p['x'] for p in to_plot if (p['frac_err']!=0. and p['sigma_diff']>=CUT_SIGMA) or (p['frac_err']==0. and p['frac']>=CUT_MEAN)],
                             [p['y'] for p in to_plot if (p['frac_err']!=0. and p['sigma_diff']>=CUT_SIGMA) or (p['frac_err']==0. and p['frac']>=CUT_MEAN)],
                             marker='o',
                             c='tab:red',
                             label='flagged')
        ax[lgi-1][4].invert_yaxis()
        ax[lgi-1][4].set_title(f'{sid} HPFs flagged using layer {ln}',fontsize=16)
        ax[lgi-1][4].legend(loc='best')
    fn=f'{sid}_overexposed_hpf_locations.png'
    with cd(workingdir) :
        plt.savefig(fn)
        cropAndOverwriteImage(fn)
        writetable(f'{sid}_rectangle_mse_comparison_info.csv',rectangle_info_objs.values())

#helper function to write out a bunch of overlaps' mse values in several different layers using raw and component tiff files
def writeOverlapMSETable(rtd,rd,sid,etof,workingdir) :
    #start the dictionary of all the overlap mse info objects keyed by overlap n
    all_olap_mse_infos = {}
    #get the correction details
    med_ets, offsets = getMedianExposureTimesAndCorrectionOffsetsForSlide(rd,sid,etof)
    #get overlap mse information from the slide's raw images in each of the brightest layers
    for lgi,ln in enumerate(UNIV_CONST.BRIGHTEST_LAYERS_35) :
        logger.info(f'Getting overlap mses for raw image layer {ln}....')
        a = AlignmentSetForWarping(rd,rtd,sid,
                                   med_et=med_ets[ln-1],offset=offsets[ln-1],flatfield=None,nclip=UNIV_CONST.N_CLIP,readlayerfile=False,
                                   layer=ln,filetype='raw')
        a.logger.setLevel(logging.WARN)
        a.align(mseonly=True)
        #add this layer's information to each overlap
        for olap in a.overlaps :
            if olap.n not in all_olap_mse_infos.keys() :
                all_olap_mse_infos[olap.n] = OverlapMSEComparisonInfo(olap.n,olap.p1,olap.p2)
            if olap.result is None :
                continue
            mse1 = olap.result.mse1
            mse2 = olap.result.mse2
            if lgi==0 :
                all_olap_mse_infos[olap.n].raw_dapi_mse1 = mse1
                all_olap_mse_infos[olap.n].raw_dapi_mse2 = mse2
            elif lgi==1 :
                all_olap_mse_infos[olap.n].raw_lg2_mse1 = mse1
                all_olap_mse_infos[olap.n].raw_lg2_mse2 = mse2
            elif lgi==2 :
                all_olap_mse_infos[olap.n].raw_lg3_mse1 = mse1
                all_olap_mse_infos[olap.n].raw_lg3_mse2 = mse2
            elif lgi==3 :
                all_olap_mse_infos[olap.n].raw_lg4_mse1 = mse1
                all_olap_mse_infos[olap.n].raw_lg4_mse2 = mse2
            elif lgi==4 :
                all_olap_mse_infos[olap.n].raw_lg5_mse1 = mse1
                all_olap_mse_infos[olap.n].raw_lg5_mse2 = mse2
        if lgi==0 :
            for rect in a.rectangles :
                for olap_n in all_olap_mse_infos.keys() :
                    if all_olap_mse_infos[olap_n].p1_rect_n==rect.n :
                        all_olap_mse_infos[olap_n].p1_file=rect.file
                        all_olap_mse_infos[olap_n].p1_x = rect.cx
                        all_olap_mse_infos[olap_n].p1_y = rect.cy
    #get overlap mse information from the slide's component .tiff images in the DAPI and autofluorescence layers
    for lgi,ln in enumerate((1,8)) :
        logger.info(f'Getting overlap mses for component .tiff image layer {ln}....')
        a = AlignmentSetComponentTiffFromXML(rd,sid,
                                             nclip=UNIV_CONST.N_CLIP,
                                             layer=ln)
        a.logger.setLevel(logging.WARN)
        a.align(mseonly=True)
        #add this layer's information to each overlap
        for olap in a.overlaps :
            if olap.n not in all_olap_mse_infos.keys() :
                all_olap_mse_infos[olap.n] = OverlapMSEComparisonInfo(olap.n,olap.p1,olap.p2)
            if olap.result is None :
                continue
            mse1 = olap.result.mse1
            mse2 = olap.result.mse2
            if lgi==0 :
                all_olap_mse_infos[olap.n].comp_tiff_dapi_mse1 = mse1
                all_olap_mse_infos[olap.n].comp_tiff_dapi_mse2 = mse2
            elif lgi==1 :
                all_olap_mse_infos[olap.n].comp_tiff_af_mse1 = mse1
                all_olap_mse_infos[olap.n].comp_tiff_af_mse2 = mse2
    #write out the final table
    with cd(workingdir) :
        writetable(f'{sid}_overlap_mse_comparison_info.csv',all_olap_mse_infos.values())

#################### MAIN SCRIPT ####################

def main(args=None) :
    parser = ArgumentParser()
    #add the common options to the parser
    addCommonArgumentsToParser(parser,flatfielding=False,warping=False)
    #group for other run options
    run_option_group = parser.add_argument_group('run options', 'other options for this run')
    run_option_group.add_argument('--max_files', default=-1, type=int,
                                  help='Maximum number of files to use (default = -1 runs all files for the requested slide)')
    run_option_group.add_argument('--max_masks', default=-1, type=int,
                                  help='Maximum number of masks to write out (prevents writing too much while testing, default=-1 writes out everything)')
    args = parser.parse_args(args=args)
    #check the arguments
    checkArgs(args)
    #add a file to the logger
    filehandler = logging.FileHandler(os.path.join(args.workingdir,f'{args.slideID}_overexposed_hpfs.log'))
    logger.addHandler(filehandler)
    #run alignments and check for overexposed HPFs
    #findOverexposedHPFsForSlide(args.rawfile_top_dir,args.root_dir,args.slideID,args.exposure_time_offset_file,args.workingdir)
    writeOverlapMSETable(args.rawfile_top_dir,args.root_dir,args.slideID,args.exposure_time_offset_file,args.workingdir)
    logger.info('Done : )')

if __name__=='__main__' :
    main()
