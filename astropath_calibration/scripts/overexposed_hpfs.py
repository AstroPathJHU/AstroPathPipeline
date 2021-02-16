#imports
from astropath_calibration.warping.alignmentset import AlignmentSetForWarping 
from astropath_calibration.utilities import units
from astropath_calibration.utilities.img_file_io import getMedianExposureTimesAndCorrectionOffsetsForSlide
from astropath_calibration.utilities.misc import cd, addCommonArgumentsToParser, cropAndOverwriteImage
from astropath_calibration.utilities.config import CONST as UNIV_CONST
from matplotlib import colors
from argparse import ArgumentParser
import numpy as np, matplotlib.pyplot as plt
import logging, os
units.setup('fast')

#constants
CUT=0.15

#logger
logger = logging.getLogger("overexposed_hpfs")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s  [%(funcName)s]","%Y-%m-%d %H:%M:%S"))
logger.addHandler(handler)

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

#################### HELPER FUNCTIONS ####################

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
                                   layer=1,filetype='raw',useGPU=True)
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
    #start up the other sheet of plots
    xs = np.array([r.cx for r in asets[UNIV_CONST.DAPI_LAYER_GROUP_INDEX_35].rectangles])
    ys = np.array([r.cy for r in asets[UNIV_CONST.DAPI_LAYER_GROUP_INDEX_35].rectangles])
    w = np.max(xs)-np.min(xs)
    h = np.max(ys)-np.min(ys)
    if h>w :
        f,ax = plt.subplots(len(UNIV_CONST.BRIGHTEST_LAYERS_35)-1,3,figsize=(3*((1.1*w)/(1.1*h))*9.6,(len(UNIV_CONST.BRIGHTEST_LAYERS_35)-1)*9.6))
    else :
        f,ax = plt.subplots(len(UNIV_CONST.BRIGHTEST_LAYERS_35)-1,3,figsize=(3*9.6,(len(UNIV_CONST.BRIGHTEST_LAYERS_35)-1)*9.6*((1.1*h)/(1.1*w))))
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
                abs_diff_devs.append(abs(abs(r12-r11)-abs(r22-r21)))
                frac_diff_devs.append(abs(abs((r12-r11)/(0.5*(r12+r11)))-abs((r22-r21)/(0.5*(r22+r21)))))
            abs_diff_devs.remove(max(abs_diff_devs))
            frac_diff_devs.remove(max(frac_diff_devs))
            mean_abs_diff_dev = np.mean(np.array(abs_diff_devs))
            mean_frac_diff_dev = np.mean(np.array(frac_diff_devs))
            to_plot.append({'x':r.cx,'y':r.cy,'n':r.n,'f':r.file,'abs':mean_abs_diff_dev,'frac':mean_frac_diff_dev})
        for p in to_plot :
            if p['frac']>=CUT :
                logger.info(f"rectangle # {p['n']} (file {p['f']}) flagged using layer {ln} with abs. diff {p['abs']:.2f}, frac. diff {p['frac']:.6f}")
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
        ax[lgi-1][2].scatter([p['x'] for p in to_plot if p['frac']<CUT],
                             [p['y'] for p in to_plot if p['frac']<CUT],
                             marker='o',
                             c='gray',
                             label='not flagged')
        ax[lgi-1][2].scatter([p['x'] for p in to_plot if p['frac']>=CUT],
                             [p['y'] for p in to_plot if p['frac']>=CUT],
                             marker='o',
                             c='tab:red',
                             label='flagged')
        ax[lgi-1][2].invert_yaxis()
        ax[lgi-1][2].set_title(f'{sid} HPFs flagged using layer {ln}',fontsize=16)
        ax[lgi-1][2].legend(loc='best')
    fn=f'{sid}_overexposed_hpf_locations.png'
    with cd(workingdir) :
        plt.savefig(fn)
        cropAndOverwriteImage(fn)

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
    findOverexposedHPFsForSlide(args.rawfile_top_dir,args.root_dir,args.slideID,args.exposure_time_offset_file,args.workingdir)
    logger.info('Done : )')

if __name__=='__main__' :
    main()
