#imports
from .alignsample import AlignSampleForWarping
from .config import CONST
from ...utilities.img_file_io import get_image_hwl_from_xml_file, get_raw_as_hwl, getMedianExposureTimeAndCorrectionOffsetForSlideLayer
from ...utilities.dataclasses import MyDataClass
from ...utilities.tableio import writetable, readtable
from ...utilities.misc import cd
from ...utilities.config import CONST as UNIV_CONST
import multiprocessing as mp
import numpy as np
import pathlib, platform

#helper classes to represent octets of overlaps
class OverlapOctet(MyDataClass) :
    root_dir             : str
    rawfile_top_dir      : str
    slide_ID             : str
    nclip                : int
    layer                : int
    threshold            : float
    p1_rect_n            : int
    olap_1_n             : int
    olap_2_n             : int
    olap_3_n             : int
    olap_4_n             : int
    olap_6_n             : int
    olap_7_n             : int
    olap_9_n             : int
    olap_8_n             : int
    olap_1_p1_pixel_frac : float
    olap_2_p1_pixel_frac : float
    olap_3_p1_pixel_frac : float
    olap_4_p1_pixel_frac : float
    olap_6_p1_pixel_frac : float
    olap_7_p1_pixel_frac : float
    olap_8_p1_pixel_frac : float
    olap_9_p1_pixel_frac : float
    olap_1_p2_pixel_frac : float
    olap_2_p2_pixel_frac : float
    olap_3_p2_pixel_frac : float
    olap_4_p2_pixel_frac : float
    olap_6_p2_pixel_frac : float
    olap_7_p2_pixel_frac : float
    olap_8_p2_pixel_frac : float
    olap_9_p2_pixel_frac : float
    @property
    def overlap_ns(self) :
        return [self.olap_1_n,self.olap_2_n,self.olap_3_n,self.olap_4_n,self.olap_6_n,self.olap_7_n,self.olap_8_n,self.olap_9_n]

# Helper function to read previously-saved octet definitions from a file
def readSlideOctetsFromOctetRunDir(octet_run_dir,rawfile_top_dir,root_dir,slide_ID,layer,logger=None) :
    #get the .csv file holding the octet p1s and overlaps ns
    octet_filepath = pathlib.Path(f'{octet_run_dir}/{slide_ID}{CONST.OCTET_OVERLAP_CSV_FILE_NAMESTEM}')
    if logger is not None :
        logger.info(f'Reading overlap octets from file {octet_filepath}...')
    #read the overlap ns from the file
    octets = readtable(octet_filepath,OverlapOctet)
    for octet_olap_n in octets :
        if octet_olap_n.root_dir.lower()!=root_dir.lower() :
            msg = f'ERROR: root_dir {root_dir} passed to readSlideOctetsFromOctetRunDir does not match '
            msg+= f'{octet_olap_n.root_dir} in octet file {octet_filepath}!'
            raise RuntimeError(msg)
        if octet_olap_n.rawfile_top_dir.lower()!=rawfile_top_dir.lower() :
            msg = f'ERROR: rawfile_top_dir {rawfile_top_dir} passed to readSlideOctetsFromOctetRunDir does not match '
            msg+= f'{octet_olap_n.rawfile_top_dir} in octet file {octet_filepath}!'
            raise RuntimeError(msg)
        if octet_olap_n.slide_ID.lower()!=slide_ID.lower() :
            msg = f'ERROR: slide_ID {slide_ID} passed to readSlideOctetsFromOctetRunDir does not match '
            msg+= f'{octet_olap_n.slide_ID} in octet file {octet_filepath}!'
            raise RuntimeError(msg)
        if octet_olap_n.nclip!=UNIV_CONST.N_CLIP :
            msg = f'ERROR: constant nclip {UNIV_CONST.N_CLIP} in readSlideOctetsFromOctetRunDir does not match '
            msg+= f'{octet_olap_n.nclip} in octet file {octet_filepath}!'
            raise RuntimeError(msg)
        if octet_olap_n.layer!=layer :
            msg = f'ERROR: layer {layer} passed to readSlideOctetsFromOctetRunDir does not match '
            msg+= f'{octet_olap_n.layer} in octet file {octet_filepath}!'
            raise RuntimeError(msg)
    octets.sort(key=lambda x:x.p1_rect_n)
    return octets

# Helper function to get the list of octets
#can be run in parallel by passing in a return list
def findSlideOctets(rtd,rootdir,threshold_file_path,req_pixel_frac,slideID,working_dir,layer,flatfield_file,et_offset_file,logger=None,return_list=None) :
    #start by getting the threshold of this slide layer from the the inputted file
    with open(threshold_file_path) as tfp :
        vals = [int(l.rstrip()) for l in tfp.readlines() if l.rstrip()!='']
    threshold_value = vals[layer-1]
    #create the alignment set, correct its files, and run its alignment
    msg = f'Performing an initial alignment to find valid octets for {slideID}'
    if logger is not None :
        logger.info(msg,slideID,rootdir)
    img_dims = get_image_hwl_from_xml_file(rootdir,slideID)
    flatfield = (get_raw_as_hwl(flatfield_file,*(img_dims),UNIV_CONST.FLATFIELD_IMAGE_DTYPE))[:,:,layer-1] if flatfield_file is not None else None
    med_et, offset = getMedianExposureTimeAndCorrectionOffsetForSlideLayer(rootdir,slideID,et_offset_file,layer) if et_offset_file is not None else None
    use_GPU = platform.system()!='Darwin'
    a = AlignSampleForWarping(rootdir,rtd,slideID,med_et=med_et,offset=offset,flatfield=flatfield,nclip=UNIV_CONST.N_CLIP,
                               readlayerfile=False,layer=layer,filetype='raw',useGPU=use_GPU)
    a.getDAPI()
    a.align()
    #get the list of overlaps
    overlaps = a.overlaps
    #filter out any that could not be aligned or that don't show enough bright pixels
    good_overlaps = []; rejected_overlaps = []
    for overlap in overlaps :
        if overlap.result.exit!=0 :
            if logger is not None :
                logger.info(f'overlap number {overlap.n} rejected: alignment status {overlap.result.exit}.')
            rejected_overlaps.append(overlap)
            continue
        ip1,ip2 = overlap.cutimages
        p1frac = (np.sum(np.where(ip1>threshold_value,1,0)))/(ip1.shape[0]*ip1.shape[1])
        p2frac = (np.sum(np.where(ip2>threshold_value,1,0)))/(ip2.shape[0]*ip2.shape[1])
        if p1frac<req_pixel_frac :
            if logger is not None :
                logger.info(f'overlap number {overlap.n} rejected: p1 image ({overlap.p1}) only has {100.*p1frac:.2f}% above threshold at flux = {threshold_value}.')
            rejected_overlaps.append((overlap,p1frac,p2frac))
            continue
        if p2frac<req_pixel_frac :
            if logger is not None :
                logger.info(f'overlap number {overlap.n} rejected: p2 image ({overlap.p2}) only has {100.*p2frac:.2f}% above threshold at flux = {threshold_value}.')
            rejected_overlaps.append((overlap,p1frac,p2frac))
            continue
        good_overlaps.append((overlap,p1frac,p2frac))
    msg = f'Found a total of {len(good_overlaps)} good overlaps from an original set of {len(overlaps)} for {slideID}'
    if logger is not None :
        logger.info(msg,slideID,rootdir)
    #find the overlaps that form full octets
    octets = []
    #begin by getting the set of all p1s
    p1s = set([o[0].p1 for o in good_overlaps])
    #for each p1, if there are eight good overlaps it forms an octet
    for p1 in p1s :
        overlapswiththisp1 = [o for o in good_overlaps if o[0].p1==p1]
        if len(overlapswiththisp1)==8 :
            overlapswiththisp1.sort(key=lambda x: x[0].tag)
            ons = [o[0].n for o in overlapswiththisp1]
            op1pfs = [o[1] for o in overlapswiththisp1]
            op2pfs = [o[2] for o in overlapswiththisp1]
            if logger is not None :
                logger.info(f'octet found for {slideID} with p1={p1} (overlaps #{min(ons)}-{max(ons)}).')
            octets.append(OverlapOctet(rootdir,rtd,slideID,UNIV_CONST.N_CLIP,layer,threshold_value,p1,*(ons),*(op1pfs),*(op2pfs)))
    octets.sort(key=lambda x: x.p1_rect_n)
    #save the file of which overlaps are in each valid octet
    if len(octets)>0 :
        with cd(working_dir) :
            writetable(f'{slideID}{CONST.OCTET_OVERLAP_CSV_FILE_NAMESTEM}',octets)
    #print how many octets there are 
    msg = f'{len(octets)} total good octets found for {slideID}'
    if logger is not None :
        logger.info(msg,slideID,rootdir)
    #return the list of octets
    if return_list is not None :
        for o in octets:
            return_list.append(o)
    else :
        return octets

#helper function to return the octets for a slide given just the command line arguments
def getOctetsFromArguments(args,logger=None) :
    if args.octet_file is not None :
        octet_file_path = pathlib.Path(args.octet_file).absolute()
        if not octet_file_path.is_file() :
            raise FileNotFoundError(f'ERROR: octet_file {octet_file_path} does not exist!')
        if logger is not None :
            logger.info(f'Reading overlap octets from file {octet_file_path}...')
        all_octets = readtable(octet_file_path,OverlapOctet)
        if hasattr(args,'slideID') :
            all_octets = [o for o in all_octets if o.slide_ID==args.slideID]
    else :
        octet_run_dir = pathlib.Path(args.octet_run_dir).absolute() if args.octet_run_dir is not None else pathlib.Path(args.workingdir).absolute()
        all_octets = []
        if hasattr(args,'slideID') :
            slide_ids_to_check=[args.slideID]
        elif hasattr(args,'slideIDs') :
            slide_ids_to_check=args.slideIDs
        else :
            raise ValueError('ERROR: neither slideID nor slideIDs are in arguments passed to getOctetsFromArguments!')
        if args.workers>1 :
            procs = []
            manager = mp.Manager()
            return_list = manager.list()
        for slideID in slide_ids_to_check :
            octet_filepath = pathlib.Path(f'{octet_run_dir}/{slideID}{CONST.OCTET_OVERLAP_CSV_FILE_NAMESTEM}')
            if octet_filepath.is_file() :
                msg = f'Copying octets for {slideID} from {octet_filepath}'
                if logger is not None :
                    logger.info(msg,slideID,args.root_dir)
                all_octets += readSlideOctetsFromOctetRunDir(octet_run_dir,args.rawfile_top_dir,args.root_dir,slideID,args.layer,logger)
            elif args.threshold_file_dir is not None :
                threshold_file_path=pathlib.Path(f'{args.threshold_file_dir}/{slideID}_{UNIV_CONST.BACKGROUND_THRESHOLD_TEXT_FILE_NAME_STEM}')
                if args.workers>1 :
                    p = mp.Process(target=findSlideOctets,args=(args.rawfile_top_dir,args.root_dir,threshold_file_path,
                                                                args.req_pixel_frac,slideID,args.workingdir,args.layer,
                                                                args.flatfield_file,args.exposure_time_offset_file,logger,return_list))
                    p.start()
                    procs.append(p)
                    while len(procs)>=args.workers :
                        for proc in procs :
                            if not proc.is_alive() :
                                proc.join()
                                procs.remove(proc)
                                break
                else :
                    all_octets += findSlideOctets(args.rawfile_top_dir,args.root_dir,threshold_file_path,args.req_pixel_frac,slideID,
                                                  args.workingdir,args.layer,args.flatfield_file,args.exposure_time_offset_file,logger)
            else :
                raise ValueError('ERROR: either an octet_run_dir or a threshold_file_dir must be supplied to define octets to run on!')
        if args.workers>1 :
            for proc in procs :
                proc.join()
            for o in return_list :
                all_octets.append(o)
    msg = f'Found a total set of {len(all_octets)} valid octets'
    if hasattr(args,'slideID') :
        msg+=f' for slide {args.slideID}'
    elif hasattr(args,'slideIDs') :
        msg+=f' for slides {args.slideIDs}'
    if logger is not None :
        logger.info(msg)
    return all_octets