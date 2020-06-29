#imports
import os, logging

#set up the logger
warp_logger = logging.getLogger("warpfitter")
warp_logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(message)s    [%(funcName)s, %(asctime)s]"))
warp_logger.addHandler(handler)

class WarpingError(Exception) :
    """
    Class for errors encountered during warping
    """
    pass

#helper function to make sure necessary directories exist and that the input choice of fixed parameters is valid
def checkDirAndFixedArgs(args) :
    #rawfile_top_dir/[sample] must exist
    rawfile_dir = os.path.join(args.rawfile_top_dir,args.sample)
    if not os.path.isdir(rawfile_dir) :
        raise ValueError(f'rawfile directory {rawfile_dir} does not exist!')
    #dbload top dir must exist
    if not os.path.isdir(args.dbload_top_dir) :
        raise ValueError(f'dbload_top_dir argument ({args.dbload_top_dir}) does not point to a valid directory!')
    #dbload top dir dir must be usable to find a metafile directory
    metafile_dir = os.path.join(args.dbload_top_dir,args.sample,'dbload')
    if not os.path.isdir(metafile_dir) :
        raise ValueError(f'dbload_top_dir ({args.dbload_top_dir}) does not contain "[sample name]/dbload" subdirectories!')
    #the parameter fixing string must correspond to some combination of options
    fix_cxcy   = 'cx' in args.fixed and 'cy' in args.fixed
    fix_fxfy   = 'fx' in args.fixed and 'fy' in args.fixed
    fix_k1k2k3 = 'k1' in args.fixed and 'k2' in args.fixed and 'k3' in args.fixed
    fix_p1p2   = 'p1' in args.fixed and 'p2' in args.fixed
    if args.fixed!=[''] and len(args.fixed)!=2*sum([fix_cxcy,fix_fxfy,fix_p1p2])+(3*int(fix_k1k2k3)) :
        raise ValueError(f'Fixed parameters argument ({args.fixed}) does not result in a valid fixed parameter condition!')