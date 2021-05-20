#imports
from .corrector import RawfileCorrector
from .utilities import getWarpFieldPathsFromWarpDef
from ...baseclasses.sample import SampleDef
from ...baseclasses.logging import getlogger
from ...utilities.img_file_io import getImageHWLFromXMLFile
from ...utilities.misc import addCommonArgumentsToParser
from ...utilities.config import CONST as UNIV_CONST
from argparse import ArgumentParser
import pathlib

#################### FILE-SCOPE HELPER FUNCTIONS ####################

def checkArgs(args) :
    #make sure the directories all exist
    workingdir_location = (pathlib.Path.resolve(pathlib.Path(args.workingdir))).parent
    dirs_to_check = [args.rawfile_top_dir,args.root_dir,workingdir_location]
    for dirpath in dirs_to_check :
        if not pathlib.Path.is_dir(pathlib.Path(dirpath)) :
            raise ValueError(f'ERROR: directory {dirpath} does not exist!')
    #make sure the rawfile directory for this slide exists
    rawfile_dirpath = pathlib.Path(f'{args.rawfile_top_dir}/{args.slideID}')
    if not pathlib.Path.is_dir(rawfile_dirpath) :
        raise ValueError(f'ERROR: rawfile directory {rawfile_dirpath} for slide {args.slideID} does not exist!')
    #make sure the root directory for this slide exists
    root_dirpath = pathlib.Path(f'{args.root_dir}/{args.slideID}')
    if not pathlib.Path.is_dir(root_dirpath) :
        raise ValueError(f'ERROR: root directory {root_dirpath} for slide {args.slideID} does not exist!')
    #make sure the image dimensions work with the layer argument
    img_dims = getImageHWLFromXMLFile(args.root_dir,args.slideID)
    if (args.layer!=-1) and (not args.layer in range(1,img_dims[-1]+1)) :
        raise ValueError(f'ERROR: requested copying layer {args.layer} but raw files have dimensions {img_dims}!')
    ##make sure the exposure time correction file exists if necessary
    #if not args.skip_exposure_time_correction :
    #    if not pathlib.Path.is_file(pathlib.Path(args.exposure_time_offset_file)) :
    #        raise FileNotFoundError(f'ERROR: exposure time offset file {args.exposure_time_offset_file} does not exist!')
    ##make sure the flatfield file exists if necessary
    #if not args.skip_flatfielding :
    #    if not pathlibe.Path.is_file(pathlib.Path(args.flatfield_file)) :
    #        raise FileNotFoundError(f'ERROR: flatfield file {args.flatfield_file} does not exist!')
    #check some arguments related to the warping
    if args.skip_warping and ((args.warp_shift_file is not None) or (args.warp_shift is not None) or (args.warping_scalefactor!=1.0)) :
        raise RuntimeError('ERROR: warping is being skipped, so the requested shifts/rescaling are irrelevant!')
    if not args.skip_warping :
        if args.warp_def.endswith('.csv') :
            if not pathlib.Path.is_file(pathlib.Path(args.warp_def)) :
                raise FileNotFoundError(f'ERROR: warp fit result file {args.warp_def} does not exist!')
            if args.warp_shift_file is not None :
                if not pathlib.Path.is_file(pathlib.Path(args.warp_shift_file)) :
                    raise FileNotFoundError(f'ERROR: warp shift file {args.warp_shift_file} doe not exist!')
        else :
            if (args.warp_shift_file is not None) or (args.warp_shift is not None) :
                raise ValueError(f"ERROR: warp_def argument {args.warp_def} is not a warping parameter fit result file, so its pattern can't be shifted!")
            dx_warp_field_path, dy_warp_field_path = getWarpFieldPathsFromWarpDef(args.warp_def)
            if not pathlib.Path.is_file(pathlib.Path(dx_warp_field_path)) :
                raise FileNotFoundError(f'ERROR: dx warp field {dx_warp_field_path} does not exist!')
            if not pathlib.Path.is_file(pathlib.Path(dy_warp_field_path)) :
                raise FileNotFoundError(f'ERROR: dy warp field {dy_warp_field_path} does not exist!')

#################### MAIN SCRIPT ####################

def main(args=None) :
    if args is None :
        #define and get the command-line arguments
        parser = ArgumentParser()
        #add the slideID as the first positional argument
        parser.add_argument('slideID', help='Name of the slide to use')
        #add the common options to the parser
        addCommonArgumentsToParser(parser,et_correction=False)
        #add the arguments for shifting the warp pattern
        warp_shift_group = parser.add_mutually_exclusive_group()
        warp_shift_group.add_argument('--warp_shift_file',
                                     help='Path to the warp_shifts.csv file that should be applied to the files in this slide')
        warp_shift_group.add_argument('--warp_shift', 
                                     help='Use this argument to define a (delta-x, delta-y) shift from the inputted warp field')
        #group for other run options
        run_option_group = parser.add_argument_group('run options', 'other options for this run')
        run_option_group.add_argument('--warping_scalefactor',   default=1.0,   type=float,         
                                      help='Scalefactor by which the warping fields should be multiplied before application (default=1.0)')
        run_option_group.add_argument('--layer',                 default=-1,     type=int,         
                                      help='Image layer to use (indexed from 1; default=-1 does all layers)')
        run_option_group.add_argument('--input_file_extension', default='.Data.dat',
                                      help='Extension for the raw files that will be read in (default = ".Data.dat")')
        run_option_group.add_argument('--output_file_extension', default=f'{UNIV_CONST.FLATW_EXT}',
                                      help=f"""Extension for the corrected files that will be written out 
                                           (default = "{UNIV_CONST.FLATW_EXT}"; 2-digit layer code will be appended if layer != -1)""")
        run_option_group.add_argument('--max_files',             default=-1,    type=int,
                                      help='Maximum number of files to use (default = -1 runs all files for the requested slide)')
        args = parser.parse_args(args=args)
    #make the working directory
    if not pathlib.Path.is_dir(pathlib.Path(args.workingdir)) :
        pathlib.Path.mkdir(pathlib.Path(args.workingdir))
    #set up the logger information and enter its context
    module='image_correction'
    #mainlog = pathlib.Path(f'{args.workingdir}/{module}.log')
    #samplelog = pathlib.Path(f'{args.workingdir}/{args.slideID}-{module}.log')
    imagelog = pathlib.Path(f'{args.workingdir}/{args.slideID}_images-{module}.log')
    samp = SampleDef(SlideID=args.slideID,root=args.root_dir)
    #with getlogger(module=module,root=args.root_dir,samp=samp,uselogfiles=True,mainlog=mainlog,samplelog=samplelog,imagelog=imagelog,reraiseexceptions=False) as logger :
    with getlogger(module=module,root=args.root_dir,samp=samp,uselogfiles=True,imagelog=imagelog,reraiseexceptions=False) as logger :
        #check the arguments
        checkArgs(args)
        #start up the corrector from the arguments
        corrector = RawfileCorrector(args,logger)
        #read in, correct, and write out file layers
        corrector.run()
    #explicitly exit logger context
    assert 1==1

if __name__=='__main__' :
    main()
