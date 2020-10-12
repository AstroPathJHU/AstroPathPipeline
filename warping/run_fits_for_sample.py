#imports
from .utilities import warp_logger, WarpingError, addCommonWarpingArgumentsToParser, checkDirArgs, getOctetsFromArguments
from .config import CONST
from ..utilities.tableio import writetable
from ..utilities.misc import cd
from argparse import ArgumentParser
import multiprocessing as mp
import os, subprocess, random

#################### FILE-SCOPE CONSTANTS ####################
INITIAL_PATTERN_DIR_STEM = 'warping_initial_pattern'
PRINCIPAL_POINT_DIR_STEM = 'warping_center_principal_point'
FINAL_PATTERN_DIR_STEM   = 'warping_final_pattern'

#################### HELPER FUNCTIONS ####################

#helper function to make sure th command line arguments are alright
def checkArgs(args) :
    #check to make sure the directories exist and the 'fixed' argument is okay
    checkDirArgs(args)
    #tell the user what's going to happen based on the mode/octet splitting arguments
    if args.mode=='fit' :
        warp_logger.info(f'Three groups of fits will be performed, using {args.workers} CPUs each, to find the warping pattern for {args.sample}:')
        warp_logger.info(f'{args.initial_pattern_octets} octets will be used to find the initial general pattern')
        warp_logger.info(f'{args.principal_point_octets} octets will be used to define the center principal point location')
        warp_logger.info(f'{args.final_pattern_octets} octets will be used to find the overall best pattern')
    elif args.mode=='find_octets' :
        warp_logger.info(f'The set of valid octets to use will be found for {args.sample}')
    elif args.mode=='check_run' :
        warp_logger.info(f'Octets for {args.sample} will be found/split, and a test command will be run for the first group of fits.')

#helper function to set up the three fit group working directories
def setUpFitDirectories(args) :
    #find the octets for the sample
    octets = getOctetsFromArguments(args)
    #randomize and split the octets into groups for the three fits
    if len(octets) < args.initial_pattern_octets+args.principal_point_octets+args.final_pattern_octets :
        msg = f'ERROR: There are {len(octets)} valid octets for {args.sample} but you requested using {args.initial_pattern_octets}, then {args.principal_point_octets},'
        msg+= f' and then {args.final_pattern_octets} in the fit groups, respectively!'
        raise WarpingError(msg)
    random.shuffle(octets)
    octets_1 = octets[:args.initial_pattern_octets];            start = args.initial_pattern_octets
    octets_2 = octets[start:start+args.principal_point_octets]; start+= args.principal_point_octets
    octets_3 = octets[start:start+args.final_pattern_octets]
    #create the working directories for the three fit groups and copy the subsets of the octets files
    wdn_1 = f'{INITIAL_PATTERN_DIR_STEM}_{args.sample}_{len(octets_1)}_octets'
    wdn_2 = f'{PRINCIPAL_POINT_DIR_STEM}_{args.sample}_{len(octets_2)}_octets'
    wdn_3 = f'{INITIAL_PATTERN_DIR_STEM}_{args.sample}_{len(octets_3)}_octets'
    wdns = [wdn_1,wdn_2,wdn_3]
    ols = [octets_1,octets_2,octets_3]
    with cd(args.workingdir) :
        for wdn,o in zip(wdns,ols) :
            os.mkdir(wdn)
            with cd(wdn) :
                writetable(f'{args.sample}{CONST.OCTET_OVERLAP_CSV_FILE_NAMESTEM}',o)
    #return the names of the created directories
    return tuple(wdns)

#helper function to make the command to run for the initial pattern fit group
def getInitialPatternFitCmd(wdn) :
    pass

#helper function to make the command to run for the principal point fit group
def getPrincipalPointFitCmd(wdn) :
    pass

#helper function to make the command to run for the final pattern fit group
def getFinalPatternFitCmd(wdn) :
    pass

#################### MAIN SCRIPT ####################

if __name__=='__main__' :
    #define and get the command-line arguments
    parser = ArgumentParser()
    #add the positional mode argument
    parser.add_argument('mode', help='What to do', choices=['fit','find_octets','check_run'])
    #add the common arguments, without those controlling the fit parameters (since that's automated) or the job organization
    addCommonWarpingArgumentsToParser(parser,fitpars=False,job_organization=False)
    #add the positional number of workers argument
    parser.add_argument('workers', default=None, type=int, help='Max # of CPUs to use in the multiprocessing pools (defaults to all available)')
    #arguments for how to split the total group of octets
    octet_splitting_group = parser.add_argument_group('octet splitting', 'how to split the total set of octets for each of the three fit groups')
    octet_splitting_group.add_argument('--initial_pattern_octets', type=int, default=50,
                                       help=f'Number of octets to use in the initial pattern fit')
    octet_splitting_group.add_argument('--principal_point_octets', type=int, default=50,
                                       help=f'Number of octets to use in the principal point location fit')
    octet_splitting_group.add_argument('--final_pattern_octets',   type=int, default=100,
                                       help=f'Number of octets to use in the final pattern fit')
    args = parser.parse_args()
    #make sure the arguments are alright
    checkArgs(args)
    #set up the three directories with their octet groupings
    dirname_1, dirname_2, dirname_3 = setUpFitDirectories(args)
    #get the command for the initial pattern fits and run them
    cmd_1 = getInitialPatternFitCmd(dirname_1)
    subprocess.call(cmd_1)

    



