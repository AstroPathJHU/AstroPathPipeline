#imports
from .utilities import addCommonWarpingArgumentsToParser, checkDirAndFixedArgs
from argparse import ArgumentParser


#################### HELPER FUNCTIONS ####################

#helper function to make sure th command line arguments are alright
def checkArgs(args) :
    #check to make sure the directories exist and the 'fixed' argument is okay
    checkDirAndFixedArgs(args)
    #make the working directory if it doesn't already exist
    if not os.path.isdir(args.workingdir_name) :
        os.mkdir(args.workingdir_name)

#helper function to make the list of commands to run for the initial pattern fit
def getInitialPatternFitCmds() :
    pass

#helper function to make the list of commands to run for the principal point fits
def getPrincipalPointFitCmds() :
    pass

#helper function to make the list of commands to run for the final pattern fits
def getFinalPatternFitCmds() :
    pass

#################### MAIN SCRIPT ####################

if __name__=='__main__' :
    mp.freeze_support()
    #define and get the command-line arguments
    parser = ArgumentParser()
    #add the common arguments
    addCommonWarpingArgumentsToParser(parser)
    args = parser.parse_args()
    #make sure the arguments are alright
    checkArgs(args)

