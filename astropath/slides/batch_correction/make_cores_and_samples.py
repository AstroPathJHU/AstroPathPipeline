"""
This script creates the Ctrl/*_cores.csv and Ctrl/*_samples.csv files for a cohort's Control TMA sample
"""

#imports
import pathlib
from argparse import ArgumentParser

#constants
DEFAULT_TMA_TISSUE_TYPES_FILE = '//bki04/astropath_processing/batch_correction/TMA_tissuetypes.xlsx'

def main() :
    #take in command line arguments
    parser = ArgumentParser()
    parser.add_argument('project_number',type=int,help='The project number whose csv files should be created')
    parser.add_argument('--tissue_types_file',type=pathlib.Path,default=DEFAULT_TMA_TISSUE_TYPES_FILE,
                        help=f'The path to the TMA tissue types file (default = {DEFAULT_TMA_TISSUE_TYPES_FILE})')
    parser.add_argument('--outdir',type=pathlib.Path,
                        help='Path to the directory that should hold the output *_cores.csv and *_samples.csv files')
    args = parser.parse_args()
    #get necessary information from files that already exist
    #write out the *_cores.csv file
    #write out the *_samples.csv file

if __name__=='__main__' :
    main()
