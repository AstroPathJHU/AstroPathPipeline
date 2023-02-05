#imports
import pathlib
import multiprocessing as mp
from argparse import ArgumentParser
import numpy as np
from astropath.utilities.img_file_io import get_raw_as_hwl, write_image_to_file

#constants
DEF_CORRECTION_FACTOR_FILEPATH = pathlib.Path(__file__).parent/'w_mk__reference_Polaris2.csv'
DEF_N_PROCS = 16
#IMAGE_DIMS = (1404, 1876, 43)
IMAGE_DIMS = (1404, 1872, 43)

def write_corrected_file(raw_file_path,correction_factors) :
    """
    Read a raw file, apply corrections, write out .fw file
    Meant to be run in parallel
    """
    image_array = get_raw_as_hwl(raw_file_path,*IMAGE_DIMS)
    corrected_image = (image_array/correction_factors[np.newaxis,np.newaxis,:]).astype(image_array.dtype)
    fw_file_path = raw_file_path.parent/(raw_file_path.name[:-len('.Data.dat')]+'.fw')
    write_image_to_file(corrected_image,fw_file_path)
    if fw_file_path.is_file() and fw_file_path.stat().st_size>0 :
        print(f'finished writing {fw_file_path.name}')
    else :
        print(f'ERROR: failed to write out {fw_file_path.name}!')

def correct_files(processloc,slideID,nprocs,correction_factors) :
    """
    Call a multiprocessed function for each raw file to read the file, apply the given correction factors, 
    and write it back out as a .fw file in the same location
    """
    file_loc = processloc/'astropath_ws'/'imagecorrection'/slideID/'flatw'/slideID
    if not file_loc.is_dir() :
        raise ValueError(f'ERROR: raw file directory {file_loc} does not exist!')
    n_raw_files = 0
    for _ in file_loc.glob('*.Data.dat') :
        n_raw_files+=1
    print(f'Found {n_raw_files} raw files to correct in {file_loc}')
    procs = []
    for ifp,fp in enumerate(file_loc.glob('*.Data.dat'),start=1) :
        fw_path = (fp.parent/(fp.name[:-len('.Data.dat')]+'.fw'))
        if fw_path.is_file() and fw_path.stat().st_size>0 :
            print(f'skipping {fp} because its fw file already exists')
            continue
        while len(procs)>=nprocs :
            p = procs.pop(0)
            p.join(0.1)
            if p.is_alive() :
                procs.append(p)
        print(f'writing out corrected file for {fp.name} ({ifp}/{n_raw_files})...')
        p = mp.Process(target=write_corrected_file,args=(fp,correction_factors))
        p.start()
        procs.append(p)
    print(f'joining all processes...')
    for p in procs :
        p.join()

def read_correction_factors(correction_factor_filepath,microscope_number) :
    """
    Read the correction factor file and return a list of the correction factors to use
    """
    if not correction_factor_filepath.is_file() :
        raise FileNotFoundError(f'ERROR: {correction_factor_filepath} does not exist!')
    correction_factors = []
    with open(correction_factor_filepath,'r') as cfp :
        lines = [line.strip() for line in cfp.readlines()]
    if microscope_number==2 :
        print('WARNING: microscope 2 should not be corrected at all because it is the standard!')
    header = lines.pop(0)
    index = header.split(',').index(f'Polaris_{microscope_number}')
    for line in lines :
        correction_factors.append(float(line.split(',')[index]))
    if not len(correction_factors)==IMAGE_DIMS[-1] :
        errmsg = f'ERROR: found {len(correction_factors)} correction factors for Polaris_{microscope_number} in '
        errmsg+= f'{correction_factor_filepath} but images have {IMAGE_DIMS[-1]} layers!'
        raise RuntimeError(errmsg)
    return np.array(correction_factors)

def main(args=None) :
    #parse the arguments
    parser = ArgumentParser()
    parser.add_argument('processloc',type=pathlib.Path,help='Path to the pwsh "processloc"')
    parser.add_argument('slideID',help='ID of the slide to correct')
    parser.add_argument('microscope_number',type=int,choices=(1,2,3),help='Which number this microscope is')
    parser.add_argument('--correction_factor_filepath',type=pathlib.Path,default=DEF_CORRECTION_FACTOR_FILEPATH,
        help=f'''Path to the CSV file with correction factors for all three microscopes as a function of image layer
                 (default = {DEF_CORRECTION_FACTOR_FILEPATH})''')
    parser.add_argument('--nprocs',type=int,default=DEF_N_PROCS,
        help=f'Number of parallel processes to use for writing out corrected files (default = {DEF_N_PROCS})')
    args = parser.parse_args(args)
    #make the list of correction factors to apply
    correction_factors = read_correction_factors(args.correction_factor_filepath,args.microscope_number)
    #read the raw files, correct them, and write them out as "fw" files
    correct_files(args.processloc,args.slideID,args.nprocs,correction_factors)
    print('Done : )')

if __name__=='__main__' :
    main()