#imports
import multiprocessing as mp
import subprocess

#constants
N_WORKERS = 3
ET_OFFSET_FILE = '\\\\bki08/maggie/best_exposure_time_offsets_Vectra_9_8_2020.csv'
FLATFIELD_FILE = '\\\\bki08/maggie/flatfield_batch_3-9_vectra_samples_21006_initial_images/flatfield_flatfield_batch_3-9_vectra_samples_21006_initial_images.bin'
THRESHOLD_FILE_DIR = '\\\\bki08/maggie/OLD_flatfield_batch_3-9_vectra_samples_21006_initial_images/thresholding_info'
RAWFILE_TOP_DIR = '\\\\bki07/dat'
ROOT_DIR = '\\\\bki02/e/Clinical_Specimen'
CMD_BASE_PREPEND = 'run_warping_fits fit'
CMD_BASE_APPEND = f'--exposure_time_offset_file {ET_OFFSET_FILE} --flatfield_file {FLATFIELD_FILE} --threshold_file_dir {THRESHOLD_FILE_DIR}'
SLIDE_IDS = ['M41_1','M51_1']
LAYERS = [5,14,22,29,34]

#helper function to run one command with subprocess
def runOneCmd(cmd) :
    subprocess.call(cmd)

def main(args=None) :
    mp.freeze_support()
    cmds = []
    for ln in LAYERS :
        for slideID in SLIDE_IDS :
            wdp = f'\\\\bki08/maggie/warping_{slideID}_layer_{ln}'
            thiscmd = f'{CMD_BASE_PREPEND} {slideID} {RAWFILE_TOP_DIR} {ROOT_DIR} {wdp} {N_WORKERS} {CMD_BASE_APPEND} --layer {ln}'
            cmds.append(thiscmd)
    print('Will run the follow commands:')
    for cmd in cmds :
        print(cmd)
    for cmd in cmds :
        runOneCmd(cmd)
    print('All processes done!')

if __name__ == '__main__' :
    main()
