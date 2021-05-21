#imports
import multiprocessing as mp
import subprocess, time

#constants
N_PROCS = 4
ET_OFFSET_FILE = '\\\\bki08/maggie/best_exposure_time_offsets_Vectra_9_8_2020.csv'
#ET_OFFSET_FILE = r'//bki08/maggie/best_exposure_time_offsets_Polaris_9_8_2020.csv'
#RAWFILE_TOP_DIR = '\\\\bki07/dat'
#RAWFILE_TOP_DIR = r'//bki04/flatw'
#ROOT_DIR = '\\\\bki02/e/Clinical_Specimen'
RAWFILE_TOP_DIR = r'//bki04/flatw_2'
ROOT_DIR = r'//bki04/Clinical_Specimen_2'
#RAWFILE_TOP_DIR = '\\\\bki07/dat_2'
#ROOT_DIR = '\\\\bki04/Clinical_Specimen_2'
#RAWFILE_TOP_DIR = r'//bki07/dat_7'
#ROOT_DIR = r'//bki04/Clinical_Specimen_7'
#RAWFILE_TOP_DIR = r'//bki07/dat_7'
#ROOT_DIR = r'//bki03/Clinical_Specimen_4'
#RAWFILE_TOP_DIR = r'//bki07/dat_4'
#ROOT_DIR = r'//bki02/G/VectraPolaris_Data/WholeSlideScans/JHU_Polaris_Melanoma/JHUPolaris_1'
CMD_BASE = f'run_flatfield slide_mean_image --rawfile_top_dir {RAWFILE_TOP_DIR} --root_dir {ROOT_DIR}'
CMD_BASE+= f' --exposure_time_offset_file {ET_OFFSET_FILE}'
CMD_BASE+= ' --skip_masking --filetype flatw'
CMD_BASE+= f' --n_threads {int(20/N_PROCS)}'
SLIDE_IDS = [
'L1_1',
'L1_2',
'L1_3',
'L1_4',
'L2_1',
'L2_2',
'L2_3',
'L2_4',
'L3_1',
'L3_2',
'L4_1',
'L4_2',
'L5_1',
'L5_2',
'L6_1',
'L7_1',
'L7_2',
'L8_1',
'L8_2',
'L9_1',
'L9_2',
'L10_1',
'L10_2',
'L11_1',
'L11_2',
'L11_3',
'L11_4',
'L11_5',
'L12_1',
'L12_2',
'L13_1',
'L13_2',
'L13_3',
'L13_4',
'L14_1',
'L14_2',
'L15_1',
'L15_2',
'L16_1',
'L17_1',
'L17_2',
'L17_3',
'L18_1',
'L18_2',
'L18_3',
'L19_1',
'L19_2',
'L19_3',
'L20_1',
'L20_2',
'L20_3',
'L20_4',
'L21_1',
'L23_1',
'L24_1',
'L25_1',
'L26_1',
'L27_1',
'L28_1',
'L29_1',
'L29_2',
'L30_1',
'L31_1',
'L32_1',
'L33_1',
'L34_1',
'L34_2',
'L34_3',
'L35_1',
'L35_2',
'L35_3',
'L37_1',
'L37_2',
'L38_1',
'L38_2',
'L39_1',
'L40_1',
'L41_1',
'L42_1',
'L43_1',
'L45_1',
'L46_1',
'L47_1',
'L48_1',
'L49_1',
'L49_2',
'L50_1',
'L51_1',
'L51_2',
'L51_3',
'L51_4',
'L52_1',
'L52_2',
'L52_3',
'L52_4',
'L53_1',
'L53_2',
'L54_1',
'L54_2',
'L54_3',
'L55_1'
]

#helper function to run one command with subprocess
def runOneCmd(cmd) :
    subprocess.call(cmd)

def main(args=None) :
    mp.freeze_support()
    cmds = []
    for slideID in SLIDE_IDS :
        thiscmd = f'{CMD_BASE} --slides {slideID}'
        cmds.append(thiscmd)
    print(f'Will run the follow commands in {N_PROCS} processes at once:')
    for cmd in cmds :
        print(cmd)
    procs = []
    for cmd in cmds :
        p = mp.Process(target=runOneCmd,args=(cmd,))
        procs.append(p)
        p.start()
        while len(procs)>=N_PROCS :
            for proc in procs :
                if not proc.is_alive() :
                    proc.join()
                    delete_p = procs.pop(procs.index(proc))
                    delete_p = delete_p
                    del delete_p
            time.sleep(10)
    for proc in procs:
        proc.join()
    print('All processes done!')

if __name__ == '__main__' :
    main()
