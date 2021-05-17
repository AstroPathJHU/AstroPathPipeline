#imports
import multiprocessing as mp
import subprocess, time

#constants
N_PROCS = 4
#ET_OFFSET_FILE = '\\\\bki08/maggie/best_exposure_time_offsets_Vectra_9_8_2020.csv'
ET_OFFSET_FILE = r'//bki08/maggie/best_exposure_time_offsets_Polaris_9_8_2020.csv'
#RAWFILE_TOP_DIR = '\\\\bki07/dat'
#ROOT_DIR = '\\\\bki02/e/Clinical_Specimen'
#RAWFILE_TOP_DIR = '\\\\bki07/dat_2'
#ROOT_DIR = '\\\\bki04/Clinical_Specimen_2'
#RAWFILE_TOP_DIR = r'//bki07/dat_7'
#ROOT_DIR = r'//bki04/Clinical_Specimen_7'
#RAWFILE_TOP_DIR = r'//bki07/dat_7'
ROOT_DIR = r'//bki03/Clinical_Specimen_4'
RAWFILE_TOP_DIR = r'//bki07/dat_4'
#ROOT_DIR = r'//bki02/G/VectraPolaris_Data/WholeSlideScans/JHU_Polaris_Melanoma/JHUPolaris_1'
CMD_BASE = f'run_flatfield slide_mean_image --exposure_time_offset_file {ET_OFFSET_FILE} --rawfile_top_dir {RAWFILE_TOP_DIR}'
CMD_BASE+= f' --root_dir {ROOT_DIR} --n_threads {int(20/N_PROCS)}'
SLIDE_IDS = [
'PZ1',
'PZ2',
'PZ3',
'PZ4',
'PZ5',
'PZ6',
'PZ7',
'PZ8',
'PZ9',
'PZ10',
#'PZ11', ran this one as a test
'PZ12',
'PZ13',
'PZ14',
'PZ15',
'PZ16',
'PZ17',
'PZ18',
'PZ19',
'PZ20',
'PZ21',
'PZ22',
'PZ23',
'PZ24',
'PZ25',
'PZ26',
'PZ27',
'PZ28',
'PZ29',
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
