#imports
import multiprocessing as mp
import subprocess, time

#constants
N_PROCS = 4
ET_OFFSET_FILE = '\\\\bki08/maggie/best_exposure_time_offsets_Vectra_9_8_2020.csv'
#ET_OFFSET_FILE = r'//bki08/maggie/best_exposure_time_offsets_Polaris_9_8_2020.csv'
#RAWFILE_TOP_DIR = '\\\\bki07/dat'
RAWFILE_TOP_DIR = r'//bki04/flatw'
ROOT_DIR = '\\\\bki02/e/Clinical_Specimen'
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
'M1_1',
'M2_3',
'M3_1',
'M4_2',
'M6_1',
'M7_2',
'M9_1',
'M10_2',
'M11_1',
'M12_1',
'M13_1',
'M14_1',
'M15_1',
'M16_1',
'M17_1',
'M19_1',
'M21_1',
'M22_7',
'M23_1',
'M24_1',
'M25_1',
'M26_1',
'M27_1',
'M28_1',
'M29_1',
'M31_1',
'M32_1',
'M34_1',
'M35_1',
'M36_1',
'M37_1',
'M38_1',
'M39_1',
'M40_1',
'M41_1',
'M42_1',
'M43_1',
'M44_1',
'M45_1',
'M47_1',
'M48_1',
'M49_1',
'M51_1',
'M52_1',
'M53_1',
'M54_1',
'M55_1',
'M56_1',
'M57_1',
'M79_1',
'M80_1',
'M81_1',
'M82_1',
'M83_1',
'M84_1',
'M85_1',
'M86_1',
'M87_1',
'M88_1',
'M89_1',
'M90_1',
'M91_1',
'M92_1',
'M93_1',
'M94_1',
'M95_1',
'M96_1',
'M97_1',
'M98_1',
'M99_1',
'M101_1',
'M102_1',
'M103_1',
'M104_1',
'M105_1',
'M106_1',
'M107_1',
'M109_1',
'M110_1',
'M111_1',
'M112_1',
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
