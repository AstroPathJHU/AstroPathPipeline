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
RAWFILE_TOP_DIR = r'//bki07/dat_7'
ROOT_DIR = r'//bki04/Clinical_Specimen_7'
CMD_BASE = f'run_flatfield slide_mean_image --exposure_time_offset_file {ET_OFFSET_FILE} --rawfile_top_dir {RAWFILE_TOP_DIR}'
CMD_BASE+= f' --root_dir {ROOT_DIR} --n_threads {int(40/N_PROCS)}'
SLIDE_IDS = [
#'M1_1',
#'M2_3',
#'M3_1',
#'M4_2',
#'M6_1',
#'M7_2',
#'M9_1',
#'M10_2',
#'M11_1',
#'M12_1',
#'M16_1',
#'M17_1',
#'M19_1',
#'M21_1',
#'M22_7',
#'M23_1',
#'M24_1',
#'M25_1',
#'M26_1',
#'M27_5',
#'M28_1',
#'M29_1',
#'M31_1',
#'M32_1',
#'M34_1',
#'M35_1',
#'M36_1',
#'M37_1',
#'M38_1',
#'M39_1',
#'M41_1',
#'M42_1',
#'M43_1',
#'M44_1',
#'M45_1',
#'M47_1',
#'M48_1',
#'M51_1',
#'M52_1',
#'M53_1',
#'M54_1',
#'M55_1',
#'M57_1',
#'M79_1',
#'M80_1',
#'M81_1',
#'M82_1',
#'M83_1',
#'M84_1',
#'M85_1',
#'M86_1',
#'M87_1',
#'M88_1',
#'M89_1',
#'M90_1',
#'M91_1',
#'M92_1',
#'M93_1',
#'M94_1',
#'M95_1',
#'M96_1',
#'M97_1',
#'M98_1',
#'M99_1',
#'M101_1',
#'M102_1',
#'M103_1',
#'M104_1',
#'M105_1',
#'M106_1',
#'M107_1',
#'M109_1',
#'M110_1',
#'M111_1',
#'M112_1'
#'L10_1',
#'L10_2',
#'L11_1',
#'L11_2',
#'L11_3',
#'L11_4',
#'L11_5',
#'L12_1',
#'L12_2',
#'L13_1',
#'L13_2',
#'L13_3',
#'L13_4',
#'L14_1',
#'L14_2',
#'L15_1',
#'L15_2',
#'L16_1',
#'L17_1',
#'L17_2',
#'L17_3',
#'L18_1',
#'L18_2',
#'L18_3',
#'L19_1',
#'L19_2',
#'L19_3',
#'L1_1',
#'L1_2',
#'L1_3',
#'L1_4',
#'L20_1',
#'L20_2',
#'L20_3',
#'L20_4',
#'L21_1',
#'L23_1',
#'L24_1',
#'L25_1',
#'L26_1',
#'L27_1',
#'L28_1',
#'L29_1',
#'L29_2',
#'L2_1',
#'L2_2',
#'L2_3',
#'L2_4',
#'L30_1',
#'L31_1',
#'L32_1',
#'L33_1',
#'L34_1',
#'L34_2',
#'L34_3',
#'L35_1',
#'L35_2',
#'L35_3',
#'L37_1',
#'L37_2',
#'L38_1',
#'L38_2',
#'L39_1',
#'L3_1',
#'L3_2',
#'L40_1',
#'L41_1',
#'L42_1',
#'L43_1',
#'L45_1',
#'L46_1',
#'L47_1',
#'L48_1',
#'L49_1',
#'L49_2',
#'L4_1',
#'L4_2',
#'L50_1',
#'L51_1',
#'L51_2',
#'L51_3',
#'L51_4',
#'L52_1',
#'L52_2',
#'L52_3',
#'L52_4',
#'L53_1',
#'L53_2',
#'L54_1',
#'L54_2',
#'L54_3',
#'L55_1',
#'L5_1',
#'L5_2',
#'L6_1',
#'L7_1',
#'L7_2',
#'L8_1',
#'L8_2',
#'L9_1',
#'L9_2',
'ZW1',
'ZW2',
'ZW3',
'ZW4',
'ZW5',
'ZW6',
'ZW7',
'ZW8',
'ZW9',
'ZW10',
'ZW11',
'ZW12',
'ZW13',
'ZW14',
'ZW15',
'ZW16',
'ZW17',
'ZW18',
'ZW19',
'ZW20',
'ZW21',
'ZW22',
'ZW23',
'ZW24',
'ZW25',
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
