#imports
import multiprocessing as mp
import os, subprocess, time

#constants
N_PROCS = 4
ET_OFFSET_FILE = '\\bki08\\maggie\\best_exposure_time_offsets_Vectra_9_8_2020.csv'
RAWFILE_TOP_DIR = '\\bki07\\dat'
ROOT_DIR = '\\bki02\\e\\Clinical_Specimen'
CMD_BASE = f'run_flatfield slide_mean_image --exposure_time_offset_file {ET_OFFSET_FILE} --rawfile_top_dir {RAWFILE_TOP_DIR}'
CMD_BASE+= f' --root_dir {ROOT_DIR} --n_threads {int(64/N_PROCS)}'
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
'M16_1',
'M17_1',
'M19_1',
'M21_1',
'M22_7',
'M23_1',
'M24_1',
'M25_1',
'M26_1',
'M27_5',
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
'M41_1',
'M42_1',
'M43_1',
'M44_1',
'M45_1',
'M47_1',
'M48_1',
'M51_1',
'M52_1',
'M53_1',
'M54_1',
'M55_1',
'M57_1',
'M107_1',
'M109_1',
'M110_1',
'M111_1',
'M112_1'
]

#helper function to run one command with subprocess
def runOneCmd(cmd) :
	subprocess.call(cmd)

cmds = []
for slideID in SLIDE_IDS :
	thiscmd = f'{CMD_BASE} --slides {slideID}'
	cmds.append(thiscmd)

print(f'Will run the follow commands in {N_PROCS} at once:')
for cmd in cmds :
	print(cmds)

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
