#imports
from microscopealignment.alignment.alignmentset import AlignmentSetFromXML
from microscopealignment.utilities.img_file_io import getExposureTimesByLayer
from microscopealignment.utilities.tableio import writetable
from microscopealignment.utilities.misc import cd
from typing import List
import matplotlib.pyplot as plt, multiprocessing as mp
import os, glob, dataclasses

#constants
RAWFILE_EXT = '.Data.dat'
N_THREADS = 32
OUTPUT_FN_VECTRA = 'overlaps_with_different_exposure_times_vectra.csv'
OUTPUT_FN_POLARIS = 'overlaps_with_different_exposure_times_polaris.csv'

samples_vectra = {
    'dat':{'rawfile_top_dir':r"X:\\",
           'metadata_top_dir':r"W:\\Clinical_Specimen\\",
           'sample_names':['M1_1','M2_3','M3_1','M4_2','M6_1','M7_2','M9_1','M10_2','M11_1','M12_1','M16_1','M17_1',
                           'M19_1','M21_1','M22_7','M23_1','M24_1','M25_1','M26_1','M27_5','M28_1','M29_1','M31_1',
                           'M32_1','M34_1','M35_1','M36_1','M37_1','M38_1','M39_1','M41_1','M42_1','M43_1','M44_1',
                           'M45_1','M47_1','M48_1','M51_1','M52_1','M53_1','M54_1','M55_1','M57_1'],
           'layers':[1,10,19,26,33],
           'nlayers':35,
           'working_dir':'exposure_times_all_vectra_samples',
          }
    }

samples_polaris = {
    'dat_4':{'rawfile_top_dir':r"T:\\",
             'metadata_top_dir':r"T:\\",
             'sample_names':[f'PZ{i}' for i in range(1,30)],
             'layers':[1,10,12,18,21,30,37],
             'nlayers':43,
             'working_dir':'exposure_times_all_polaris_samples',
          },
    'dat_5':{'rawfile_top_dir':r"S:\\",
             'metadata_top_dir':r"S:\\",
             'sample_names':[f'YY{i}' for i in range(1,36)],
             'layers':[1,10,12,18,21,30,37],
             'nlayers':43,
             'working_dir':'exposure_times_all_polaris_samples',
          },
    'dat_6':{'rawfile_top_dir':r"R:\\",
             'metadata_top_dir':r"R:\\",
             'sample_names':[f'YX{i}' for i in range(1,36)],
             'layers':[1,10,12,18,21,30,37],
             'nlayers':43,
             'working_dir':'exposure_times_all_polaris_samples',
          },
    'dat_7':{'rawfile_top_dir':r"Q:\\",
             'metadata_top_dir':r"Q:\\",
             'sample_names':[f'ZW{i}' for i in range(1,26)],
             'layers':[1,10,12,18,21,30,37],
             'nlayers':43,
             'working_dir':'exposure_times_all_polaris_samples',
          },
    'dat_8':{'rawfile_top_dir':r"V:\\",
             'metadata_top_dir':r"V:\\",
             'sample_names':[f'YZ{i}' for i in range(50,74)],
             'layers':[1,10,12,18,21,30,37],
             'nlayers':43,
             'working_dir':'exposure_times_all_polaris_samples',
          },
    }

@dataclasses.dataclass
class SampleOverlapsWithDifferentExposureTimes :
    name : str
    n_overlaps_per_layer_group : List[int]

def getNOverlapsWithDifferentExposureTimes(rtd,mtd,sn,nlayers,layers,sd,return_dict) :
    to_return = []
    with cd(os.path.join(rtd,sn)) :
        all_rfps = [os.path.join(rtd,sn,fn) for fn in glob.glob(f'*{RAWFILE_EXT}')]
    exp_times = {}
    for fi,rfp in enumerate(all_rfps,start=1) :
        if fi%100==0 :
            print(f'Getting exposure times for {sn} image {fi} of {len(all_rfps)}....')
        rfkey = os.path.basename(os.path.normpath(rfp)).rstrip(RAWFILE_EXT)
        exp_times[rfkey] = []
        all_exp_times = getExposureTimesByLayer(rfp,nlayers,metadata_top_dir=mtd,subdirectory=sd)
        for ln in layers :
            exp_times[rfkey].append(all_exp_times[ln-1])
    n_overlaps = [0 for ln in layers]
    for li,ln in enumerate(layers) :
        print(f'Making AlignmentSet for {sn} layer {ln}....')
        a = AlignmentSetFromXML(mtd,rtd,sn,nclip=8,readlayerfile=False,layer=ln)
        print(f'Finding overlaps for {sn} layer {ln}....')
        rfkeys_by_rect_n = {}
        for r in a.rectangles :
            rfkeys_by_rect_n[r.n] = r.file.rstrip('.im3')
        for olap in a.overlaps :
            p1key = rfkeys_by_rect_n[olap.p1]
            p2key = rfkeys_by_rect_n[olap.p2]
            if p1key in exp_times.keys() and p2key in exp_times.keys() :
                if exp_times[p1key][li] != exp_times[p2key][li] :
                    n_overlaps[li]+=1
    print(f'{sn} # of different overlaps in each layer group = {n_overlaps}')
    return_dict[sn]=n_overlaps

if __name__=='__main__' :
    mp.freeze_support()

    n_diff_olaps_vectra = {}
    manager = mp.Manager()
    rdict_vectra = manager.dict()
    procs = []
    for samples in samples_vectra.values() :
        for si,sn in enumerate(samples['sample_names'],start=1) :
            print(f'Getting exposure times for images in {sn} ({si} of {len(samples["sample_names"])})')
            p = mp.Process(target=getNOverlapsWithDifferentExposureTimes,
                           args=(samples['rawfile_top_dir'],
                                 samples['metadata_top_dir'],
                                 sn,
                                 samples['nlayers'],
                                 samples['layers'],
                                 True,
                                 rdict_vectra)
                           )
            procs.append(p)
            p.start()
            if len(procs)>=N_THREADS :
                for proc in procs :
                    proc.join()
                procs = []
    for proc in procs :
        proc.join()
    for samples in samples_vectra.values() :
        for si,sn in enumerate(samples['sample_names'],start=1) :
            n_diff_olaps_vectra[sn] = rdict_vectra[sn]
    all_results = []
    for sn in n_diff_olaps_vectra.keys() :
        all_results.append(SampleOverlapsWithDifferentExposureTimes(sn,n_diff_olaps_vectra[sn]))
    writetable(OUTPUT_FN_VECTRA,all_results)
    for li,ln in enumerate(list(samples_vectra.values())[0]['layers']) :
        thislayer_n_overlaps_vectra = [(sn,n_diff_olaps_vectra[sn][li]) for sn in n_diff_olaps_vectra.keys()]
        thislayer_n_overlaps_vectra.sort(key=lambda x : x[1],reverse=True)
        print(f'Layer {ln} samples sorted by number of overlaps with different exposure times: {thislayer_n_overlaps_vectra}')

    n_diff_olaps_polaris = {}
    manager = mp.Manager()
    rdict_polaris = manager.dict()
    procs = []
    for samples in samples_polaris.values() :
        for si,sn in enumerate(samples['sample_names'],start=1) :
            print(f'Getting exposure times for images in {sn} ({si} of {len(samples["sample_names"])})')
            p = mp.Process(target=getNOverlapsWithDifferentExposureTimes,
                           args=(samples['rawfile_top_dir'],
                                 samples['metadata_top_dir'],
                                 sn,
                                 samples['nlayers'],
                                 samples['layers'],
                                 False,
                                 rdict_polaris)
                           )
            procs.append(p)
            p.start()
            if len(procs)>=N_THREADS :
                for proc in procs :
                    proc.join()
                procs = []
    for proc in procs :
        proc.join()
    for samples in samples_polaris.values() :
        for si,sn in enumerate(samples['sample_names'],start=1) :
            n_diff_olaps_polaris[sn] = rdict_polaris[sn]
    all_results = []
    for sn in n_diff_olaps_polaris.keys() :
        all_results.append(SampleOverlapsWithDifferentExposureTimes(sn,n_diff_olaps_polaris[sn]))
    writetable(OUTPUT_FN_POLARIS,all_results)
    for li,ln in enumerate(list(samples_polaris.values())[0]['layers']) :
        thislayer_n_overlaps_polaris = [(sn,n_diff_olaps_polaris[sn][li]) for sn in n_diff_olaps_polaris.keys()]
        thislayer_n_overlaps_polaris.sort(key=lambda x : x[1],reverse=True)
        print(f'Layer {ln} samples sorted by number of overlaps with different exposure times: {thislayer_n_overlaps_polaris}')
