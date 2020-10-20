#imports
from ..warping.warp import CameraWarp
from ..warping.utilities import WarpFitResult, WarpShift
from ..flatfield.config import CONST as FF_CONST
from ..warping.config import CONST as WARP_CONST
from ..utilities.img_correction import correctImageForExposureTime, correctImageLayerForExposureTime
from ..utilities.img_correction import correctImageLayerWithFlatfield, correctImageWithFlatfield, correctImageLayerWithWarpFields
from ..utilities.img_file_io import getImageHWLFromXMLFile, getRawAsHWL, getRawAsHW, writeImageToFile, findExposureTimeXMLFile
from ..utilities.img_file_io import writeModifiedExposureTimeXMLFile, getMedianExposureTimesAndCorrectionOffsetsForSample
from ..utilities.img_file_io import getMedianExposureTimeAndCorrectionOffsetForSampleLayer, getExposureTimesByLayer 
from ..utilities.tableio import readtable, writetable
from ..utilities.misc import cd, addCommonArgumentsToParser
import numpy as np, matplotlib.pyplot as plt
from argparse import ArgumentParser
from shutil import copy2
import os, time, logging, glob

#################### FILE-SCOPE VARIABLES ####################

APPLIED_CORRECTION_PLOT_DIR_NAME = 'applied_correction_plots'

correction_logger = logging.getLogger("correct_and_copy")
correction_logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s  [%(funcName)s]","%Y-%m-%d %H:%M:%S"))
correction_logger.addHandler(handler)

#################### RawfileCorrector CLASS ####################

class RawfileCorrector :

    #################### PROPERTIES ####################
    @property
    def logfile_timestamp(self) :
        return time.strftime("[%Y-%m-%d %H:%M:%S] ")

    #################### PUBLIC FUNCTIONS ####################

    def __init__(self,args) :
        """
        args = the set of command line arguments from ArgumentParser
        """
        #make sure the directories all exist
        workingdir_location = os.path.dirname(os.path.normpath(args.workingdir))
        dirs_to_check = [args.rawfile_top_dir,args.metadata_top_dir,workingdir_location]
        for dirpath in dirs_to_check :
            if not os.path.isdir(dirpath) :
                raise ValueError(f'ERROR: directory {dirpath} does not exist!')
        #make sure the rawfile directory for this sample exists
        rawfile_dirpath = os.path.join(args.rawfile_top_dir,args.sample)
        if not os.path.isdir(rawfile_dirpath) :
            raise ValueError(f'ERROR: rawfile directory {rawfile_dirpath} for sample {args.sample} does not exist!')
        self._rawfile_top_dir = args.rawfile_top_dir
        #make sure the metadata directory for this sample exists
        metadata_dirpath = os.path.join(args.metadata_top_dir,args.sample)
        if not os.path.isdir(metadata_dirpath) :
            raise ValueError(f'ERROR: metadata directory {metadata_dirpath} for sample {args.sample} does not exist!')
        self._metadata_top_dir = args.metadata_top_dir
        #set the sample name
        self._sample_name = args.sample
        #get the image dimensions and make sure the layer argument is valid
        self._img_dims = getImageHWLFromXMLFile(self._metadata_top_dir,self._sample_name)
        if (args.layer!=-1) and (not args.layer in range(1,self._img_dims[-1]+1)) :
            raise ValueError(f'ERROR: requested copying layer {args.layer} but raw files have dimensions {self._img_dims}')
        self._layer = args.layer
        #make the working directory
        if not os.path.isdir(args.workingdir) :
            os.mkdir(args.workingdir)
        self._working_dir_path = args.workingdir
        #see which layer(s) will be run
        layers_to_run = list(range(1,self._img_dims[-1]+1)) if self._layer==-1 else [self._layer]
        #make sure the exposure time correction file exists if necessary
        if not args.skip_exposure_time_correction :
            if not os.path.isfile(args.exposure_time_offset_file) :
                raise FileNotFoundError(f'ERROR: exposure time offset file {args.exposure_time_offset_file} does not exist!')
            eto_filepath = args.exposure_time_offset_file
            if self._layer==-1 :
                self._med_exp_time, self._et_correction_offset = getMedianExposureTimesAndCorrectionOffsetsForSample(self._metadata_top_dir,
                                                                                       self._sample_name,
                                                                                       eto_filepath)
            else :
                self._med_exp_time, self._et_correction_offset = getMedianExposureTimeAndCorrectionOffsetForSampleLayer(self._metadata_top_dir,
                                                                                                                        self._sample_name,
                                                                                                                        eto_filepath,
                                                                                                                        self._layer)
        else :
            self._med_exp_time = None; self._et_correction_offset = None
        #make sure the flatfield file exists (if necessary) and set the flatfield variable
        if not args.skip_flatfielding :
            if not os.path.isfile(args.flatfield_file) :
                raise FileNotFoundError(f'ERROR: flatfield file {args.flatfield_file} does not exist!')
            ff_filepath = args.flatfield_file
            self._ff = getRawAsHWL(ff_filepath,*(self._img_dims),dtype=FF_CONST.IMG_DTYPE_OUT)
            if self._layer!=-1 :
                self._ff = self._ff[:,:,self._layer-1]
            with cd(self._working_dir_path) :
                if not os.path.isdir(APPLIED_CORRECTION_PLOT_DIR_NAME) :
                    os.mkdir(APPLIED_CORRECTION_PLOT_DIR_NAME)
                with cd(APPLIED_CORRECTION_PLOT_DIR_NAME) :
                    for ln in layers_to_run :
                        f,ax=plt.subplots(figsize=(6.4,(self._img_dims[0]/self._img_dims[1])*6.4))
                        if self._layer==-1 :
                            pos = ax.imshow(self._ff[:,:,ln-1])
                        else :
                            pos = ax.imshow(self._ff)
                        ax.set_title(f'applied flatfield, layer {ln}')
                        f.colorbar(pos,ax=ax)
                        plt.savefig(f'applied_flatfield_layer_{ln}.png')
                        plt.close()
        else :
            self._ff = None
        #make sure the dx and dy warping fields can be defined if necessary
        self._warps = None; self._dx_warp_field = None; self._dy_warp_field = None
        if not args.skip_warping :
            with cd(self._working_dir_path) :
                if not os.path.isdir(APPLIED_CORRECTION_PLOT_DIR_NAME) :
                    os.mkdir(APPLIED_CORRECTION_PLOT_DIR_NAME)
            #first try to define the warping from a parameter file
            if args.warp_def.endswith('.csv') :
                if not os.path.isfile(args.warp_def) :
                    raise FileNotFoundError(f'ERROR: warp fit result file {args.warp_def} does not exist!')
                warp_fit_result = readtable(args.warp_def,WarpFitResult)
                if len(warp_fit_result)==1 :
                    raise ValueError(f'ERROR: warp fit result file {args.warp_def} has more than one set of parameters!')
                wfr = warp_fit_result[0]
                warp_shifts = []
                if args.warp_shift_file is not None :
                    if not os.path.isfile(args.warp_shift_file) :
                        raise FileNotFoundError(f'ERROR: warp shift file {args.warp_shift_file} doe not exist!')
                    warp_shifts = readtable(args.warp_shift_file,WarpShift)
                elif args.warp_shift is not None :
                    cx_shift,cy_shift = args.warp_shift.split(',')
                    for ln in layers_to_run :
                        warp_shifts = warp_shifts.append(WarpShift(ln,cx_shift,cy_shift))
                if len(warp_shifts)>0 :
                    with cd(os.path.join(self._working_dir_path,APPLIED_CORRECTION_PLOT_DIR_NAME)) :
                        writetable('applied_warp_shifts.csv',warp_shifts)
                self._warps = {}
                for ln in layers_to_run :
                    cx_shift = 0.; cy_shift = 0.
                    if ln in [ws.ln for ws in warp_shifts] :
                        this_ws = ([ws for ws in warp_shifts if ws.ln==ln])[0]
                        cx_shift = this_ws.cx_shift; cy_shift = this_ws.cy_shift
                    sf = args.warping_scalefactor
                    self._warps[ln] = CameraWarp(self._img_dims[1],self._img_dims[0],wfr.cx+cx_shift,wfr.cy+cy_shift,
                                                 wfr.fx,wfr.fy,sf*wfr.k1,sf*wfr.k2,sf*wfr.k3,sf*wfr.p1,sf*wfr.p2)
                    fs = f'applied_warping_correction_layer_{ln}'
                    with cd(os.path.join(self._working_dir_path,APPLIED_CORRECTION_PLOT_DIR_NAME)) :
                        self._warps[ln].writeOutWarpFields(fs,save_fields=False)
            #if not, try to define the fields by the actual .bin file
            else :
                if (args.warp_shift_file is not None) or (args.warp_shift is not None) :
                    raise ValueError(f"ERROR: warp_def argument {args.warp_def} is not a warping parameter fit result file, so its pattern can't be shifted!")
                wf_dirname = os.path.basename(os.path.normpath(args.warp_def))
                dx_warp_field_path = os.path.join(args.warp_def,f'{WARP_CONST.X_WARP_BIN_FILENAME}_{wf_dirname}.bin')
                dy_warp_field_path = os.path.join(args.warp_def,f'{WARP_CONST.Y_WARP_BIN_FILENAME}_{wf_dirname}.bin')
                if not os.path.isfile(dx_warp_field_path) :
                    raise FileNotFoundError(f'ERROR: dx warp field {dx_warp_field_path} does not exist!')
                if not os.path.isfile(dy_warp_field_path) :
                    raise FileNotFoundError(f'ERROR: dy warp field {dy_warp_field_path} does not exist!')
                self._dx_warp_field = (args.warping_scalefactor)*(getRawAsHW(dx_warp_field_path,*(self._img_dims[:-1]),dtype=WARP_CONST.OUTPUT_FIELD_DTYPE))
                self._dy_warp_field = (args.warping_scalefactor)*(getRawAsHW(dy_warp_field_path,*(self._img_dims[:-1]),dtype=WARP_CONST.OUTPUT_FIELD_DTYPE))
                r_warp_field = np.sqrt((self._dx_warp_field**2)+(self._dy_warp_field**2))
                with cd(os.path.join(self._working_dir_path,APPLIED_CORRECTION_PLOT_DIR_NAME)) :
                    f,ax = plt.subplots(1,3,figsize=(3*6.4,(self._img_dims[0]/self._img_dims[1])*6.4))
                    pos = ax[0].imshow(r_warp_field)
                    ax[0].set_title('total warp correction')
                    f.colorbar(pos,ax=ax[0])
                    pos = ax[1].imshow(self._dx_warp_field)
                    ax[1].set_title('applied dx warp')
                    f.colorbar(pos,ax=ax[1])
                    pos = ax[2].imshow(self._dy_warp_field)
                    ax[2].set_title('applied dy warp')
                    f.colorbar(pos,ax=ax[2])
                    plt.savefig('applied_warping_correction_model.png')
                    plt.close()
        else :
            if (args.warp_shift_file is not None) or (args.warp_shift is not None) or (args.warping_scalefactor is not None) :
                raise RuntimeError('ERROR: warping is being skipped, so the requested shifts/rescaling are irrelevant!!')
        #start up the logfile and add some information to it
        self._logfile_name = f'{args.logfile_name_stem}_{time.strftime("%Y_%m_%d-%H_%M_%S")}.log'
        with cd(self._working_dir_path) :
            with open(self._logfile_name,'w') as fp :
                fp.write('LOGFILE for correct_and_copy_rawfiles\n')
                fp.write('-------------------------------------\n\n')
        self.__writeLog(f'Working directory {os.path.basename(os.path.normpath(self._working_dir_path))} has been created in {workingdir_location}.')
        if args.layer==-1 :
            self.__writeLog(f'Corrected {self._img_dims[-1]}-layer files will be written out to {self._working_dir_path}')
        else :
            self.__writeLog(f'Corrected layer {self._layer} files will be written out to {self._working_dir_path}.')
        if self._med_exp_time is None and self._et_correction_offset is None :
            self.__writeLog('Corrections for exposure time WILL NOT be applied.')
        else :
            self.__writeLog(f'Exposure time corrections WILL be applied based on offset factors in {eto_filepath}')
            if self._layer==-1 :
                for ln in layers_to_run :
                    msg = f'(Layer {ln} median sample exposure time={self._med_exp_time[ln-1]};'
                    msg+= f' exposure time correction offset = {self._et_correction_offset[ln-1]})'
                    self.__writeLog(msg)
            else :
                self.__writeLog(f'(Median sample exposure time={self._med_exp_time}; exposure time correction offset = {self._et_correction_offset})')
        if self._ff is None :
            self.__writeLog('Flatfielding corrections WILL NOT be applied.')
        else :
            self.__writeLog(f'Flatfield corrections WILL be applied as read from {ff_filepath}')
        if self._warps is None and self._dx_warp_field is None and self._dy_warp_field is None :
            self.__writeLog('Warping corrections WILL NOT be applied.')
        else :
            msg='Warping corrections will be applied as read from '
            if self._warps is not None :
                msg+=f'{args.warp_def}'
                if args.warp_shift_file is not None :
                    msg+=f' and shifted as read from {args.warp_shift_file}'
                elif args.warp_shift is not None :
                    msg+=f' and shifted by {args.warp_shift}'
            elif self._dx_warp_field is not None and self._dy_warp_field is not None :
                msg+=f'{dx_warp_field_path} and {dy_warp_field_path}'
            if args.warping_scalefactor!=1.0 :
                msg+=f' and multiplied by {args.warping_scalefactor}'
            self.__writeLog(msg)
        #set a couple more instance variables
        self._infile_ext = args.input_file_extension
        self._outfile_ext = f'{args.output_file_extension}'
        if self._layer!=-1 :
            self._outfile_ext += f'{self._layer:02d}'
        self._max_files = args.max_files

    def run(self) :
        """
        Read in the rawfile layers, correct them for exposure time, flatfielding, and warping, 
        and write out the corrected file layers to the working directory
        """
        #first get the list of filepaths to run
        with cd(os.path.join(self._rawfile_top_dir,self._sample_name)) :
            all_rawfile_paths = [os.path.join(self._rawfile_top_dir,self._sample_name,fn) for fn in glob.glob(f'*{self._infile_ext}')]
        self.__writeLog(f'Found {len(all_rawfile_paths)} total raw files in {os.path.join(self._rawfile_top_dir,self._sample_name)}')
        if self._max_files!=-1 :
            all_rawfile_paths=all_rawfile_paths[:self._max_files]
            self.__writeLog(f'Will correct and write out {len(all_rawfile_paths)} file layers')
        #next run the correction and copying of the files
        for irfp,rfp in enumerate(all_rawfile_paths,start=1) :
            self._correctAndCopyWorker(rfp,irfp,len(all_rawfile_paths))
        self.__writeLog('All files corrected and copied!')

    #################### PRIVATE HELPER FUNCTIONS ####################

    #helper function to write (and optionally print) a timestamped line to the logfile
    def __writeLog(self,txt,printline=True) :
        line = f'{self.logfile_timestamp}{txt}'
        if printline :
            correction_logger.info(line)
        with cd(self._working_dir_path) :
            with open(self._logfile_name,'a') as fp :
                fp.write(f'{line}\n')

    #helper function to read, correct, and write out a single image layer
    #can be run in parallel
    def _correctAndCopyWorker(self,rawfile_path,file_i,n_total_files) :
        #start up the message of what was done
        msg=''
        if self._layer==-1 :
            msg+=f'all {self._img_dims[-1]} layers '
        else :
            msg+=f'layer {self._layer} '
        msg+=f'of image {rawfile_path} ({file_i} of {n_total_files}) '
        if ( ((self._med_exp_time is not None) and (self._et_correction_offset is not None)) or 
             (self._ff is not None) or 
             ((self._dx_warp_field is not None) and (self._dy_warp_field is not None)) ) :
            msg+='corrected for '
        #first read in the rawfile
        raw = getRawAsHWL(rawfile_path,*(self._img_dims))
        #get the layer of interest (if necessary)
        if self._layer!=-1 :
            raw = raw[:,:,self._layer-1]
        #correct for exposure time differences
        original_et_xml_filepath = findExposureTimeXMLFile(rawfile_path,self._metadata_top_dir)
        if (self._med_exp_time is not None) and (self._et_correction_offset is not None) :
            #correct the file or layer
            if self._layer==-1 :
                et_corrected = correctImageForExposureTime(raw,rawfile_path,self._metadata_top_dir,self._med_exp_time,self._et_correction_offset)
            else :
                layer_exp_time = (getExposureTimesByLayer(rawfile_path,self._img_dims[-1],self._metadata_top_dir))[self._layer-1]
                et_corrected = correctImageLayerForExposureTime(raw,layer_exp_time,self._med_exp_time,self._et_correction_offset)
            #write out the modified exposure time xml file
            with cd(self._working_dir_path) :
                writeModifiedExposureTimeXMLFile(original_et_xml_filepath,self._med_exp_time)
            msg+='exposure time, '
        else :
            copy2(original_et_xml_filepath,self._working_dir_path)
            et_corrected = raw
        #correct with the flatfield
        if self._ff is not None :
            if self._layer==-1 :
                ff_corrected = correctImageWithFlatfield(et_corrected,self._ff)
            else :
                ff_corrected = correctImageLayerWithFlatfield(et_corrected,self._ff)
            msg+='flatfielding, '
        else :
            ff_corrected = et_corrected
        #correct the layers with the warping fields
        if self._warps is not None :
            if self._layer==-1 :
                unwarped = np.zeros_like(ff_corrected)
                for li in range(self._img_dims[-1]) :
                    unwarped[:,:,li] = self._warps[li+1].getWarpedLayer(ff_corrected[:,:,li])
            else :
                unwarped = self._warps[self._layer].getWarpedLayer(ff_corrected)
            msg+='warping, '
        elif (self._dx_warp_field is not None) and (self._dy_warp_field is not None) :
            if self._layer==-1 :
                unwarped = np.zeros_like(ff_corrected)
                for li in range(self._img_dims[-1]) :
                    unwarped[:,:,li] = correctImageLayerWithWarpFields(ff_corrected[:,:,li],self._dx_warp_field,self._dy_warp_field)
            else :
                unwarped = correctImageLayerWithWarpFields(ff_corrected,self._dx_warp_field,self._dy_warp_field)
            msg+='warping, '
        else :
            unwarped = ff_corrected
        #write out the new image to the working directory
        outfile_name = os.path.basename(os.path.normpath(rawfile_path)).replace(self._infile_ext,self._outfile_ext)
        with cd(self._working_dir_path) :
            writeImageToFile(unwarped,outfile_name)
        #double check that it's there
        new_image_path = os.path.join(self._working_dir_path,outfile_name)
        if os.path.isfile(new_image_path) :
            msg+=f'written as {new_image_path}'
        #log the message
        self.__writeLog(msg)

#################### MAIN SCRIPT ####################

if __name__=='__main__' :
    #define and get the command-line arguments
    parser = ArgumentParser()
    #add the common options to the parser
    addCommonArgumentsToParser(parser)
    #add the arguments for shifting the warp pattern
    warp_shift_group = parser.add_mutually_exclusive_group()
    warp_shift_group.add_argument('--warp_shift_file',
                                 help='Path to the warp_shifts.csv file that should be applied to the files in this sample')
    warp_shift_group.add_argument('--warp_shift', 
                                 help='Use this argument to define a (delta-x, delta-y) shift from the inputted warp field')
    #group for other run options
    run_option_group = parser.add_argument_group('run options', 'other options for this run')
    run_option_group.add_argument('--warping_scalefactor',   default=1.0,   type=float,         
                                  help='Scalefactor by which the warping fields should be multiplied before application (default=1.0)')
    run_option_group.add_argument('--layer',                 default=-1,     type=int,         
                                  help='Image layer to use (indexed from 1; default=-1 does all layers)')
    run_option_group.add_argument('--input_file_extension', default='.Data.dat',
                                  help='Extension for the raw files that will be read in (default = ".Data.dat")')
    run_option_group.add_argument('--output_file_extension', default='.fw',
                                  help="""Extension for the corrected files that will be written out 
                                       (default = ".fw"; 2-digit layer code will be appended if layer != -1)""")
    run_option_group.add_argument('--max_files',             default=-1,    type=int,
                                  help='Maximum number of files to use (default = -1 runs all files for the requested sample)')
    run_option_group.add_argument('--logfile_name_stem',     default='correct_and_copy_rawfiles',
                                  help="""Filename stem for the log that will be created (default="correct_and_copy_rawfiles"; 
                                          timestamp and ".log" will be appended)""")
    args = parser.parse_args()
    #start up the corrector from the arguments
    corrector = RawfileCorrector(args)
    #read in, correct, and write out file layers
    corrector.run()
    correction_logger.info('Done.')
