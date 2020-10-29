#imports
from .utilities import correction_logger, getWarpFieldPathsFromWarpDef
from ..warping.warp import CameraWarp
from ..warping.utilities import WarpingSummary, WarpShift
from ..flatfield.config import CONST as FF_CONST
from ..warping.config import CONST as WARP_CONST
from ..utilities.img_correction import correctImageForExposureTime, correctImageLayerForExposureTime
from ..utilities.img_correction import correctImageLayerWithFlatfield, correctImageWithFlatfield, correctImageLayerWithWarpFields
from ..utilities.img_file_io import getImageHWLFromXMLFile, getRawAsHWL, getRawAsHW, writeImageToFile, findExposureTimeXMLFile
from ..utilities.img_file_io import writeModifiedExposureTimeXMLFile, getMedianExposureTimesAndCorrectionOffsetsForSlide
from ..utilities.img_file_io import getMedianExposureTimeAndCorrectionOffsetForSlideLayer, getExposureTimesByLayer, LayerOffset, CORRECTED_EXPOSURE_XML_EXT
from ..utilities.tableio import readtable, writetable
from ..utilities.misc import cd, cropAndOverwriteImage
import numpy as np, matplotlib.pyplot as plt
import os, time, glob

#################### FILE-SCOPE VARIABLES ####################

APPLIED_CORRECTION_PLOT_DIR_NAME = 'applied_correction_plots'

#################### RawfileCorrector CLASS ####################

class RawfileCorrector :

    #################### PROPERTIES ####################
    @property
    def logfile_timestamp(self) :
        return time.strftime("[%Y-%m-%d %H:%M:%S] ")

    #################### PUBLIC FUNCTIONS ####################

    def __init__(self,args,logger=None) :
        """
        args = the set of command line arguments from ArgumentParser
        logger = the baseclasses.logging.MyLogger logger to use (if None, a custom logger with be used)
        """
        #set the rawfile/root directories to use
        self._rawfile_top_dir = args.rawfile_top_dir
        self._root_dir = args.root_dir
        #set the slide ID
        self._slide_ID = args.slideID
        #set the working directory path
        self._working_dir_path = args.workingdir
        #start up the logfile
        self._setUpLogger(logger)
        #get the image dimensions and layer argument
        self._img_dims = getImageHWLFromXMLFile(self._root_dir,self._slide_ID)
        if args.layer!=-1 and (args.layer<1 or args.layer>self._img_dims[2]) :
            raise ValueError(f'ERROR: layer argument {args.layer} is not compatible with image dimensions {self._img_dims}!')
        self._layer = args.layer
        if self._layer==-1 :
            self.__writeLog(f'Corrected {self._img_dims[-1]}-layer files will be written out to {self._working_dir_path}')
        else :
            self.__writeLog(f'Corrected layer {self._layer} files will be written out to {self._working_dir_path}.')
        #see which layer(s) will be run
        layers_to_run = list(range(1,self._img_dims[-1]+1)) if self._layer==-1 else [self._layer]
        #set the median slide exposure time and dark current offsets
        self._setExposureTimeVariables(args.skip_exposure_time_correction,args.exposure_time_offset_file)
        #set the flatfield variable
        self._setFlatfieldVariable(args.skip_flatfielding,args.flatfield_file,layers_to_run)
        #set the warping correction variables
        self._setWarpingVariables(args.skip_warping,args.warp_def,args.warp_shift_file,args.warp_shift,args.warping_scalefactor,layers_to_run)
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
        with cd(os.path.join(self._rawfile_top_dir,self._slide_ID)) :
            all_rawfile_paths = [os.path.join(self._rawfile_top_dir,self._slide_ID,fn) for fn in glob.glob(f'*{self._infile_ext}')]
        self.__writeLog(f'Found {len(all_rawfile_paths)} total raw files in {os.path.join(self._rawfile_top_dir,self._slide_ID)}')
        if self._max_files!=-1 :
            if self._max_files>len(all_rawfile_paths) :
                msg = f'only {len(all_rawfile_paths)} were found for {self._slide_ID}, but {self._max_files} were requested,'
                msg+=f' so all {len(all_rawfile_paths)} files will be run for this slide instead.'
                self.__writeLog(msg,level='warning')
            else :
                all_rawfile_paths=all_rawfile_paths[:self._max_files]
            msg = f'Will correct and write out {len(all_rawfile_paths)} file'
            if self._layer==-1 :
                msg+='s'
            else :
                msg+=' layers'
            self.__writeLog(msg)
        #next run the correction and copying of the files
        self.__writeLog('Starting loop over files')
        for irfp,rfp in enumerate(all_rawfile_paths,start=1) :
            try :
                self._correctAndCopyWorker(rfp,irfp,len(all_rawfile_paths))
            except Exception as e :
                self.__writeLog(f'correcting/copying file {rfp} FAILED with exception: {e}',level='warningglobal')
        self.__writeLog('Done looping over files!')

    #################### PRIVATE HELPER FUNCTIONS ####################

    #helper function to write (and optionally print) a timestamped line to the logfile
    def __writeLog(self,txt,level='info',printline=True) :
        #try to write using the context-managed logger object
        if self._logger_obj is not None and self._logger_fn is None :
            if level=='info' :
                self._logger_obj.info(txt)
            elif level=='imageinfo' :
                self._logger_obj.imageinfo(txt)
            elif level=='error' :
                self._logger_obj.error(txt)
            elif level=='warningglobal' :
                self._logger_obj.warningglobal(txt)
            elif level=='warning' :
                self._logger_obj.warning(txt)
            elif level=='debug' :
                self._logger_obj.debug(txt)
            else :
                raise ValueError(f'ERROR: logger level {level} is not recognized!')
        #otherwise write to the custom file
        else :
            if level not in ('info','imageinfo','error','warningglobal','warning','debug') :
                raise ValueError(f'ERROR: logger level {level} is not recognized!')
            line = f'{self.logfile_timestamp}'
            if level=='error' :
                line+='ERROR: '
            elif level in ('warningglobal','warning') :
                line+='WARNING: '
            line+=f'{txt}'
            if printline or level=='debug' :
                correction_logger.info(line)
            if level!='debug' :
                with cd(self._working_dir_path) :
                    with open(self._logger_fn,'a') as fp :
                        fp.write(f'{line}\n')

    #helper function to get either the logger object or the name of the logfile (one will be None) for the corrector
    def _setUpLogger(self,input_logger) :
        self._logger_obj = None; self._logger_fn = None
        if input_logger is None :
            logfile_name = f'correct_and_copy_rawfiles_{time.strftime("%Y_%m_%d-%H_%M_%S")}.log'
            with cd(self._working_dir_path) :
                with open(logfile_name,'w') as fp :
                    fp.write('LOGFILE for correct_and_copy_rawfiles\n')
                    fp.write('-------------------------------------\n\n')
            self._logger_fn = logfile_name
        else :
            self._logger_obj = input_logger
        msg = f'Working directory {os.path.basename(os.path.normpath(self._working_dir_path))} has been created'
        msg+= f' in {os.path.dirname(os.path.abspath(os.path.normpath(self._working_dir_path)))}.'
        self.__writeLog(msg)

    #helper function to set the exposure time correction variables
    def _setExposureTimeVariables(self,skip_etc,eto_file) :
        if skip_etc :
            self._med_exp_time = None; self._et_correction_offset = None
            self.__writeLog('Corrections for exposure time WILL NOT be applied.')
            return
        if self._layer==-1 :
            self._med_exp_time, self._et_correction_offset = getMedianExposureTimesAndCorrectionOffsetsForSlide(self._root_dir,
                                                                                                                 self._slide_ID,
                                                                                                                 eto_file)
            los = [LayerOffset(li+1,-1,self._et_correction_offset[li],-1.) for li in range(self._img_dims[-1])]
        else :
            self._med_exp_time, self._et_correction_offset = getMedianExposureTimeAndCorrectionOffsetForSlideLayer(self._root_dir,
                                                                                                                    self._slide_ID,
                                                                                                                    eto_file,
                                                                                                                    self._layer)
            los = [LayerOffset(self._layer,-1,self._et_correction_offset,-1.)]
        try :
            with cd(self._working_dir_path) :
                if not os.path.isdir(APPLIED_CORRECTION_PLOT_DIR_NAME) :
                    os.mkdir(APPLIED_CORRECTION_PLOT_DIR_NAME)
                with cd(APPLIED_CORRECTION_PLOT_DIR_NAME) :
                    writetable('applied_exposure_time_correction_offsets.csv',los)
        except Exception as e :
            self.__writeLog(f'applied exposure time offset file could not be written out. Exception: {e}',level='warning')
        self.__writeLog(f'Exposure time corrections WILL be applied based on offset factors in {eto_file}')
        self.__writeLog(f'Corrected *{CORRECTED_EXPOSURE_XML_EXT} files will be written out to {self._working_dir_path}')
        if self._layer==-1 :
            for ln in range(1,self._img_dims[-1]+1) :
                msg = f'(Layer {ln} median slide exposure time={self._med_exp_time[ln-1]},'
                msg+= f' exposure time correction offset = {self._et_correction_offset[ln-1]})'
                self.__writeLog(msg)
        else :
            self.__writeLog(f'(Median slide exposure time={self._med_exp_time}, exposure time correction offset = {self._et_correction_offset})')

    #helper function to set the flatfield correction variable
    def _setFlatfieldVariable(self,skip_ff,ff_file,layers_to_run) :
        if skip_ff :
            self._ff = None
            self.__writeLog('Flatfielding corrections WILL NOT be applied.')
            return
        self._ff = getRawAsHWL(ff_file,*(self._img_dims),dtype=FF_CONST.IMG_DTYPE_OUT)
        if self._layer!=-1 :
            self._ff = self._ff[:,:,self._layer-1]
        try :
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
                        savename = f'applied_flatfield_layer_{ln}.png'
                        plt.savefig(savename)
                        plt.close()
                        cropAndOverwriteImage(savename)
        except Exception as e :
            self.__writeLog(f'applied flatfield plots could not be saved. Exception: {e}',level='warning')
        self.__writeLog(f'Flatfield corrections WILL be applied as read from {ff_file}')

    #helper function to set the warping variables
    def _setWarpingVariables(self,skip_w,w_def,ws_file,arg_ws,w_sf,layers_to_run) :
        if skip_w :
            self._warps = None; self._dx_warp_field = None; self._dy_warp_field = None
            self.__writeLog('Warping corrections WILL NOT be applied.')
            return
        msg='Warping corrections will be applied as read from '
        with cd(self._working_dir_path) :
            if not os.path.isdir(APPLIED_CORRECTION_PLOT_DIR_NAME) :
                os.mkdir(APPLIED_CORRECTION_PLOT_DIR_NAME)
        #first try to define the warping from a parameter file
        if w_def.endswith('.csv') :
            msg+=f'{w_def}'
            warp_fit_result = readtable(w_def,WarpingSummary)
            if len(warp_fit_result)>1 :
                raise ValueError(f'ERROR: warp fit result file {w_def} has more than one set of parameters!')
            wfr = warp_fit_result[0]
            warp_shifts = []
            if ws_file is not None :
                try :
                    warp_shifts = readtable(ws_file,WarpShift)
                except Exception as e :
                    raise ValueError(f'ERROR: file {ws_file} is not recognized as a set of WarpShift objects! Exception: {e}')
                msg+=f' and shifted as read from {ws_file}'
            elif arg_ws is not None :
                cx_shift,cy_shift = arg_ws.split(',')
                cx_shift = float(cx_shift); cy_shift = float(cy_shift)
                for ln in layers_to_run :
                    warp_shifts.append(WarpShift(ln,cx_shift,cy_shift))
                msg+=f' and shifted by {arg_ws}'
            if self._layer!=-1 :
                warp_shifts = [ws for ws in warp_shifts if ws.layer_n==self._layer]
            if len(warp_shifts)>0 :
                with cd(os.path.join(self._working_dir_path,APPLIED_CORRECTION_PLOT_DIR_NAME)) :
                    writetable('applied_warp_shifts.csv',warp_shifts)
            self._warps = {}
            for ln in layers_to_run :
                cx_shift = 0.; cy_shift = 0.
                if ln in [ws.layer_n for ws in warp_shifts] :
                    this_ws = ([ws for ws in warp_shifts if ws.layer_n==ln])[0]
                    cx_shift = this_ws.cx_shift; cy_shift = this_ws.cy_shift
                self._warps[ln] = CameraWarp(self._img_dims[1],self._img_dims[0],wfr.cx+cx_shift,wfr.cy+cy_shift,
                                             wfr.fx,wfr.fy,w_sf*wfr.k1,w_sf*wfr.k2,w_sf*wfr.k3,w_sf*wfr.p1,w_sf*wfr.p2)
                try :
                    fs = f'applied_warping_correction_layer_{ln}'
                    with cd(os.path.join(self._working_dir_path,APPLIED_CORRECTION_PLOT_DIR_NAME)) :
                        self._warps[ln].writeOutWarpFields(fs,save_fields=False)
                except Exception as e :
                    self.__writeLog(f'applied warp field plots could not be saved. Exception: {e}',level='warning')
        #otherwise try to define the fields by the actual .bin files
        else :
            dx_warp_field_path, dy_warp_field_path = getWarpFieldPathsFromWarpDef(w_def)
            if not os.path.isfile(dx_warp_field_path) :
                raise FileNotFoundError(f'ERROR: dx warp field path {dx_warp_field_path} does not exist!')
            if not os.path.isfile(dy_warp_field_path) :
                raise FileNotFoundError(f'ERROR: dy warp field path {dy_warp_field_path} does not exist!')
            self._dx_warp_field = (w_sf)*(getRawAsHW(dx_warp_field_path,*(self._img_dims[:-1]),dtype=WARP_CONST.OUTPUT_FIELD_DTYPE))
            self._dy_warp_field = (w_sf)*(getRawAsHW(dy_warp_field_path,*(self._img_dims[:-1]),dtype=WARP_CONST.OUTPUT_FIELD_DTYPE))
            msg+=f'{dx_warp_field_path} and {dy_warp_field_path}'
            try :
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
                    savename = 'applied_warping_correction_model.png'
                    plt.savefig(savename)
                    plt.close()
                    cropAndOverwriteImage(savename)
            except Exception as e :
                self.__writeLog(f'applied warp field plots could not be saved. Exception: {e}',level='warning')
        if w_sf!=1.0 :
                msg+=f' and multiplied by {w_sf}'
        self.__writeLog(msg)

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
        original_et_xml_filepath = findExposureTimeXMLFile(rawfile_path,self._root_dir)
        if (self._med_exp_time is not None) and (self._et_correction_offset is not None) :
            #correct the file or layer
            if self._layer==-1 :
                et_corrected = correctImageForExposureTime(raw,rawfile_path,self._root_dir,self._med_exp_time,self._et_correction_offset)
            else :
                layer_exp_times = getExposureTimesByLayer(rawfile_path,self._img_dims[-1],self._root_dir)
                layer_exp_time = layer_exp_times[self._layer-1]
                et_corrected = correctImageLayerForExposureTime(raw,layer_exp_time,self._med_exp_time,self._et_correction_offset)
            #write out the modified exposure time xml file
            with cd(self._working_dir_path) :
                if self._layer==-1 :
                    writeModifiedExposureTimeXMLFile(original_et_xml_filepath,self._med_exp_time,logger=self._logger_obj)
                else :
                    exp_times_to_write = layer_exp_times
                    layer_exp_times[self._layer-1] = self._med_exp_time
                    writeModifiedExposureTimeXMLFile(original_et_xml_filepath,exp_times_to_write,logger=self._logger_obj)
            msg+='exposure time, '
        else :
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
        self.__writeLog(msg,level='imageinfo')
