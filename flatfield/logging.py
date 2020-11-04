#imports
from ..baseclasses.sample import SampleDef
from ..baseclasses.logging import getlogger
from contextlib import ExitStack
import os, logging, traceback

class RunLogger(ExitStack) :
    """
    Class for the logger used by an entire run
    """

    #################### PROPERTIES ####################
    @property
    def formatter(self) :
        return logging.Formatter("-1;-1;-1;None;%(message)s;%(asctime)s","%Y-%m-%d %H:%M:%S")

    #################### OVERLOADED RESERVED METHODS ####################

    def __init__(self,mode,workingdir_path) :
        """
        mode            = the mode the code is running in, which is the module name for any sample logging
        workingdir_path = path to the working directory for the run
        """
        super().__init__()
        self._module = mode
        self._batch_mode = self._module in ('slide_mean_image','batch_flatfield')
        self._workingdir_path = workingdir_path
        self._global_logger = self._getGlobalLogger()
        self._slide_loggers = {}

    def __enter__(self) :
        super().__enter__()
        #add the imageinfo-level file in the working directory
        self._global_logger_filepath = os.path.join(self._workingdir_path,f'global-{self._module}.log')
        filehandler = logging.FileHandler(self._global_logger_filepath)
        filehandler.setFormatter(self.formatter)
        filehandler.setLevel(logging.INFO-1)
        self._global_logger.addHandler(filehandler)
        self._global_logger.info(f'{self._module}')
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback) :
        super().__exit__(exc_type,exc_value,exc_traceback)
        if exc_value is not None:
            errmsg = str(exc_value).replace(";", ",")
            infomsg = repr(traceback.format_exception(exc_type, exc_value, exc_traceback)).replace(";", "")
            self._global_logger.error(errmsg)
            self._global_logger.info(infomsg)
        self._global_logger.info(f'end {self._module}')
        return True #Don't ever fully crash (other loggers will handle that if not in batch mode)

    #################### PUBLIC FUNCTIONS ####################

    #The below functions write messages at different levels to the logger. 
    #If they are called with slideID and slide_root None, use the relevant SlideLogger instead of the global logger
    def info(self,msg,slideID=None,slide_root=None) :
        self._doLog('info',msg,slideID,slide_root)
    def imageinfo(self,msg,slideID=None,slide_root=None) :
        self._doLog('imageinfo',msg,slideID,slide_root)
    def error(self,msg,slideID=None,slide_root=None) :
        self._doLog('error',msg,slideID,slide_root)
    def warningglobal(self,msg,slideID=None,slide_root=None) :
        self._doLog('warningglobal',msg,slideID,slide_root)
    def warning(self,msg,slideID=None,slide_root=None) :
        self._doLog('warning',msg,slideID,slide_root)
    def debug(self,msg,slideID=None,slide_root=None) :
        self._doLog('debug',msg,slideID,slide_root)

    #################### PRIVATE HELPER FUNCTIONS ####################

    #helper function to set up the global logger, which is a debug-level print to start with
    def _getGlobalLogger(self) :
        logger = logging.getLogger(self._module)
        logger.setLevel(logging.DEBUG)
        printhandler = logging.StreamHandler()
        printhandler.setFormatter(self.formatter)
        printhandler.setLevel(logging.DEBUG)
        logger.addHandler(printhandler)
        return logger

    #helper function to add a new single slide logger to the global logger's dictionary
    def _addSlideLogger(self,slideid,root_dir) :
        samp = SampleDef(SlideID=slideid,root=root_dir)
        mainlog = os.path.join(self._workingdir_path,f'{self._module}.log') if not self._batch_mode else None
        samplelog = os.path.join(self._workingdir_path,f'{slideid}-{self._module}.log') if not self._batch_mode else None
        newlogger = getlogger(module=self._module,root=root_dir,samp=samp,uselogfiles=True,mainlog=mainlog,samplelog=samplelog,
                              imagelog=self._global_logger_filepath,reraiseexceptions=(not self._batch_mode))
        self.enter_context(newlogger)
        self._slide_loggers[slideid] = newlogger

    #helper function to find which log to use and send a message at a given level to it
    def _doLog(self,level,msg,slideID,slide_root) :
        if slideID is None and slide_root is None :
            logger = self._global_logger
        else :
            if slideID not in self._slide_loggers.keys() :
                self._addSlideLogger(slideID,slide_root)
            logger = self._slide_loggers[slideID]
        if level=='info' :
            logger.info(msg)
        elif level=='imageinfo' :
            logger.log(logging.INFO-1,msg)
        elif level=='error' :
            for logger in [self._global_logger]+list(self._slide_loggers.values()) :
                logger.error(msg)
        elif level=='warningglobal' :
            for logger in [self._global_logger]+list(self._slide_loggers.values()) :
                logger.log(logging.WARNING+1,msg)
        elif level=='warning' :
            logger.warning(msg)
        elif level=='debug' :
            logger.debug(msg)
        else :
            raise ValueError(f'ERROR: logger level {level} is not recognized!')
