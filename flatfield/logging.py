#imports
from ..baseclasses.sample import SampleDef
from ..baseclasses.logging import getlogger
import logging

class RunLogger :
    """
    Class for the logger used by an entire run
    """

    #################### OVERLOADED PUBLIC FUNCTIONS ####################

    def __init__(self,mode,workingdir_path) :
        """
        mode            = the mode the code is running in, which is the module name for any sample logging
        workingdir_path = path to the working directory for the run
        """
        self._module = mode
        self._batch_mode = self._module in ('slide_mean_image')
        self._workingdir_path = workingdir_path
        self._global_logger, self._global_logger_filepath = self._getGlobalLogger()
        self._slide_loggers = {}

    def __enter__(self) :
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback) :
        if exc_value is not None:
            errmsg = str(exc_value).replace(";", ",")
            infomsg = repr(traceback.format_exception(exc_type, exc_value, exc_traceback)).replace(";", "")
            for logger in [self._global_logger]+list(self._slide_loggers.values()) :
                logger.error(errmsg)
                logger.info(infomsg)
                logger.info(f"end {self.module}")
        return True #Don't ever fully crash (other loggers will handle that if not in batch mode)

    #################### PUBLIC FUNCTIONS ####################

    #write an info message to the logger. If slideID and slide_root are not None, use the relevant SlideLogger instead of the global logger
    def info(self,msg,slideID=None,slide_root=None) :
        if slideID is None and slide_root is None :
            self._global_logger.info(msg)
        else :
            if slideID not in self._slide_loggers.keys() :
                self._addSlideLogger(slideID,slide_root)
            self._slide_loggers[slideID].info(msg)

    #################### PRIVATE HELPER FUNCTIONS ####################

    #helper function to set up the global logger, which is a debug-level print and an info-level file in the working directory
    def _getGlobalLogger(self) :
        logging.getLogger(self._module)
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter("-1;-1;-1;None;%(message)s;%(asctime)s","%Y-%m-%d %H:%M:%S")
        printhandler = logging.StreamHandler()
        printhandler.setFormatter(formatter)
        printhandler.setLevel(logging.DEBUG)
        logger.addHandler(printhandler)
        logger_filepath = os.path.join(workingdir_path,f'global-{self._module}.log')
        filehandler = logging.FileHandler(logger_filepath)
        filehandler.setFormatter(formatter)
        filehandler.setLevel(logging.INFO)
        logger.addHandler(filehandler)
        return logger, logger_filepath

    #helper function to add a new single slide logger to the global logger's dictionary
    def _addSlideLogger(self,slideid,root_dir) :
        samp = SampleDef(SlideID=slideid,root=root_dir)
        newlogger = getLogger(module=self._module,root=root_dir,samp=samp,uselogfiles=True,
                              ########### comment out the two lines below to actually put the logs in the right place ################
                              mainlog=os.path.join(self._workingdir_path,f'{self._module}.log'),
                              samplelog=os.path.join(self._workingdir_path,f'{slideid}-{self._module}.log'),
                              ########### comment out the two lines above to actually put the logs in the right place ################
                              imagelog=self._global_logger_filepath,reraiseexceptions=(not self._batch_mode))
        newlogger.critical(self._module)
        self._slide_loggers[slideid] = newlogger
