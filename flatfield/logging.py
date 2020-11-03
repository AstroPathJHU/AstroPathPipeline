#imports
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
        self._global_logger = self._getGlobalLogger()
        self._slide_loggers = {}

    def __enter__(self) :
        return self

    #################### PUBLIC FUNCTIONS ####################

    #write an info message to the logger. If slideID is not None, use the relevant SlideLogger instead of the global logger
    def info(self,msg,slideID=None) :
        if slideID is None :
            self._global_logger.info(msg)
        else :
            pass

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
        filehandler = logging.FileHandler(os.path.join(workingdir_path,f'global-{self._module}.log'))
        filehandler.setFormatter(formatter)
        filehandler.setLevel(logging.INFO)
        logger.addHandler(filehandler)
        return logger