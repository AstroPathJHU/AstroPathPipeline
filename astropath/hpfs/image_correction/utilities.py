#imports
from ...utilities.config import CONST as UNIV_CONST
import logging, os

#logger
correction_logger = logging.getLogger("correct_and_copy")
correction_logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s  [%(funcName)s]","%Y-%m-%d %H:%M:%S"))
correction_logger.addHandler(handler)

#helper function to get the binary warp shift field filepaths from the warp_def argument
def getWarpFieldPathsFromWarpDef(w_def) :
    wf_dirname = os.path.basename(os.path.normpath(w_def))
    dx_warp_field_path = os.path.join(w_def,f'{UNIV_CONST.X_WARP_BIN_FILENAME}_{wf_dirname}.bin')
    dy_warp_field_path = os.path.join(w_def,f'{UNIV_CONST.Y_WARP_BIN_FILENAME}_{wf_dirname}.bin')
    return dx_warp_field_path, dy_warp_field_path