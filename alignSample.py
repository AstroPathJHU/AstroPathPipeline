import logging

logger = logging.getLogger("align")
logger.setFormatter(Formatter("%(message)s, %(funcname)s, %(asctime)s"))