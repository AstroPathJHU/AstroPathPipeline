import pyopencl, reikna as rk

def get_GPU_thread(interactive,logger) :
  """
  Create and return a Reikna Thread object to use for running some computations on the GPU
  If the Thread can't be created, a globa warning is logged and None is returned

  interactive : if True (and some GPU is available), user will be given the option to choose a device 
  logger : used to log a warning if the GPU thread can't be created
  """
  api = rk.cluda.ocl_api()
  #return a thread from the API
  try :
    thread = api.Thread.create(interactive=interactive)
    return thread
  except pyopencl._cl.LogicError :
    warnmsg = 'WARNING: A GPU Thread could not be created using PyOpenCL and Reikna. '
    warnmsg+= 'Please make sure an OpenCL-compatible GPU is available and that the OpenCL driver for it is installed. '
    warnmsg+= 'GPU computation will be disabled. Rerun with "--noGPU" to remove this warning.'
    logger.warningglobal(warnmsg)
    return None
