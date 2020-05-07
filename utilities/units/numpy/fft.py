def __setup(mode):
  global fft
  if mode == "safe":
    from ..safe.numpy.fft import fft
  elif mode == "fast":
    from numpy.fft import fft
  else:
    raise ValueError(f"Invalid mode {mode}")
