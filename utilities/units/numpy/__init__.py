from . import fft, linalg, testing

def __setup(*args, **kwargs):
  for _ in fft, linalg, testing:
    _.__setup(*args, **kwargs)
