def __setup(mode):
  global curve_fit
  if mode == "safe":
    from .safe.optimize import curve_fit
  elif mode == "fast":
    from scipy.optimize import curve_fit
  else:
    raise ValueError(f"Invalid mode {mode}")
