def __setup(mode):
  global inv, solve
  if mode == "safe":
    from .safe.linalg import inv, solve
  elif mode == "pixels" or mode == "microns":
    from numpy.linalg import inv, solve
  else:
    raise ValueError(f"Invalid mode {mode}")
