def __setup(mode):
  global inv, solve
  if mode == "safe":
    from ..safe.numpy.linalg import inv, solve
  elif mode == "fast":
    from numpy.linalg import inv, solve
  else:
    raise ValueError(f"Invalid mode {mode}")
