def __setup(mode):
  global assert_allclose
  if mode == "safe":
    from .safe.testing import assert_allclose
  elif mode == "fast":
    from numpy.testing import assert_allclose
  else:
    raise ValueError(f"Invalid mode {mode}")
