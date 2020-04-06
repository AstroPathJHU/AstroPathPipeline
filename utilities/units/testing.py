import numpy as np
from .core import pixels

def assert_allclose(distance1, distance2, *args, **kwargs):
  np.testing.assert_allclose(pixels(distance1), pixels(distance2), *args, **kwargs)
