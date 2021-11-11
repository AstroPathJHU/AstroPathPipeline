import collections, contextlib, numpy as np, PIL.Image

class PILmaximagepixels(contextlib.AbstractContextManager):
  """
  Context manager to increase the maximum image pixels for PIL.
  Using this context manager will never decrease the maximum pixels.
  You can also set it to None, which is equivalent to infinity.
  """
  __maximagepixelscounter = collections.Counter()
  __defaultmaximagepixels = PIL.Image.MAX_IMAGE_PIXELS
  def __init__(self, maximagepixels):
    self.__maximagepixels = maximagepixels
  def __enter__(self):
    self.__maximagepixelscounter[self.__maximagepixels] += 1
    self.__updatemaximagepixels()
  def __exit__(self, *exc):
    self.__maximagepixelscounter[self.__maximagepixels] -= 1
    self.__updatemaximagepixels()
  @classmethod
  def __updatemaximagepixels(cls):
    elements = set(cls.__maximagepixelscounter.elements()) | {cls.__defaultmaximagepixels}
    if None in elements:
      PIL.Image.MAX_IMAGE_PIXELS = None
    else:
      PIL.Image.MAX_IMAGE_PIXELS = max(elements)

def vips_format_dtype(format_or_dtype):
  """
  https://libvips.github.io/pyvips/intro.html#numpy-and-pil
  """
  result = {
    'uchar': np.uint8,
    'char': np.int8,
    'ushort': np.uint16,
    'short': np.int16,
    'uint': np.uint32,
    'int': np.int32,
    'float': np.float32,
    'double': np.float64,
    'complex': np.complex64,
    'dpcomplex': np.complex128,
  }
  for k, v in list(result.items()):
    result[v] = k
    result[np.dtype(v)] = k
  return result[format_or_dtype]

def vips_image_to_array(img, *, singlelayer=True):
  """
  https://libvips.github.io/pyvips/intro.html#numpy-and-pil
  """
  shape = [img.height, img.width, img.bands]
  if singlelayer:
    if shape[-1] != 1: raise ValueError("Have to write singlelayer=False if the image has more than one channel")
    del shape[-1]
  return np.ndarray(
    buffer=img.write_to_memory(),
    dtype=vips_format_dtype(img.format),
    shape=shape,
  )

def array_to_vips_image(array):
  """
  https://libvips.github.io/pyvips/intro.html#numpy-and-pil
  """
  try:
    import pyvips
  except ImportError:
    raise ImportError("Please pip install pyvips to use this functionality")

  if len(array.shape) == 2:
    height, width = array.shape
    bands = 1
  else:
    height, width, bands = array.shape

  return pyvips.Image.new_from_memory(
    array,
    format=vips_format_dtype(array.dtype),
    width=width,
    height=height,
    bands=bands,
  )

def vips_sinh(image):
  try:
    #https://github.com/libvips/pyvips/pull/282
    return image.sinh()
  except AttributeError:
    exp = image.exp()
    minusexp = 1/exp
    return (exp - minusexp) / 2
