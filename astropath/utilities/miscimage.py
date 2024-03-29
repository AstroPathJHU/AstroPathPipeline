import collections, contextlib, integv, numpy as np, PIL.Image
from .optionalimports import pyvips

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

def vips_sinh(image, *, allowfallback=True):
  """
  >>> import os
  >>> i = array_to_vips_image(np.arange(25, dtype=float).reshape(5, 5) / 10)
  >>> vips_image_to_array(vips_sinh(i, allowfallback=not bool(int(os.environ.get("RUNNING_ON_JENKINS", 0)))))
  array([[0.        , 0.10016675, 0.201336  , 0.30452029, 0.41075233],
         [0.52109531, 0.63665358, 0.7585837 , 0.88810598, 1.02651673],
         [1.17520119, 1.33564747, 1.50946136, 1.69838244, 1.9043015 ],
         [2.12927946, 2.37556795, 2.64563193, 2.94217429, 3.26816291],
         [3.62686041, 4.02185674, 4.45710517, 4.93696181, 5.46622921]])
  """
  try:
    #https://github.com/libvips/pyvips/pull/282
    return image.sinh()
  except AttributeError:
    if not allowfallback: raise
    fallback = True
  except pyvips.error.Error as e:
    if not allowfallback: raise
    if 'VipsOperation: class "sinh" not found' in str(e) or "enum 'VipsOperationMath' has no member 'sinh'" in str(e):
      fallback = True
    else:
      raise

  assert fallback
  exp = image.exp()
  minusexp = 1/exp
  return (exp - minusexp) / 2

def check_image_integrity(filename, *, remove, error, logger=None):
  """
  Checks whether the image file is valid and returns True if it is
  or False if it's not.

    error: raises an error if the file is not valid instead of returning False
    remove: also removes the image file if not valid
    logger: logger to print a warning message if remove=True and error=False
  """

  if not filename.exists(): return False

  if remove and not error and logger is None:
    raise TypeError("For check_image_integrity(remove=True, error=False), have to provide a logger")

  if integv.verify(filename, file_type=filename.suffix):
    return True
  else:
    message = f"{filename} is corrupt"
    if remove:
      message += ", removing it"

    if remove:
      filename.unlink()

    if error:
      raise IOError(message)
    elif logger is not None:
      logger.warning(message)

    return False

class TIFFIntegrityVerifier(integv._IntegrityVerifierBase):
  MIME = "image/tiff"
  def verify(self, file):
    try:
      with PIL.Image.open(file) as im:
        np.asarray(im)
    except (IOError, OSError):
      return False
    else:
      return True

#patch integv to work with Pillow >= 9.5.0
def readline(self, *args, **kwargs):
  return self._file.readline(*args, **kwargs)
integv._file.NormalizedFile.readline = readline
del readline
