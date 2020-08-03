import abc, collections, dataclasses, datetime, methodtools, numpy as np, pathlib
from ..utilities import units
from ..utilities.misc import dataclass_dc_init, memmapcontext
from ..utilities.units.dataclasses import DataClassWithDistances, distancefield

@dataclasses.dataclass
class Rectangle(DataClassWithDistances):
  pixelsormicrons = "microns"

  n: int = dataclasses.field()
  x: units.Distance = distancefield(pixelsormicrons=pixelsormicrons)
  y: units.Distance = distancefield(pixelsormicrons=pixelsormicrons)
  w: units.Distance = distancefield(pixelsormicrons=pixelsormicrons, dtype=int)
  h: units.Distance = distancefield(pixelsormicrons=pixelsormicrons, dtype=int)
  cx: units.Distance = distancefield(pixelsormicrons=pixelsormicrons, dtype=int)
  cy: units.Distance = distancefield(pixelsormicrons=pixelsormicrons, dtype=int)
  t: datetime.datetime = dataclasses.field(metadata={"readfunction": lambda x: datetime.datetime.fromtimestamp(int(x)), "writefunction": lambda x: int(datetime.datetime.timestamp(x))})
  file: str
  pscale: dataclasses.InitVar[float]
  readingfromfile: dataclasses.InitVar[bool]

  @property
  def xvec(self):
    return np.array([self.x, self.y])

  @property
  def cxvec(self):
    return np.array([self.cx, self.cy])

  @property
  def shape(self):
    return np.array([self.w, self.h])

@dataclass_dc_init
class RectangleWithLayer(Rectangle):
  layer: dataclasses.InitVar[int]
  def __init__(self, *args, rectangle=None, **kwargs):
    rectanglekwargs = {}
    if rectangle is not None:
      rectanglekwargs = {
        "pscale": rectangle.pscale,
        **{
          field.name: getattr(rectangle, field.name)
          for field in dataclasses.fields(type(rectangle))
        }
      }
      if hasattr(rectangle, "layer"):
        rectanglekwargs["layer"] = rectangle.layer
      rectanglekwargs = {kw: kwarg for kw, kwarg in rectanglekwargs.items() if kw not in kwargs}
    return self.__dc_init__(
      *args,
      **rectanglekwargs,
      **kwargs,
    )

  def __post_init__(self, pscale, readingfromfile, layer, *args, **kwargs):
    super().__post_init__(pscale=pscale, readingfromfile=readingfromfile, *args, **kwargs)
    self.layer = layer

class RectangleWithImage(RectangleWithLayer):
  def __init__(self, *args, imagefolder, filetype, width, height, readlayerfile=True, **kwargs):
    super().__init__(*args, **kwargs)
    self.__imagefolder = pathlib.Path(imagefolder)
    self.__filetype = filetype
    self.__readlayerfile = readlayerfile
    self.__width = width
    self.__height = height

  @property
  def imageshape(self): return units.pixels(self.__height, pscale=self.pscale), units.pixels(self.__width, pscale=self.pscale)

  #do not override this property
  #override getimage() instead and call super().getimage()
  @methodtools.lru_cache()
  @property
  def image(self):
    return self.getimage()

  def getimage(self):
    if self.__filetype=="flatWarp" :
      if self.__readlayerfile :
        ext = f".fw{self.layer:02d}"
      else:
        ext = ".fw"
    elif self.__filetype=="camWarp" :
      if self.__readlayerfile:
        ext = f".camWarp_layer{self.layer:02d}"
      else:
        ext = ".camWarp"
    elif self.__filetype=="raw" :
      if self.__readlayerfile:
        ext = f".Data.dat_layer{self.layer:02d}"
      else:
        ext = ".Data.dat"
    else :
      raise ValueError(f"requested file type {self.__filetype} not recognized")

    image = np.ndarray(shape=self.imageshape, dtype=np.uint16)

    if self.__readlayerfile:
      shape = units.pixels((self.__height, self.__width), pscale=self.pscale)
      transpose = (0, 1)
      slc = slice(None), slice(None)
    else:
      shape = units.pixels((self.nlayers, self.__width, self.__height), pscale=self.pscale, power=[0, 1, 1])
      transpose = (2, 1, 0)
      slc = slice(None), slice(None), self.layer-1

    filename = self.__imagefolder/self.file.replace(".im3", ext)
    with open(filename, "rb") as f:
      #use fortran order, like matlab!
      with memmapcontext(
        f,
        dtype=np.uint16,
        shape=tuple(shape),
        order="F",
        mode="r"
      ) as memmap:
        image[:] = memmap.transpose(transpose)[slc]

    return image

class RectangleCollection(abc.ABC):
  @abc.abstractproperty
  def rectangles(self): pass
  @methodtools.lru_cache()
  def __rectangledict(self):
    return rectangledict(self.rectangles)
  @property
  def rectangledict(self): return self.__rectangledict()
  @property
  def rectangleindices(self):
    return {r.n for r in self.rectangles}

class RectangleList(list, RectangleCollection):
  @property
  def rectangles(self): return self

def rectangledict(rectangles):
  return {rectangle.n: i for i, rectangle in enumerate(rectangles)}

def rectangleoroverlapfilter(selection, *, compatibility=False):
  if compatibility:
    if selection == -1:
      selection = None
    if isinstance(selection, tuple):
      if len(selection) == 2:
        selection = range(selection[0], selection[1]+1)
      else:
        selection = str(selection) #to get the right error message below

  if selection is None:
    return lambda r: True
  elif isinstance(selection, collections.abc.Collection) and not isinstance(selection, str):
    return lambda r: r.n in selection
  elif isinstance(selection, collections.abc.Callable):
    return selection
  else:
    raise ValueError(f"Unknown rectangle or overlap selection: {selection}")

rectanglefilter = rectangleoroverlapfilter
