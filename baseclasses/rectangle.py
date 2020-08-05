import abc, collections, dataclasses, datetime, methodtools, numpy as np, pathlib
from ..utilities import units
from ..utilities.misc import dataclass_dc_init, memmapcontext
from ..utilities.units.dataclasses import DataClassWithDistances, distancefield

@dataclass_dc_init
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
      if "pscale" in kwargs:
        del rectanglekwargs["pscale"]
    return self.__dc_init__(
      *args,
      **rectanglekwargs,
      **kwargs,
    )

  @property
  def xvec(self):
    return np.array([self.x, self.y])

  @property
  def cxvec(self):
    return np.array([self.cx, self.cy])

  @property
  def shape(self):
    return np.array([self.w, self.h])

class RectangleWithImageBase(Rectangle):
  #do not override this property
  #override getimage() instead and call super().getimage()
  @methodtools.lru_cache()
  @property
  def image(self):
    return self.getimage()

  @abc.abstractmethod
  def getimage(self):
    pass

class RectangleReadImageBase(RectangleWithImageBase):
  @abc.abstractproperty
  def imageshape(self): pass
  @abc.abstractproperty
  def imagefile(self): pass
  @abc.abstractproperty
  def imageshapeininput(self): pass
  @abc.abstractproperty
  def imagetransposefrominput(self): pass
  @abc.abstractproperty
  def imageslicefrominput(self): pass

  def getimage(self):
    image = np.ndarray(shape=self.imageshape, dtype=np.uint16)

    with open(self.imagefile, "rb") as f:
      #use fortran order, like matlab!
      with memmapcontext(
        f,
        dtype=np.uint16,
        shape=tuple(self.imageshapeininput),
        order="F",
        mode="r"
      ) as memmap:
        print(self.imagetransposefrominput, self.imageslicefrominput, memmap.shape, memmap.transpose(self.imagetransposefrominput)[self.imageslicefrominput].shape, image.shape)
        image[:] = memmap.transpose(self.imagetransposefrominput)[self.imageslicefrominput]

    return image

class RectangleWithImageMultiLayer(RectangleReadImageBase):
  def __init__(self, *args, imagefolder, filetype, width, height, layers, nlayers, **kwargs):
    super().__init__(*args, **kwargs)
    self.__imagefolder = pathlib.Path(imagefolder)
    self.__filetype = filetype
    self.__width = width
    self.__height = height
    self.__nlayers = nlayers
    self.__layers = layers

  @property
  def imageshape(self):
    return [
      len(self.__layers),
      units.pixels(self.__height, pscale=self.pscale),
      units.pixels(self.__width, pscale=self.pscale),
    ]

  @property
  def imagefile(self):
    if self.__filetype=="flatWarp" :
      ext = ".fw"
    elif self.__filetype=="camWarp" :
      ext = ".camWarp"
    elif self.__filetype=="raw" :
      ext = ".Data.dat"
    else :
      raise ValueError(f"requested file type {self.__filetype} not recognized")

    return self.__imagefolder/self.file.replace(".im3", ext)

  @property
  def imageshapeininput(self):
    return units.pixels((self.__nlayers, self.__width, self.__height), pscale=self.pscale, power=[0, 1, 1])
  @property
  def imagetransposefrominput(self):
    return (0, 2, 1)
  @property
  def imageslicefrominput(self):
    return self.layers, slice(None), slice(None)

class RectangleWithImage(RectangleWithImageMultiLayer):
  def __init__(self, *args, layer, readlayerfile=True, **kwargs):
    morekwargs = {
      "layers": (layer,),
    }
    if readlayerfile:
      morekwargs.update({
        "nlayers": 1,
      })
    super().__init__(*args, **kwargs, **morekwargs)
    self.__readlayerfile = readlayerfile
    self.__layer = layer

  @property
  def layer(self): return self.__layer

  @property
  def imageshape(self):
    return super().imageshape[1:]

  @property
  def imagefile(self):
    result = super().imagefile
    if self.__readlayerfile:
      folder = result.parent
      basename = result.name
      if basename.endswith(".camWarp") or basename.endswith(".dat"):
        basename += f"_layer{self.__layer:02d}"
      elif basename.endswith(".fw"):
        basename += f"{self.__layer:02d}"
      else:
        assert False
      result = folder/basename

    return result

  @property
  def imageshapeininput(self):
    result = super().imageshapeininput
    if self.__readlayerfile:
      assert result[0] == 1
      return result[0], result[2], result[1]
    return result
  @property
  def imagetransposefrominput(self):
    if self.__readlayerfile:
      return (0, 1, 2)
    else:
      return (0, 2, 1)
  @property
  def imageslicefrominput(self):
    return 0, slice(None), slice(None)

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
