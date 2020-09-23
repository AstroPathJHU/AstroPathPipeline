import abc, collections, contextlib, dataclasses, datetime, jxmlease, methodtools, numpy as np, pathlib, warnings
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
  __DEBUG = True

  def __init__(self, *args, transformations=[], **kwargs):
    super().__init__(*args, **kwargs)
    self.__images_cache = [None for _ in range(len(transformations)+1)]
    self.__accessed_image = np.zeros(dtype=bool, shape=len(transformations)+1)
    self.__using_image_counter = np.zeros(dtype=int, shape=len(transformations)+1)
    self.__debug_load_images_counter = np.zeros(dtype=int, shape=len(transformations)+1)
    self.__transformations = transformations

  def __del__(self):
    if self.__DEBUG:
      for i, ctr in enumerate(self.__debug_load_images_counter):
        if ctr > 1:
           warnings.warn(f"Loaded image {i} for rectangle {self} {ctr} times")

  @abc.abstractmethod
  def getimage(self):
    pass

  #do not override any of these functions or call them from super()
  #override getimage() instead and call super().getimage()

  def any_image(self, index):
    #previous_image(-1) gives the actual image
    #previous_image(-2) gives the previous image
    #etc.
    self.__accessed_image[index] = True
    return self.__image(index)

  def delete_any_image(self, index):
    self.__accessed_image[index] = False
    self.__check_delete_image_sequence()

  @property
  def image(self):
    return self.any_image(-1)
  @image.deleter
  def image(self):
    self.delete_any_image(-1)
    self.__check_delete_images()

  @property
  def all_images(self):
    return [self.any_image(i) for i in range(len(self.__images_cache))]
  def delete_all_images(self, index):
    self.__accessed_image[:] = False
    self.__check_delete_image_sequence()

  def __check_delete_images(self):
    for i, (ctr, usingproperty) in enumerate(zip(self.__using_image_counter, self.__accessed_image)):
      if not ctr and not usingproperty:
        self.__images_cache[i] = None

  def __image(self, i):
    if self.__images_cache[i] is None:
      if i < 0: i = (len(self.__transformations)+1) + i
      if i == 0:
        self.__images_cache[i] = self.getimage()
      else:
        with self.using_image(i-1) as previous:
          self.__images_cache[i] = self.__transformations[i-1].transform(previous)
    return self.__images_cache[i]

  @contextlib.contextmanager
  def using_image(self, index=-1):
    self.__using_image_counter[index] += 1
    try:
      yield self.__image(index)
    finally:
      self.__using_image_counter[index] -= 1
      self.__check_delete_images()
  @contextlib.contextmanager
  def using_all_images(self):
    with contextlib.ExitStack() as stack:
      for i in range(len(self.__images_cache)):
        stack.enter_context(self.using_image(i))
      yield stack

class RectangleTransformationBase(abc.ABC):
  @abc.abstractmethod
  def transform(self, previousimage): pass

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
        image[:] = memmap.transpose(self.imagetransposefrominput)[self.imageslicefrominput]

    return image

class RectangleWithImageMultiLayer(RectangleReadImageBase):
  def __init__(self, *args, imagefolder, filetype, width, height, layers, nlayers, xmlfolder=None, **kwargs):
    super().__init__(*args, **kwargs)
    self.__imagefolder = pathlib.Path(imagefolder)
    self.__filetype = filetype
    self.__width = width
    self.__height = height
    self.__nlayers = nlayers
    self.__layers = layers
    self.__xmlfolder = xmlfolder

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
    return tuple(_-1 for _ in self.__layers), slice(None), slice(None)

  @property
  def layers(self):
    return self.__layers

  @property
  def xmlfile(self):
    if self.__xmlfolder is None:
      raise ValueError("Can't get xml info if you don't provide the rectangle with an xml folder")
    return self.__xmlfolder/self.file.replace(".im3", ".SpectralBasisInfo.Exposure.xml")

  @methodtools.lru_cache()
  @property
  def __allexposuretimesandbroadbandfilters(self):
    result = []
    with open(self.xmlfile, "rb") as f:
      broadbandfilter = 0
      for path, _, node in jxmlease.parse(f, generator="/IM3Fragment/D"):
        if node.xml_attrs["name"] == "Exposure":
          thisbroadbandfilter = [float(_) for _ in str(node).split()]
          #sanity check
          assert len(thisbroadbandfilter) == int(node.get_xml_attr("size"))
          broadbandfilter += 1
          for exposuretime in thisbroadbandfilter:
            result.append((exposuretime, broadbandfilter))
    return result

  @property
  def exposuretimes(self):
    all = self.__allexposuretimesandbroadbandfilters
    return [all[layer-1][0] for layer in self.layers]

  @property
  def broadbandfilters(self):
    all = self.__allexposuretimesandbroadbandfilters
    return [all[layer-1][1] for layer in self.layers]

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
    if self.__readlayerfile:
      return 0, slice(None), slice(None)
    else:
      return self.layer-1, slice(None), slice(None)

  @property
  def exposuretime(self):
    _, = self.exposuretimes
    return _
  @property
  def broadbandfilter(self):
    _, = self.broadbandfilters
    return _

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

class RectangleProvideImage(RectangleWithImageBase):
  def __init__(self, *args, image, **kwargs):
    self.__image = image
    super().__init__(*args, **kwargs)
  def getimage(self):
    return self.__image

rectanglefilter = rectangleoroverlapfilter
