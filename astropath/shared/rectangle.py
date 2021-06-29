import abc, collections, contextlib, dataclassy, datetime, jxmlease, matplotlib.pyplot as plt, methodtools, numpy as np, pathlib, tifffile, traceback, warnings
from ..utilities import units
from ..utilities.misc import floattoint, memmapcontext
from ..utilities.tableio import timestampfield
from ..utilities.units.dataclasses import DataClassWithPscale, distancefield
from ..utilities.config import CONST as UNIV_CONST
from .rectangletransformation import RectangleExposureTimeTransformationMultiLayer, RectangleFlatfieldTransformationMultilayer

class Rectangle(DataClassWithPscale):
  """
  Base class for all HPFs
  n, x, y, w, h, cx, cy, t, and file are columns in SlideID_rect.csv

  n: the id of the HPF, starting from 1
  x, y: the coordinates of the top left corner
  w, h: the size of the field
  cx, cy: the coordinates of the center, in integer microns
  t: what time the HPF image was recorded
  file: the filename of the im3 for the HPF

  One of the following two arguments is required if you want to get the exposure times:
    xmlfolder: the folder where the xml file with metadata for the HPF lives
    allexposures: the list of exposures from the exposures csv
  """

  n: int
  x: units.Distance = distancefield(pixelsormicrons="microns")
  y: units.Distance = distancefield(pixelsormicrons="microns")
  w: units.Distance = distancefield(pixelsormicrons="microns", dtype=int)
  h: units.Distance = distancefield(pixelsormicrons="microns", dtype=int)
  cx: units.Distance = distancefield(pixelsormicrons="microns", dtype=int)
  cy: units.Distance = distancefield(pixelsormicrons="microns", dtype=int)
  t: datetime.datetime = timestampfield()
  file: str

  def __post_init__(self, *args, xmlfolder=None, allexposures=None, **kwargs):
    self.__xmlfolder = xmlfolder
    self.__allexposures = allexposures
    super().__post_init__(*args, **kwargs)

  @classmethod
  def transforminitargs(cls, *args, rectangle=None, **kwargs):
    """
    If you give an existing Rectangle to init, the current rectangle
    will be identical to that one.  This is useful for subclasses.
    """
    rectanglekwargs = {}
    if rectangle is not None:
      rectanglekwargs = {
        **{
          field: getattr(rectangle, field)
          for field in dataclassy.fields(type(rectangle))
        }
      }
    return super().transforminitargs(
      *args,
      **rectanglekwargs,
      **kwargs,
    )

  @property
  def xvec(self):
    """
    Gives [x, y] as a numpy array
    """
    return np.array([self.x, self.y])

  @property
  def cxvec(self):
    """
    Gives [cx, cy] as a numpy array
    """
    return np.array([self.cx, self.cy])

  @property
  def shape(self):
    """
    Gives [w, h] as a numpy array
    """
    return np.array([self.w, self.h])

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

  @methodtools.lru_cache()
  @property
  def allexposuretimes(self):
    """
    The exposure times for the HPF layers
    """
    exposures1 = exposures2 = None
    if self.__allexposures is not None:
      exposures = [_ for _ in self.__allexposures if _.n == self.n]
      exposures1 = [e.exp for e in exposures]
    if self.__xmlfolder is not None:
      exposures2 = [exposuretimeandbroadbandfilter[0] for exposuretimeandbroadbandfilter in self.__allexposuretimesandbroadbandfilters]
    if exposures1 is exposures2 is None:
      raise ValueError("Can't get the exposure times unless you provide the xml folder or exposures csv")
    if None is not exposures1 != exposures2 is not None:
      raise ValueError(f"Found inconsistent exposure times from exposures.csv and from xml file:\n{exposures1}\n{exposures2}")
    return exposures1 if exposures1 is not None else exposures2

  @methodtools.lru_cache()
  @property
  def allbroadbandfilters(self):
    """
    The broadband filter ids (numbered from 1) of the layers
    """
    return [exposuretimeandbroadbandfilter[1] for exposuretimeandbroadbandfilter in self.__allexposuretimesandbroadbandfilters]

class RectangleWithImageBase(Rectangle):
  """
  Base class for any kind of rectangle that has a way of getting an image.
  To get the image and keep it indefinitely, you can access `rectangle.image`.
  However, you might also want to do instead
  ```
  with rectangle.using_image() as im:
    ... #do stuff with im
  ```
  This frees up the memory of im after the with block is finished.

  Subclasses have to implement `getimage`, which loads the image from
  an im3 or component tiff or somewhere else.

  A rectangle can also have transformations, which are applied to the
  raw image to make the final image returned by `using_image()` or `image`.
  They should inherit from RectangleTransformationBase.
  """

  #if _DEBUG is true, then when the rectangle is deleted, it will print
  #a warning if its image has been loaded multiple times, for debug
  #purposes.  If __DEBUG_PRINT_TRACEBACK is also true, it will print the
  #tracebacks for each of the times the image was loaded.
  _DEBUG = True
  def __DEBUG_PRINT_TRACEBACK(self, i):
    return False

  def __post_init__(self, *args, transformations=[], **kwargs):
    self.__transformations = transformations
    self.__images_cache = [None for _ in range(self.nimages)]
    self.__accessed_image = np.zeros(dtype=bool, shape=self.nimages)
    self.__using_image_counter = np.zeros(dtype=int, shape=self.nimages)
    self.__debug_load_images_counter = np.zeros(dtype=int, shape=self.nimages)
    self.__debug_load_images_tracebacks = [[] for _ in range(self.nimages)]
    super().__post_init__(*args, **kwargs)

  def __del__(self):
    if self._DEBUG:
      for i, ctr in enumerate(self.__debug_load_images_counter):
        if ctr > 1:
          for formattedtb in self.__debug_load_images_tracebacks[i]:
            print("".join(formattedtb))
          warnings.warn(f"Loaded image {i} for rectangle {self} {ctr} times")

  def add_transformation(self,new_transformation) :
    """
    Add a new transformation to this Rectangle post-initialization
    Helpful in case a Rectangle needs to be transformed after calculating something from the whole set of Rectangles
    
    new_transformation: the new transformation to add (should inherit from RectangleTransformationBase)
    """
    old_nimages = self.nimages
    self.__transformations.append(new_transformation)
    self.__images_cache.append(None)
    self.__accessed_image = np.append(self.__accessed_image,0)
    self.__using_image_counter = np.append(self.__using_image_counter,0)
    self.__debug_load_images_counter = np.append(self.__debug_load_images_counter,0)
    self.__debug_load_images_tracebacks.append([])

  @abc.abstractmethod
  def getimage(self):
    """
    Override this function in subclasses that actually implement
    a way of loading the image
    """

  @property
  def nimages(self):
    return len(self.__transformations)+1

  #do not override any of these functions or call them from super()
  #override getimage() instead and call super().getimage()

  def any_image(self, index):
    """
    any_image(-1) gives the actual image
    any_image(-2) gives the previous image, immediately before the last transformation
    The image gets saved indefinitely until you call delete_any_image
    etc.
    """
    self.__accessed_image[index] = True
    return self.__image(index)

  def delete_any_image(self, index):
    """
    Call this to free the memory from an image accessed by any_image
    """
    self.__accessed_image[index] = False
    self.__check_delete_images()

  @property
  def image(self):
    """
    Gives the HPF image.
    It gets saved in memory until you call `del rectangle.image`
    """
    return self.any_image(-1)
  @image.deleter
  def image(self):
    self.delete_any_image(-1)
    self.__check_delete_images()

  @property
  def all_images(self):
    return [self.any_image(i) for i in range(len(self.__images_cache))]
  def delete_all_images(self, index):
    """
    Free up the memory from all previously accessed images
    """
    self.__accessed_image[:] = False
    self.__check_delete_images()

  def __check_delete_images(self):
    """
    This gets called whenever you delete an image or leave a using_image context.
    It deletes images that are no longer needed in memory.
    """
    for i, (ctr, usingproperty) in enumerate(zip(self.__using_image_counter, self.__accessed_image)):
      if not ctr and not usingproperty:
        self.__images_cache[i] = None

  def __image(self, i):
    if self.__images_cache[i] is None:
      self.__debug_load_images_counter[i] += 1
      if self.__DEBUG_PRINT_TRACEBACK(i):
        self.__debug_load_images_tracebacks[i].append(traceback.format_stack())
      if i < 0: i = self.nimages + i
      if i == 0:
        self.__images_cache[i] = self.getimage()
      else:
        with self.using_image(i-1) as previous:
          self.__images_cache[i] = self.__transformations[i-1].transform(previous)
    return self.__images_cache[i]

  @contextlib.contextmanager
  def using_image(self, index=-1):
    """
    Use this in a with statement to load the image for the HPF.
    It gets freed from memory when the with statement ends.
    """
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

  @property
  def _imshowextent(self):
    return self.x, self.x+self.w, self.y+self.h, self.y
  def imshow(self, *, imagescale=None, xlim=(), ylim=()):
    """
    Convenience function to show the image.
    """
    if imagescale is None: imagescale = self.pscale
    extent = units.convertpscale(self._imshowextent, self.pscale, imagescale)
    xlim = (np.array(xlim) / units.onepixel(imagescale)).astype(float)
    ylim = (np.array(ylim) / units.onepixel(imagescale)).astype(float)
    with self.using_image() as im:
      plt.imshow(im, extent=(extent / units.onepixel(imagescale)).astype(float))
      plt.xlim(*xlim)
      plt.ylim(*ylim)

class RectangleReadIm3MultiLayer(RectangleWithImageBase):
  """
  Rectangle class that reads the image from a sharded im3
  (could be raw, flatw, etc.)

  imagefolder: folder where the im3 image is located
  filetype: flatWarp, camWarp, or raw (determines the file extension)
  width, height: the shape of the HPF image
  nlayers: the number of layers in the *input* file
  layers: which layers you actually want to access
  """

  def __post_init__(self, *args, imagefolder, filetype, width, height, nlayers, layers, **kwargs):
    super().__post_init__(*args, **kwargs)
    self.__imagefolder = pathlib.Path(imagefolder)
    self.__filetype = filetype
    self.__width = width
    self.__height = height
    self.__nlayers = nlayers
    self.__layers = layers

  @property
  def imageshape(self):
    return [
      floattoint(float(self.__height / self.onepixel)),
      floattoint(float(self.__width / self.onepixel)),
      len(self.__layers),
    ]

  @property
  def imagefile(self):
    """
    The full file path to the image file
    """
    if self.__filetype=="flatWarp" :
      ext = UNIV_CONST.FLATW_EXT
    elif self.__filetype=="camWarp" :
      ext = ".camWarp"
    elif self.__filetype=="raw" :
      ext = UNIV_CONST.RAW_EXT
    else :
      raise ValueError(f"requested file type {self.__filetype} not recognized")

    return self.__imagefolder/self.file.replace(".im3", ext)

  @property
  def imageshapeininput(self):
    return self.__nlayers, floattoint(float(self.__width / self.onepixel)), floattoint(float(self.__height / self.onepixel))
  @property
  def imagetransposefrominput(self):
    #it's saved as (layers, width, height), we want (height, width, layers)
    return (2, 1, 0)
  @property
  def imageslicefrominput(self):
    return slice(None), slice(None), tuple(_-1 for _ in self.__layers)
  @property
  def imageshapeinoutput(self):
    return (np.empty((self.imageshapeininput)).transpose(self.imagetransposefrominput)[self.imageslicefrominput]).shape

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

  @property
  def layers(self) :
    return self.__layers

  @property
  def exposuretimes(self):
    """
    The exposure times for the HPF layers you access
    """
    all = self.allexposuretimes
    return [all[layer-1] for layer in self.__layers]

  @property
  def broadbandfilters(self):
    """
    The broadband filter ids (numbered from 1) of the layers you access
    """
    all = self.allbroadbandfilters
    return [all[layer-1] for layer in self.__layers]

class RectangleReadIm3(RectangleReadIm3MultiLayer):
  """
  Single layer image read from a sharded im3.
  You can also use RectangleReadIm3MultiLayer and write layers=[i],
  but this class gives you a 2D array as the image instead of a 3D array
  with shape[0] = 1.
  Also, in this class you can read a layer file (e.g. fw01).
  """

  def __post_init__(self, *args, layer, readlayerfile=True, **kwargs):
    morekwargs = {
      "layers": (layer,),
    }
    if readlayerfile:
      if kwargs.pop("nlayers", 1) != 1:
        raise ValueError("Provided nlayers!=1, readlayerfile=True")
      morekwargs.update({
        "nlayers": 1,
      })
    self.__readlayerfile = readlayerfile
    super().__post_init__(*args, **kwargs, **morekwargs)

  @property
  def layer(self):
    layer, = self.layers
    return layer

  @property
  def imageshape(self):
    return super().imageshape[:-1]

  @property
  def imagefile(self):
    result = super().imagefile
    if self.__readlayerfile:
      folder = result.parent
      basename = result.name
      if basename.endswith(".camWarp") or basename.endswith(".dat"):
        basename += f"_layer{self.layer:02d}"
      elif basename.endswith(UNIV_CONST.FLATW_EXT):
        basename += f"{self.layer:02d}"
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
      #it's saved as (height, width), which is what we want
      return (0, 1, 2)
    else:
      #it's saved as (layers, width, height), we want (height, width, layers)
      return (2, 1, 0)
  @property
  def imageslicefrominput(self):
    if self.__readlayerfile:
      return 0, slice(None), slice(None)
    else:
      return slice(None), slice(None), self.layer-1

  @property
  def exposuretime(self):
    """
    The exposure time for this layer
    """
    _, = self.exposuretimes
    return _
  @property
  def broadbandfilter(self):
    """
    The broadband filter id for this layer
    """
    _, = self.broadbandfilters
    return _

class RectangleCorrectedIm3MultiLayer(RectangleReadIm3MultiLayer):
  """
  Class for Rectangles whose multilayer im3 data should be corrected for differences in exposure time 
  and/or flatfielding (either or both can be omitted)
  """
  
  def __post_init__(self, *args, transformations=None, **kwargs) :
    if transformations is None : 
      transformations = []
    super().__post_init__(*args, transformations=transformations, **kwargs)

  def add_exposure_time_correction_transformation(self,med_ets,offsets) :
    """
    Add a transformation to a rectangle to correct it for differences in exposure time given:

    med_ets = the median exposure times in the rectangle's slide 
    offsets = the list of dark current offsets for the rectangle's slide
    """
    if (med_ets is not None) and (offsets is not None) :
      self.add_transformation(RectangleExposureTimeTransformationMultiLayer(self.allexposuretimes,med_ets,offsets))

  def add_flatfield_correction_transformation(self,flatfield) :
    """
    Add a transformation to a rectangle to correct it with a given flatfield

    flatfield = the flatfield correction factor image to apply
    """
    if flatfield is not None:
      self.add_transformation(RectangleFlatfieldTransformationMultilayer(flatfield))

class RectangleReadComponentTiffMultiLayer(RectangleWithImageBase):
  """
  Rectangle class that reads the image from a component tiff

  imagefolder: folder where the component tiff is located
  nlayers: the number of layers in the *input* file (optional, just used as a sanity check)
  layers: which layers you actually want to access
  with_seg: indicates if you want to use the _w_seg.tif which contains some segmentation info from inform
  """

  def __post_init__(self, *args, imagefolder, layers, nlayers=None, with_seg=False, nsegmentations=None, **kwargs):
    super().__post_init__(*args, **kwargs)
    self.__imagefolder = pathlib.Path(imagefolder)
    self.__layers = layers
    self.__nlayers = nlayers
    self.__with_seg = with_seg
    self.__nsegmentations = nsegmentations
    if with_seg and nsegmentations is None:
      raise ValueError("To use segmented component tiffs, you have to provide nsegmentations")

  @property
  def imagefile(self):
    return self.__imagefolder/self.file.replace(".im3", f"_component_data{'_w_seg' if self.__with_seg else ''}.tif")

  @property
  def layers(self):
    return self.__layers

  def getimage(self):
    with tifffile.TiffFile(self.imagefile) as f:
      pages = []
      shape = None
      dtype = None
      segmentationisblank = False
      #make sure the tiff is self consistent in shape and dtype
      for page in f.pages:
        if len(page.shape) == 2:
          pages.append(page)
          if shape is None:
            shape = page.shape
          elif shape != page.shape:
            raise ValueError(f"Found pages with different shapes in the component tiff {shape} {page.shape}")
          if dtype is None:
            dtype = page.dtype
          elif dtype != page.dtype:
            raise ValueError(f"Found pages with different dtypes in the component tiff {dtype} {page.dtype}")
      expectpages = self.__nlayers
      if expectpages is not None:
        if self.__with_seg: expectpages += 1 + 2*self.__nsegmentations
        if len(pages) != expectpages:
          #compatibility with inform errors, the segmentation is all blank and sometimes the wrong number of layers
          if self.__with_seg and len(pages) > self.__nlayers:
            if all(not np.any(page.asarray()) for page in pages[self.__nlayers:]):
              segmentationisblank = True
          if not segmentationisblank:
            raise IOError(f"Wrong number of pages {len(pages)} in the component tiff, expected {expectpages}")

      #make the destination array
      image = np.ndarray(shape=shape+(len(self.__layers),), dtype=dtype)

      #load the desired layers
      for i, layer in enumerate(self.__layers):
        image[:,:,i] = pages[layer-1].asarray()

      return image

class RectangleReadComponentTiff(RectangleReadComponentTiffMultiLayer):
  """
  Single layer image read from a component tiff.
  You can also use RectangleReadIm3MultiLayer and write layers=[i],
  but this class gives you a 2D array as the image instead of a 3D array
  with shape[2] = 1.
  """
  def __post_init__(self, *args, layer, **kwargs):
    morekwargs = {
      "layers": (layer,),
    }
    super().__post_init__(*args, **kwargs, **morekwargs)

  @property
  def layer(self):
    layer, = self.layers
    return layer

  def getimage(self):
    image, = super().getimage().transpose(2, 0, 1)
    return image

class RectangleCollection(abc.ABC):
  """
  Base class for a group of rectangles.
  You can get a rectangledict from it, which allows indexing rectangles
  by their id.
  """
  @property
  @abc.abstractmethod
  def rectangles(self): pass
  @methodtools.lru_cache()
  @property
  def rectangledict(self):
    """
    Make a dict that allows accessing the rectangles in the collection
    by their index
    """
    return rectangledict(self.rectangles)
  @property
  def rectangleindices(self):
    """
    Indices of all rectangles in the collection.
    """
    return {r.n for r in self.rectangles}

class RectangleList(list, RectangleCollection):
  """
  A list of rectangles.  You can get rectangledict and rectangleindices from it.
  """
  @property
  def rectangles(self): return self

def rectangledict(rectangles):
  """
  Make a dict that allows accessing the rectangles by their index
  """
  return {rectangle.n: i for i, rectangle in enumerate(rectangles)}

def rectangleoroverlapfilter(selection, *, compatibility=False):
  """
  Makes a filter that can be called to determine whether or not a rectangle
  or overlap is selected.
  selection can be:
    - None - selects all rectangles or overlaps
    - list, tuple, set, etc. of numbers - returns rectangles or overlaps with those ids
    - a function: just calls that function to determine whether it's selected
  """
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
  """
  Rectangle where you just input an image and that will be the image returned by image or using_image.
  """
  def __post_init__(self, *args, image, **kwargs):
    self.__image = image
    super().__post_init__(*args, **kwargs)
  def getimage(self):
    return self.__image

class RectangleFromOtherRectangle(RectangleWithImageBase):
  """
  Rectangle where the image comes from another rectangle.
  The reason you'd want to do this is if you have transformations, but
  also want the original rectangle to keep its images.
  """
  def __post_init__(self, *args, originalrectangle, **kwargs):
    self.__originalrectangle = originalrectangle
    super().__post_init__(*args, rectangle=originalrectangle, readingfromfile=False, **kwargs)
  @property
  def originalrectangle(self):
    return self.__originalrectangle
  def getimage(self):
    with self.__originalrectangle.using_image() as image:
      return image

class GeomLoadRectangle(Rectangle):
  """
  Rectangle that has a cellGeomLoad.csv
  You have to provide the folder where that csv lives.
  """
  def __post_init__(self, *args, geomfolder, **kwargs):
    self.__geomfolder = pathlib.Path(geomfolder)
    super().__post_init__(*args, **kwargs)
  @property
  def geomloadcsv(self):
    return self.__geomfolder/self.file.replace(".im3", "_cellGeomLoad.csv")

class MaskRectangle(Rectangle):
  """
  Rectangle that has mask files, e.g. _tissue_mask.bin
  You have to provide the folder where those files live.
  """
  def __post_init__(self, *args, maskfolder, **kwargs):
    self.__maskfolder = pathlib.Path(maskfolder)
    super().__post_init__(*args, **kwargs)
  @property
  def tissuemaskfile(self):
    return self.__maskfolder/self.file.replace(".im3", "_tissue_mask.bin")
  @property
  def fullmaskfile(self):
    return self.__maskfolder/self.file.replace(".im3", "_full_mask.bin")

class PhenotypedRectangle(Rectangle):
  """
  Rectangle that has a _cleaned_phenotype_table.csv
  You have to provide the folder where that csv lives.
  """
  def __post_init__(self, *args, phenotypefolder, **kwargs):
    self.__phenotypefolder = pathlib.Path(phenotypefolder)
    super().__post_init__(*args, **kwargs)
  @property
  def __phenotypetablesfolder(self):
    return self.__phenotypefolder/"Results"/"Tables"
  @property
  def phenotypecsv(self):
    return self.__phenotypetablesfolder/self.file.replace(".im3", "_cleaned_phenotype_table.csv")
  @property
  def __phenotypeQAQCtablesfolder(self):
    return self.__phenotypefolder/"Results"/"QA_QC"/"Tables_QA_QC"
  @property
  def phenotypeQAQCcsv(self):
    return self.__phenotypeQAQCtablesfolder/self.file.replace(".im3", "_cleaned_phenotype_table.csv")

rectanglefilter = rectangleoroverlapfilter
