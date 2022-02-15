import abc, collections, contextlib, dataclassy, datetime, jxmlease, matplotlib.pyplot as plt, methodtools, numpy as np, pathlib, tifffile, traceback, warnings
from ..utilities import units
from ..utilities.config import CONST as UNIV_CONST
from ..utilities.miscfileio import memmapcontext
from ..utilities.miscmath import floattoint
from ..utilities.tableio import MetaDataAnnotation, pathfield, timestampfield
from ..utilities.units.dataclasses import DataClassWithPscale, distancefield
from .logging import printlogger
from .rectangletransformation import RectangleExposureTimeTransformationMultiLayer, RectangleFlatfieldTransformationMultilayer, RectangleWarpingTransformationMultilayer
from .rectangletransformation import RectangleExposureTimeTransformationSingleLayer, RectangleFlatfieldTransformationSinglelayer, RectangleWarpingTransformationSinglelayer

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
          for field in set(dataclassy.fields(type(rectangle))) & set(dataclassy.fields(cls))
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
    for xml_file_ext in UNIV_CONST.EXPOSURE_XML_EXTS :
      xml_filepath = self.__xmlfolder/self.file.replace(UNIV_CONST.IM3_EXT,xml_file_ext)
      if xml_filepath.is_file() :
        return xml_filepath
    raise FileNotFoundError(f'ERROR: Could not find an xml file for {self.file} with any of the expected file extensions in {self.__xmlfolder}')

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

class RectangleReadIm3MultiLayer(RectangleReadIm3Base):
  """
  Rectangle class that reads the image from a sharded im3
  (could be raw, flatw, etc.)

  im3folder: folder where the im3 image is located
  im3filetype: flatWarp, camWarp, or raw (determines the file extension)
  width, height: the shape of the HPF image
  nlayersim3: the number of layersim3 in the *input* file
  layersim3: which layersim3 you actually want to access
  """

  @property
  def im3folder(self): return self.__im3folder
  @im3folder.setter
  def im3folder(self, im3folder): self.__im3folder = im3folder
  im3folder: pathlib.Path = pathfield(im3folder, includeintable=False, use_default=False)
  @property
  def im3filetype(self): return self.__im3filetype
  @im3filetype.setter
  def im3filetype(self, im3filetype): self.__im3filetype = im3filetype
  im3filetype: str = MetaDataAnnotation(im3filetype, includeintable=False, use_default=False)
  @property
  def width(self): return self.__width
  @width.setter
  def width(self, width): self.__width = width
  width: units.Distance = distancefield(width, includeintable=False, pixelsormicrons="pixels", use_default=False)
  @property
  def height(self): return self.__height
  @height.setter
  def height(self, height): self.__height = height
  height: units.Distance = distancefield(height, includeintable=False, pixelsormicrons="pixels", use_default=False)
  @property
  def nlayersim3(self): return self.__nlayersim3
  @nlayersim3.setter
  def nlayersim3(self, nlayersim3): self.__nlayersim3 = nlayersim3
  nlayersim3: int = MetaDataAnnotation(nlayersim3, includeintable=False, use_default=False)
  @property
  def layersim3(self): return self.__layersim3
  @layersim3.setter
  def layersim3(self, layersim3): self.__layersim3 = layersim3
  layersim3: list = MetaDataAnnotation(layersim3, includeintable=False, use_default=False)

  def __post_init__(self, *args, **kwargs):
    super().__post_init__(*args, **kwargs)
    if self.layersim3 is None: self.layersim3 = range(1, self.nlayersim3+1)
    if -1 in self.layersim3:
      if len(self.layersim3) > 1:
        raise ValueError(f"layersim3 given are {self.layersim3}: if you want to include -1, meaning all layersim3, that should be the only one in the list")
      self.layersim3 = range(1, self.nlayersim3+1)
    self.layersim3 = tuple(self.layersim3)

  @property
  def imageshape(self):
    return [
      floattoint(float(self.height / self.onepixel)),
      floattoint(float(self.width / self.onepixel)),
      len(self.layersim3),
    ]

  @property
  def im3file(self):
    """
    The full file path to the image file
    """
    if self.__im3filetype=="flatWarp" :
      ext = UNIV_CONST.FLATW_EXT
    elif self.__im3filetype=="camWarp" :
      ext = ".camWarp"
    elif self.__im3filetype=="raw" :
      ext = UNIV_CONST.RAW_EXT
    else :
      raise ValueError(f"requested file type {self.__im3filetype} not recognized")

    return self.im3folder/self.file.replace(UNIV_CONST.IM3_EXT, ext)

  @property
  def exposuretimes(self):
    """
    The exposure times for the HPF layersim3 you access
    """
    all = self.allexposuretimes
    return [all[layer-1] for layer in self.__layersim3]

  @property
  def broadbandfilters(self):
    """
    The broadband filter ids (numbered from 1) of the layersim3 you access
    """
    all = self.allbroadbandfilters
    return [all[layer-1] for layer in self.__layersim3]

class RectangleReadIm3(RectangleReadIm3MultiLayer):
  """
  Single layer image read from a sharded im3.
  You can also use RectangleReadIm3MultiLayer and write layersim3=[i],
  but this class gives you a 2D array as the image instead of a 3D array
  with shape[0] = 1.
  Also, in this class you can read a layer file (e.g. fw01).
  """

  @property
  def readlayerfile(self): return self.__readlayerfile
  @readlayerfile.setter
  def readlayerfile(self, readlayerfile): self.__readlayerfile = readlayerfile
  readlayerfile: bool = MetaDataAnnotation(True, includeintable=False, use_default=False)

  @classmethod
  def transforminitargs(cls, *args, layer, readlayerfile=True, **kwargs):
    morekwargs = {
      "layersim3": (layer,),
      "readlayerfile": readlayerfile,
    }
    if readlayerfile and "nlayersim3" not in kwargs:
      morekwargs["nlayersim3"] = 1
    return super().transforminitargs(*args, **kwargs, **morekwargs)

  def __post_init__(self, *args, **kwargs):
    if self.nlayersim3 != 1 and self.readlayerfile:
      raise ValueError("Provided nlayersim3!=1, readlayerfile=True")
    super().__post_init__(*args, **kwargs)

  @property
  def layerim3(self):
    layerim3, = self.layersim3
    return layerim3

  @property
  def im3file(self):
    result = super().im3file
    if self.readlayerfile:
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

class RectangleCorrectedIm3SingleLayer(RectangleReadIm3MultiLayer) :
  """
  Class for Rectangles whose multilayer im3 data should have one layer extracted and corrected
  for differences in exposure time, flatfielding effects, and warping effects (and or all can be omitted)
  """
  _DEBUG = False #tend to load these more than once

  def __post_init__(self,*args,transformations=None,**kwargs) :
    if transformations is None :
      transformations = []
    super().__post_init__(*args, transformations=transformations, **kwargs)

  def add_exposure_time_correction_transformation(self,med_et,offset) :
    """
    Add a transformation to a rectangle to correct it for differences in exposure time given:

    med_et = the median exposure time of the image layer in question across the whole sample
    offset = the dark current offset to use for correcting the image layer of interest
    """
    if med_et is not None and offset is not None :
      self.add_transformation(RectangleExposureTimeTransformationSingleLayer(self.allexposuretimes[self.layers[0]-1],
                                                                             med_et,offset))

  def add_flatfield_correction_transformation(self,flatfield_layer) :
    """
    Add a transformation to a rectangle to correct it with a given flatfield layer
    """
    if flatfield_layer is not None :
      self.add_transformation(RectangleFlatfieldTransformationSinglelayer(flatfield_layer))

  def add_warping_correction_transformation(self,warp) :
    """
    Add a transformation to a rectangle to correct its image layer with a given warping pattern
    """
    if warp is not None :
      self.add_transformation(RectangleWarpingTransformationSinglelayer(warp))

class RectangleCorrectedIm3MultiLayer(RectangleReadIm3MultiLayer):
  """
  Class for Rectangles whose multilayer im3 data should be corrected for differences in exposure time,
  flatfielding effects, and/or warping effects (any or all can be omitted)
  """
  _DEBUG = False #Tend to use these images more than once per run
  
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

  def add_warping_correction_transformation(self,warps_by_layer) :
    """
    Add a transformation to a rectangle to correct it with given warping patterns

    warps_by_layer = a list of the warping objects to use in each image layer
    """
    self.add_transformation(RectangleWarpingTransformationMultilayer(warps_by_layer))


class RectangleReadComponentTiffMultiLayer(Rectangle):
  """
  Rectangle class that reads the image from a component tiff

  componenttifffolder: folder where the component tiff is located
  nlayers: the number of layers in the *input* file (optional, just used as a sanity check)
  layers: which layers you actually want to access
  with_seg: indicates if you want to use the _w_seg.tif which contains some segmentation info from inform
  """

  @property
  def componenttifffolder(self): return self.__componenttifffolder
  @componenttifffolder.setter
  def componenttifffolder(self, componenttifffolder): self.__componenttifffolder = componenttifffolder
  componenttifffolder: pathlib.Path = pathfield(componenttifffolder, includeintable=False, use_default=False)
  @property
  def nlayerscomponenttiff(self): return self.__nlayerscomponenttiff
  @nlayerscomponenttiff.setter
  def nlayerscomponenttiff(self, nlayerscomponenttiff): self.__nlayerscomponenttiff = nlayerscomponenttiff
  nlayerscomponenttiff: int = MetaDataAnnotation(nlayerscomponenttiff, includeintable=False, use_default=False)
  @property
  def layerscomponenttiff(self): return self.__layerscomponenttiff
  @layerscomponenttiff.setter
  def layerscomponenttiff(self, layerscomponenttiff): self.__layerscomponenttiff = layerscomponenttiff
  layerscomponenttiff: list = MetaDataAnnotation(layerscomponenttiff, includeintable=False, use_default=False)
  @property
  def with_seg(self): return self.__with_seg
  @with_seg.setter
  def with_seg(self, with_seg): self.__with_seg = with_seg
  with_seg: bool = MetaDataAnnotation(with_seg, includeintable=False, use_default=False)
  @property
  def nsegmentations(self): return self.__nsegmentations
  @nsegmentations.setter
  def nsegmentations(self, nsegmentations): self.__nsegmentations = nsegmentations
  nsegmentations: int = MetaDataAnnotation(nsegmentations, includeintable=False, use_default=False)

  def __post_init__(self, *args, componenttifffolder, layers, nlayers=None, with_seg=False, nsegmentations=None, **kwargs):
    super().__post_init__(*args, **kwargs)
    if self.with_seg and self.nsegmentations is None:
      raise ValueError("To use segmented component tiffs, you have to provide nsegmentations")

  @property
  def componenttifffile(self):
    return self.__componenttifffolder/self.file.replace(UNIV_CONST.IM3_EXT, f"_component_data{'_w_seg' if self.__with_seg else ''}.tif")

  @property
  def layerscomponenttiff(self):
    return self.__layerscomponenttiff

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

class RectangleCollection(units.ThingWithPscale):
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

  @methodtools.lru_cache()
  @property
  def hpfoffset(self):
    """
    The distance in x and y between adjacent HPFs
    """
    rectxvecs = np.round(np.array([rect.xvec for rect in self.rectangles])/self.onepixel, 5) * self.onepixel

    result = np.zeros(2, dtype=units.unitdtype)
    for idx in 0, 1:
      otheridx = 1-idx

      c = collections.Counter()

      for xvec1 in rectxvecs:
        mindiff = None
        for xvec2 in rectxvecs[rectxvecs[:, otheridx] == xvec1[otheridx]]:
          thisdiff = (xvec2 - xvec1)[idx]
          if thisdiff > 0 and (mindiff is None or thisdiff < mindiff):
            mindiff = thisdiff
        if mindiff is not None:
          c[mindiff] += 1

      mostcommon = c.most_common()
      result[idx], mostndiffs = mostcommon[0]
      for diff, ndiffs in mostcommon:
        if ndiffs == mostndiffs:
          if diff < result[idx]:
            result[idx] = diff
        else:
          break

    return result

class RectangleList(list, RectangleCollection):
  """
  A list of rectangles.  You can get rectangledict and rectangleindices from it.
  """
  @property
  def rectangles(self): return self

  @methodtools.lru_cache()
  @property
  def pscale(self):
    result, = {rect.pscale for rect in self.rectangles}
    return result

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
    return self.__geomfolder/self.file.replace(UNIV_CONST.IM3_EXT, "_cellGeomLoad.csv")

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
    return self.__maskfolder/self.file.replace(UNIV_CONST.IM3_EXT, "_tissue_mask.bin")
  @property
  def fullmaskfile(self):
    return self.__maskfolder/self.file.replace(UNIV_CONST.IM3_EXT, "_full_mask.bin")

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
    return self.__phenotypetablesfolder/self.file.replace(UNIV_CONST.IM3_EXT, "_cleaned_phenotype_table.csv")
  @property
  def __phenotypeQAQCtablesfolder(self):
    return self.__phenotypefolder/"Results"/"QA_QC"/"Tables_QA_QC"
  @property
  def phenotypeQAQCcsv(self):
    return self.__phenotypeQAQCtablesfolder/self.file.replace(UNIV_CONST.IM3_EXT, "_cleaned_phenotype_table.csv")

rectanglefilter = rectangleoroverlapfilter
