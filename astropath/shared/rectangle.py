import abc, collections, dataclassy, datetime, jxmlease, methodtools, numpy as np, pathlib
from ..utilities import units
from ..utilities.config import CONST as UNIV_CONST
from ..utilities.miscfileio import with_stem
from ..utilities.miscmath import floattoint
from ..utilities.tableio import MetaDataAnnotation, pathfield, timestampfield
from ..utilities.units.dataclasses import DataClassWithPscale, distancefield
from .image_masking.maskloader import ThingWithMask, ThingWithTissueMask
from .imageloader import ImageLoaderBin, ImageLoaderComponentTiffMultiLayer, ImageLoaderComponentTiffSingleLayer, ImageLoaderIm3MultiLayer, ImageLoaderIm3SingleLayer, ImageLoaderHasSingleLayerTiff, ImageLoaderNpz, ImageLoaderSegmentedComponentTiffMultiLayer, ImageLoaderSegmentedComponentTiffSingleLayer, TransformedImage
from .rectangletransformation import AsTypeTransformation, RectangleExposureTimeTransformationMultiLayer, RectangleExposureTimeTransformationSingleLayer, RectangleFlatfieldTransformationMultilayer, RectangleFlatfieldTransformationSinglelayer, RectangleWarpingTransformationMultilayer, RectangleWarpingTransformationSinglelayer

class RectangleBase(DataClassWithPscale):
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
  file: pathlib.PurePath = pathfield()
  SlideID: str = MetaDataAnnotation(None, includeintable=False)

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
      xml_filepath = self.__xmlfolder/self.file.name.replace(UNIV_CONST.IM3_EXT,xml_file_ext)
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
    return np.array(exposures1 if exposures1 is not None else exposures2)

  @methodtools.lru_cache()
  @property
  def allbroadbandfilters(self):
    """
    The broadband filter ids (numbered from 1) of the layers
    """
    return [exposuretimeandbroadbandfilter[1] for exposuretimeandbroadbandfilter in self.__allexposuretimesandbroadbandfilters]

  @property
  @abc.abstractmethod
  def expectedfilename(self):
    pass

class Rectangle(RectangleBase):
  @property
  def expectedfilename(self):
    if self.SlideID is None:
      raise TypeError("Have to give SlideID to the Rectangle constructor if you want to get the expected filename")
    return type(self.file)(f"{self.SlideID}_[{floattoint(float(self.cx/self.onemicron)):d},{floattoint(float(self.cy/self.onemicron)):d}]{UNIV_CONST.IM3_EXT}")

class TMARectangle(RectangleBase):
  TMAsector: int
  TMAname1: int
  TMAname2: int
  @property
  def expectedfilename(self):
    if self.SlideID is None:
      raise TypeError("Have to give SlideID to the Rectangle constructor if you want to get the expected filename")
    return type(self.file)(f"{self.SlideID}_Core[{self.TMAsector},{self.TMAname1},{self.TMAname2}]_[{floattoint(float(self.cx/self.onemicron)):d},{floattoint(float(self.cy/self.onemicron)):d}]{UNIV_CONST.IM3_EXT}")

class RectangleWithImageLoaderBase(Rectangle):
  def __post_init__(self, *args, _DEBUG=True, _DEBUG_PRINT_TRACEBACK=False, **kwargs):
    super().__post_init__(*args, **kwargs)
    self._DEBUG = _DEBUG
    self._DEBUG_PRINT_TRACEBACK = _DEBUG_PRINT_TRACEBACK

class RectangleWithImageSize(Rectangle):
  """
  width, height: the shape of the HPF image (!= w, h, which are rounded)
  """
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
  def imageshape(self):
    return np.array([self.width, self.height])

class RectangleReadIm3Base(RectangleWithImageLoaderBase, RectangleWithImageSize):
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
  def nlayersim3(self): return self.__nlayersim3
  @nlayersim3.setter
  def nlayersim3(self, nlayersim3): self.__nlayersim3 = nlayersim3
  nlayersim3: int = MetaDataAnnotation(nlayersim3, includeintable=False, use_default=False)
  @property
  @abc.abstractmethod
  def layersim3(self): pass
  @property
  def usememmap(self): return self.__usememmap
  @usememmap.setter
  def usememmap(self, usememmap): self.__usememmap = usememmap
  usememmap: bool = MetaDataAnnotation(usememmap, includeintable=False, use_default=False)

  @classmethod
  def transforminitargs(cls, *args, usememmap=False, **kwargs):
    return super().transforminitargs(*args, usememmap=usememmap, **kwargs)

  @property
  def im3shape(self):
    return (
      floattoint(float(self.height / self.onepixel)),
      floattoint(float(self.width / self.onepixel)),
      len(self.layersim3),
    )

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

    return self.im3folder/self.file.name.replace(UNIV_CONST.IM3_EXT, ext)

  @property
  def exposuretimes(self):
    """
    The exposure times for the HPF layersim3 you access
    """
    all = self.allexposuretimes
    return [all[layer-1] for layer in self.layersim3]

  @property
  def broadbandfilters(self):
    """
    The broadband filter ids (numbered from 1) of the layersim3 you access
    """
    all = self.allbroadbandfilters
    return [all[layer-1] for layer in self.layersim3]

  @methodtools.lru_cache()
  @property
  def im3loader(self):
    return self.im3loadertype(**self.im3loaderkwargs)
  def using_im3(self):
    return self.im3loader.using_image()
  @property
  @abc.abstractmethod
  def im3loadertype(self): pass
  @property
  @abc.abstractmethod
  def im3loaderkwargs(self): return {
    "filename": self.im3file,
    "width": floattoint(float(self.width / self.onepixel)),
    "height": floattoint(float(self.height / self.onepixel)),
    "usememmap": self.usememmap,
    "_DEBUG": self._DEBUG,
    "_DEBUG_PRINT_TRACEBACK": self._DEBUG_PRINT_TRACEBACK,
  }

class RectangleReadIm3MultiLayer(RectangleReadIm3Base):
  @property
  def layersim3(self): return self.__layersim3
  @layersim3.setter
  def layersim3(self, layersim3): self.__layersim3 = layersim3
  layersim3: list = MetaDataAnnotation(layersim3, includeintable=False, use_default=False)

  def __post_init__(self, *args, **kwargs):
    super().__post_init__(*args, **kwargs)
    if self.layersim3 is None: self.layersim3 = range(1, self.nlayersim3+1)
    if -1 in self.layersim3:
      self.layersim3 = range(1, self.nlayersim3+1)
    self.layersim3 = tuple(self.layersim3)

  @property
  def im3loadertype(self):
    return ImageLoaderIm3MultiLayer
  @property
  def im3loaderkwargs(self): return {
    **super().im3loaderkwargs,
    "nlayers": self.nlayersim3,
    "selectlayers": self.layersim3,
  }

class RectangleReadIm3SingleLayer(RectangleReadIm3Base):
  """
  Single layer image read from a sharded im3.
  You can also use RectangleReadIm3MultiLayer and write layersim3=[i],
  but this class gives you a 2D array as the image instead of a 3D array
  with shape[0] = 1.
  Also, in this class you can read a layer file (e.g. fw01).
  """

  @property
  def layerim3(self): return self.__layerim3
  @layerim3.setter
  def layerim3(self, layerim3): self.__layerim3 = layerim3
  layerim3: list = MetaDataAnnotation(layerim3, includeintable=False, use_default=False)
  @property
  def readlayerfile(self): return self.__readlayerfile
  @readlayerfile.setter
  def readlayerfile(self, readlayerfile): self.__readlayerfile = readlayerfile
  readlayerfile: bool = MetaDataAnnotation(True, includeintable=False, use_default=False)

  @classmethod
  def transforminitargs(cls, *args, readlayerfile=True, **kwargs):
    morekwargs = {
      "readlayerfile": readlayerfile,
    }
    if readlayerfile and "nlayersim3" not in kwargs:
      morekwargs["nlayersim3"] = 1
    return super().transforminitargs(*args, **kwargs, **morekwargs)

  @property
  def layersim3(self):
    return self.layerim3,

  @property
  def im3file(self):
    result = super().im3file
    if self.readlayerfile:
      folder = result.parent
      basename = result.name
      if basename.endswith(".camWarp") or basename.endswith(".dat"):
        basename += f"_layer{self.layerim3:02d}"
      elif basename.endswith(UNIV_CONST.FLATW_EXT):
        basename += f"{self.layerim3:02d}"
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

  @property
  def im3loadertype(self):
    if self.readlayerfile:
      return ImageLoaderIm3SingleLayer
    else:
      return ImageLoaderIm3MultiLayer
  @property
  def im3loaderkwargs(self):
    result = {
      **super().im3loaderkwargs,
    }
    if self.readlayerfile:
      result.update({
      })
    else:
      result.update({
        "nlayers": self.nlayersim3,
        "selectlayers": self.layerim3,
      })
    return result

class RectangleCorrectedIm3Base(RectangleReadIm3Base) :
  """
  Class for Rectangles whose multilayer im3 data should have one layer extracted and corrected
  for differences in exposure time, flatfielding effects, and warping effects (and or all can be omitted)

  To correct for differences in exposure time:
    et_offset = the dark current offset to use for correcting the image layer of interest
    have to also call set_med_et to set the median exposure time of the image layer in question across the whole sample
  To correct for flatfield:
    flatfield = the flatfield array
  To correct for warp:
    warp = the warping object
  """

  def __post_init__(self, *args, et_offset=None, use_flatfield=False, use_warp=None, **kwargs) :
    super().__post_init__(*args, **kwargs)
    self.__et_offset = et_offset
    self.__use_flatfield = use_flatfield
    self.__use_warp = use_warp

  @property
  def et_offset(self): return self.__et_offset
  @property
  def use_flatfield(self): return self.__use_flatfield
  @property
  def use_warp(self): return self.__use_warp

  @property
  @abc.abstractmethod
  def exposuretimetransformation(self): pass
  @property
  @abc.abstractmethod
  def flatfieldtransformation(self): pass
  def set_flatfield(self, flatfield):
    self.flatfieldtransformation.set_flatfield(flatfield)
  @property
  @abc.abstractmethod
  def warpingtransformation(self): pass
  def set_warp(self, warp):
    self.warpingtransformation.set_warp(warp)

  @methodtools.lru_cache()
  @property
  def correctedim3loader(self):
    loader = self.im3loader

    kwargs = {
      "_DEBUG": self._DEBUG,
      "_DEBUG_PRINT_TRACEBACK": self._DEBUG_PRINT_TRACEBACK,
    }

    if self.__et_offset is not None:
      transformation = self.exposuretimetransformation
      loader = TransformedImage(loader, transformation, **kwargs)

    if self.__use_flatfield:
      transformation = self.flatfieldtransformation
      loader = TransformedImage(loader, transformation, **kwargs)

    if self.__use_warp:
      transformation = self.warpingtransformation
      loader = TransformedImage(loader, transformation, **kwargs)

    return loader

  def using_corrected_im3(self):
    return self.correctedim3loader.using_image()

class RectangleCorrectedIm3SingleLayer(RectangleCorrectedIm3Base, RectangleReadIm3SingleLayer) :
  """
  Class for Rectangles whose single layer im3 data should be corrected for differences in exposure time,
  flatfielding effects, and/or warping effects (any or all can be omitted)
  """

  @methodtools.lru_cache()
  @property
  def exposuretimetransformation(self):
    return RectangleExposureTimeTransformationSingleLayer(self.allexposuretimes[self.layerim3-1],
                                                          self.et_offset)
  def set_med_et(self, med_et):
    self.exposuretimetransformation.set_med_et(med_et)
  @methodtools.lru_cache()
  @property
  def flatfieldtransformation(self):
    return RectangleFlatfieldTransformationSinglelayer()
  @methodtools.lru_cache()
  @property
  def warpingtransformation(self):
    return RectangleWarpingTransformationSinglelayer()

class RectangleCorrectedIm3MultiLayer(RectangleCorrectedIm3Base, RectangleReadIm3MultiLayer):
  """
  Class for Rectangles whose multilayer im3 data should be corrected for differences in exposure time,
  flatfielding effects, and/or warping effects (any or all can be omitted)
  """

  @methodtools.lru_cache()
  @property
  def exposuretimetransformation(self):
    return RectangleExposureTimeTransformationMultiLayer(self.allexposuretimes[tuple(_-1 for _ in self.layersim3),],
                                                         self.et_offset)
  def set_med_ets(self, med_ets):
    self.exposuretimetransformation.set_med_ets(med_ets)
  @methodtools.lru_cache()
  @property
  def flatfieldtransformation(self):
    return RectangleFlatfieldTransformationMultilayer()
  @methodtools.lru_cache()
  @property
  def warpingtransformation(self):
    return RectangleWarpingTransformationMultilayer()

class RectangleReadIHCTiff(RectangleWithImageLoaderBase) :
  """
  Rectangle class that reads the image from an IHC .tif
  """

  @property
  def ihctifffolder(self): return self.__ihctifffolder
  @ihctifffolder.setter
  def ihctifffolder(self, ihctifffolder): self.__ihctifffolder = ihctifffolder
  ihctifffolder: pathlib.Path = pathfield(ihctifffolder, includeintable=False, use_default=False)  

  @property
  def ihctifffile(self):
    return self.ihctifffolder/self.file.name.replace(UNIV_CONST.IM3_EXT, '_IHC.tif')

  @methodtools.lru_cache()
  @property
  def ihctiffloader(self): return ImageLoaderHasSingleLayerTiff(**self.ihctiffloaderkwargs)
  def using_ihc_tiff(self):
    return self.ihctiffloader.using_image()
  @property
  def ihctiffloaderkwargs(self): return {
    "filename": self.ihctifffile,
    "_DEBUG": self._DEBUG,
    "_DEBUG_PRINT_TRACEBACK": self._DEBUG_PRINT_TRACEBACK,
  }

class RectangleReadComponentTiffBase(RectangleWithImageLoaderBase):
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
  def componenttifffile(self):
    return self.componenttifffolder/self.file.name.replace(UNIV_CONST.IM3_EXT, UNIV_CONST.COMPONENT_TIFF_SUFFIX)

  @methodtools.lru_cache()
  @property
  def componenttiffloader(self): return self.componenttiffloadertype(**self.componenttiffloaderkwargs)
  def using_component_tiff(self):
    return self.componenttiffloader.using_image()
  @property
  @abc.abstractmethod
  def componenttiffloadertype(self): pass
  @property
  @abc.abstractmethod
  def componenttiffloaderkwargs(self): return {
    "nlayers": self.nlayerscomponenttiff,
    "filename": self.componenttifffile,
    "_DEBUG": self._DEBUG,
    "_DEBUG_PRINT_TRACEBACK": self._DEBUG_PRINT_TRACEBACK,
  }

class RectangleReadSegmentedComponentTiffBase(RectangleReadComponentTiffBase):
  @property
  def nsegmentations(self): return self.__nsegmentations
  @nsegmentations.setter
  def nsegmentations(self, nsegmentations): self.__nsegmentations = nsegmentations
  nsegmentations: int = MetaDataAnnotation(nsegmentations, includeintable=False, use_default=False)
  @property
  def componenttifffile(self):
    withoutseg = super().componenttifffile
    return with_stem(withoutseg, withoutseg.stem+"_w_seg")
  @property
  def componenttiffloaderkwargs(self):
    return {
      **super().componenttiffloaderkwargs,
      "nsegmentations": self.nsegmentations,
    }


class RectangleReadComponentTiffMultiLayer(RectangleReadComponentTiffBase):
  @property
  def componenttiffloadertype(self):
    return ImageLoaderComponentTiffMultiLayer

  @property
  def componenttiffloaderkwargs(self):
    return {
      **super().componenttiffloaderkwargs,
      "layers": self.layerscomponenttiff,
    }

class RectangleReadComponentTiffSingleLayer(RectangleReadComponentTiffBase):
  """
  Single layer image read from a component tiff.
  You can also use RectangleReadComponentTiffMultiLayer and write layers=[i],
  but this class gives you a 2D array as the image instead of a 3D array
  with shape[2] = 1.
  """
  @classmethod
  def transforminitargs(cls, *args, layercomponenttiff, **kwargs):
    return super().transforminitargs(*args, layerscomponenttiff=(layercomponenttiff,), **kwargs)

  @property
  def layercomponenttiff(self):
    layercomponenttiff, = self.layerscomponenttiff
    return layercomponenttiff

  @property
  def componenttiffloadertype(self):
    return ImageLoaderComponentTiffSingleLayer

  @property
  def componenttiffloaderkwargs(self):
    return {
      **super().componenttiffloaderkwargs,
      "layer": self.layercomponenttiff,
    }

class RectangleReadSegmentedComponentTiffMultiLayer(RectangleReadComponentTiffMultiLayer, RectangleReadSegmentedComponentTiffBase):
  @property
  def componenttiffloadertype(self):
    return ImageLoaderSegmentedComponentTiffMultiLayer
  @property
  def componenttiffloaderkwargs(self):
    return {
      **super().componenttiffloaderkwargs,
    }

class RectangleReadSegmentedComponentTiffSingleLayer(RectangleReadComponentTiffSingleLayer, RectangleReadSegmentedComponentTiffBase):
  @property
  def componenttiffloadertype(self):
    return ImageLoaderSegmentedComponentTiffSingleLayer
  @property
  def componenttiffloaderkwargs(self):
    return {
      **super().componenttiffloaderkwargs,
    }

class RectangleReadComponentSingleLayerAndIHCTiff(RectangleReadComponentTiffSingleLayer,RectangleReadIHCTiff) :
  pass
class RectangleReadComponentMultiLayerAndIHCTiff(RectangleReadComponentTiffMultiLayer,RectangleReadIHCTiff) :
  pass

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
  def shape(self):
    result = None
    for r in self.rectangles:
      if result is None:
        result = r.shape
      else:
        if not np.all(result == r.shape):
          raise ValueError(f"Inconsistent shapes: {result} {r.shape}")
    return result

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

      if not c: continue
      mostcommon = c.most_common()
      result[idx], mostndiffs = mostcommon[0]
      for diff, ndiffs in mostcommon:
        if ndiffs == mostndiffs:
          if diff < result[idx]:
            result[idx] = diff
        else:
          break

    assert np.count_nonzero(result)
    if result[0] == 0: result[0] = result[1] * self.shape[0] / self.shape[1]
    if result[1] == 0: result[1] = result[0] * self.shape[1] / self.shape[0]

    return result

  def showrectanglelayout(self, *, showplot=None, saveas=None, showprimaryregion=False):
    import matplotlib.patches as patches, matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    xmin = float("inf") * self.onepixel
    xmax = -float("inf") * self.onepixel
    ymin = float("inf") * self.onepixel
    ymax = -float("inf") * self.onepixel
    for r in self.rectangles:
      x, y = xy = r.xvec
      width, height = shape = r.shape
      xmin = min(xmin, x)
      xmax = max(xmax, x+width)
      ymin = min(ymin, y)
      ymax = max(ymax, y+height)
      patch = patches.Rectangle(xy / r.onepixel, *shape / r.onepixel, color="red", alpha=0.25)
      ax.add_patch(patch)
    margin = .05
    left = float((xmin - (xmax-xmin)*margin) / r.onepixel)
    right = float((xmax + (xmax-xmin)*margin) / r.onepixel)
    top = float((ymin - (ymax-ymin)*margin) / r.onepixel)
    bottom = float((ymax + (ymax-ymin)*margin) / r.onepixel)
    
    ax.set_xlim(left=left, right=right)
    ax.set_ylim(top=top, bottom=bottom)
    ax.set_aspect('equal', adjustable='box')

    if showplot is None: showplot = saveas is None
    if showplot:
      plt.show()
    if saveas is not None:
      fig.savefig(saveas)
    if not showplot:
      plt.close()

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

class GeomLoadRectangle(Rectangle):
  """
  Rectangle that has a cellGeomLoad.csv
  You have to provide the folder where that csv lives.
  """
  def __post_init__(self, *args, geomfolder, **kwargs):
    self.__geomfolder = pathlib.Path(geomfolder)
    super().__post_init__(*args, **kwargs)
  def geomloadcsv(self, segmentationalgorithm):
    return self.__geomfolder/segmentationalgorithm/self.file.name.replace(UNIV_CONST.IM3_EXT, "_cellGeomLoad.csv")

class SegmentationRectangle(Rectangle):
  """
  Rectangle that has segmentation npz files
  You have to provide the segmentation folder
  """
  def __post_init__(self, *args, segmentationfolder, **kwargs):
    self.__segmentationfolder = pathlib.Path(segmentationfolder)
    super().__post_init__(*args, **kwargs)
  @property
  def segmentationnpzfile(self):
    return self.__segmentationfolder/self.file.name.replace(UNIV_CONST.IM3_EXT, self.segmentationnpzsuffix)
  @property
  @abc.abstractmethod
  def segmentationnpzsuffix(self): pass

  @methodtools.lru_cache()
  @property
  def segmentationarrayloader(self):
    return ImageLoaderNpz(filename=self.segmentationnpzfile, key="arr_0")

  def using_segmentation_array(self):
    return self.segmentationarrayloader.using_image()

class SegmentationRectangleDeepCell(SegmentationRectangle):
  @property
  def segmentationnpzsuffix(self):
    return "_deepcell_nuclear_segmentation.npz"

class SegmentationRectangleMesmer(SegmentationRectangle):
  @property
  def segmentationnpzsuffix(self):
    return "_mesmer_segmentation.npz"

class MaskRectangleBase(Rectangle, ThingWithMask):
  pass

class TissueMaskRectangleBase(MaskRectangleBase, ThingWithTissueMask):
  pass

class AstroPathMaskRectangle(MaskRectangleBase, RectangleWithImageSize):
  """
  Rectangle that has mask files, e.g. _tissue_mask.bin
  You have to provide the folder where those files live.
  """
  def __post_init__(self, *args, maskfolder, **kwargs):
    self.__maskfolder = pathlib.Path(maskfolder)
    super().__post_init__(*args, **kwargs)
  @property
  def maskfolder(self):
    return self.__maskfolder

class IHCMaskRectangle(MaskRectangleBase, RectangleWithImageSize):
  """
  Rectangle that has IHC mask files, e.g. _tissue_mask.bin
  You have to provide the folder where those files live.
  """
  def __post_init__(self, *args, ihcmaskfolder, **kwargs):
    self.__ihcmaskfolder = pathlib.Path(ihcmaskfolder)
    super().__post_init__(*args, **kwargs)
  @property
  def ihcmaskfolder(self):
    return self.__ihcmaskfolder

class AstroPathTissueMaskRectangle(AstroPathMaskRectangle, TissueMaskRectangleBase):
  @property
  def tissuemaskfile(self):
    return self.maskfolder/self.file.name.replace(UNIV_CONST.IM3_EXT, "_tissue_mask.bin")
  @methodtools.lru_cache()
  @property
  def maskloader(self):
    return ImageLoaderBin(
      filename=self.tissuemaskfile,
      dimensions=(
        floattoint(float(self.height/self.onepixel)),
        floattoint(float(self.width/self.onepixel)),
      )
    )

  @property
  def tissuemasktransformation(self):
    return AsTypeTransformation(bool)

class IHCTissueMaskRectangle(IHCMaskRectangle, TissueMaskRectangleBase):
  @property
  def ihctissuemaskfile(self):
    return self.ihcmaskfolder/self.file.name.replace(UNIV_CONST.IM3_EXT, "_tissue_mask.bin")
  @methodtools.lru_cache()
  @property
  def maskloader(self):
    return ImageLoaderBin(
      filename=self.ihctissuemaskfile,
      dimensions=(
        floattoint(float(self.height/self.onepixel)),
        floattoint(float(self.width/self.onepixel)),
      )
    )

  @property
  def tissuemasktransformation(self):
    return AsTypeTransformation(bool)

class FullMaskRectangle(MaskRectangleBase):
  @property
  def fullmaskfile(self):
    return self.__maskfolder/self.file.name.replace(UNIV_CONST.IM3_EXT, "_full_mask.bin")
  @methodtools.lru_cache()
  @property
  def maskloader(self):
    return ImageLoaderBin(
      filename=self.fullmaskfile,
      dimensions=(
        floattoint(float(self.height/self.onepixel)),
        floattoint(float(self.width/self.onepixel)),
      )
    )

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
    return self.__phenotypetablesfolder/self.file.name.replace(UNIV_CONST.IM3_EXT, "_cleaned_phenotype_table.csv")
  @property
  def __phenotypeQAQCtablesfolder(self):
    return self.__phenotypefolder/"Results"/"QA_QC"/"Tables_QA_QC"
  @property
  def phenotypeQAQCcsv(self):
    return self.__phenotypeQAQCtablesfolder/self.file.name.replace(UNIV_CONST.IM3_EXT, "_cleaned_phenotype_table.csv")

rectanglefilter = rectangleoroverlapfilter
