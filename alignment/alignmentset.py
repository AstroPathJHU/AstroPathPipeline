#!/usr/bin/env python3

import cv2, logging, methodtools, numpy as np, os

from .flatfield import meanimage
from .overlap import AlignmentResult, Overlap, OverlapCollection
from .rectangle import ImageStats, Rectangle, RectangleCollection, rectangleoroverlapfilter
from .stitch import ReadStitchResult, stitch
from ..utilities.tableio import readtable, writetable

logger = logging.getLogger("align")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(message)s, %(funcName)s, %(asctime)s"))
logger.addHandler(handler)

class AlignmentSet(RectangleCollection, OverlapCollection):
  """
  Main class for aligning a set of images
  """
  def __init__(self, root1, root2, samp, *, interactive=False, selectrectangles=None, selectoverlaps=None, useGPU=False, forceGPU=False):
    """
    Directory structure should be
    root1/
      samp/
        dbload/
          samp_*.csv
          samp_qptiff.jpg
    root2/
      samp/
        samp_*.(fw01/camWarp_layer01) (if using DAPI, could also be 02 etc. to align with other markers)

    interactive: if this is true, then the script might try to prompt
                 you for input if things go wrong
    """
    logger.info(samp)
    self.root1 = root1
    self.root2 = root2
    self.samp = samp
    self.interactive = interactive

    self.rectanglefilter = rectangleoroverlapfilter(selectrectangles)
    overlapfilter = rectangleoroverlapfilter(selectoverlaps)
    self.overlapfilter = lambda o: overlapfilter(o) and o.p1 in self.rectangleindices and o.p2 in self.rectangleindices

    if not os.path.exists(os.path.join(self.root1, self.samp)):
      raise IOError(f"{os.path.join(self.root1, self.samp)} does not exist")

    self.readmetadata()
    self.rawimages=None

    self.gputhread=self.__getGPUthread(interactive=interactive, force=forceGPU) if useGPU else None

  @property
  def dbload(self):
    return os.path.join(self.root1, self.samp, "dbload")

  def readmetadata(self):
    """
    Read metadata from csv files
    """
    def intorfloat(string):
      assert isinstance(string, str)
      try: return int(string)
      except ValueError: return float(string)

    self.constants     = readtable(os.path.join(self.dbload, self.samp+"_constants.csv"), "Constant", value=intorfloat)
    self.constantsdict = {constant.name: constant.value for constant in self.constants}

    self.fwidth    = self.constantsdict["fwidth"]
    self.fheight   = self.constantsdict["fheight"]
    self.pscale    = self.constantsdict["pscale"]
    self.qpscale   = self.constantsdict["qpscale"]
    self.xposition = self.constantsdict["xposition"]
    self.yposition = self.constantsdict["yposition"]
    self.nclip     = self.constantsdict["nclip"]
    self.layer     = self.constantsdict["layer"]

    self.batch = readtable(os.path.join(self.dbload, self.samp+"_batch.csv"), "Batch", SampleID=int, Scan=int, Batch=int)
    self.scan  = f"Scan{self.batch[0].Scan:d}"

    self.annotations = readtable(os.path.join(self.dbload, self.samp+"_annotations.csv"), "Annotation", sampleid=int, layer=int, visible=int)
    self.regions     = readtable(os.path.join(self.dbload, self.samp+"_regions.csv"), "Region", regionid=int, sampleid=int, layer=int, rid=int, isNeg=int, nvert=int)
    self.vertices    = readtable(os.path.join(self.dbload, self.samp+"_vertices.csv"), "Vertex", regionid=int, vid=int, x=int, y=int)
    self.imagetable  = readtable(os.path.join(self.dbload, self.samp+"_qptiff.csv"), "ImageInfo", SampleID=int, XPosition=float, YPosition=float, XResolution=float, YResolution=float, qpscale=float, img=int)
    self.__image     = None

    self.__rectangles  = readtable(os.path.join(self.dbload, self.samp+"_rect.csv"), Rectangle, extrakwargs={"pscale": self.pscale})
    self.__rectangles = [r for r in self.rectangles if self.rectanglefilter(r)]
    self.__overlaps  = readtable(os.path.join(self.dbload, self.samp+"_overlap.csv"), self.overlaptype, extrakwargs={"pscale": self.pscale, "layer": self.layer, "rectangles": self.rectangles, "nclip": self.nclip})
    self.__overlaps = [o for o in self.overlaps if self.overlapfilter(o)]

  @property
  def overlaps(self): return self.__overlaps
  @property
  def rectangles(self): return self.__rectangles

  @property
  def rectanglesoverlaps(self): return self.rectangles, self.overlaps
  @rectanglesoverlaps.setter
  def rectanglesoverlaps(self, rectanglesoverlaps):
    rectangles, overlaps = rectanglesoverlaps
    self.__rectangles = rectangles
    self.__overlaps = overlaps
    self.initializerectangles()
    self.initializeoverlaps()

  @methodtools.lru_cache()
  def image(self):
    return cv2.imread(os.path.join(self.dbload, self.samp+"_qptiff.jpg"))

  @property
  def aligncsv(self):
    return os.path.join(self.dbload, self.samp+"_align.csv")

  def align(self,*,skip_corners=False,write_result=True,return_on_invalid_result=False,warpwarnings=False,**kwargs):
    #if the raw images haven't already been loaded, load them with the default argument
    #if self.rawimages is None :
    #  self.getDAPI()

    logger.info("starting align loop for "+self.samp)

    sum_mse = 0.; norm=0.
    done = set()

    for i, overlap in enumerate(self.overlaps, start=1):
      if skip_corners and overlap.tag in [1,3,7,9] :
        continue
      logger.info(f"aligning overlap {i}/{len(self.overlaps)}")
      if (overlap.p2, overlap.p1) in done:
        result = overlap.getinversealignment(self.overlapsdict[overlap.p2, overlap.p1])
      else:
        result = overlap.align(gputhread=self.gputhread, gpufftdict=self.gpufftdict, **kwargs)
      done.add((overlap.p1, overlap.p2))

      norm+=((overlap.cutimages[0]).shape[0])*((overlap.cutimages[0]).shape[1])
      if result is not None and result.exit == 0: 
        sum_mse+=result.mse[2]
      else :
        if result is None:
          reason = "is None"
        else:
          reason = f"has exit status {result.exit}"
        if return_on_invalid_result :
          if warpwarnings: logger.warning(f'WARNING: Overlap number {i} alignment result {reason}, returning 1e10!!')
          return 1e10
        else :
          if warpwarnings: logger.warning(f'WARNING: Overlap number {i} alignment result {reason}, adding 1e10 to sum_mse!!')
          sum_mse+=1e10

    if write_result :
      self.writealignments()

    logger.info("finished align loop for "+self.samp)
    return sum_mse/norm

  def writealignments(self, *, filename=None):
    if filename is None: filename = self.aligncsv
    writetable(filename, [o.result for o in self.overlaps if hasattr(o, "result")], retry=self.interactive)

  def readalignments(self, *, filename=None):
    if filename is None: filename = self.aligncsv
    alignmentresults = {o.n: o for o in readtable(filename, AlignmentResult, extrakwargs={"pscale": self.pscale})}
    for o in self.overlaps:
      try:
        o.result = alignmentresults[o.n]
      except KeyError:
        pass

  def getDAPI(self, filetype="flatWarpDAPI", keeprawimages=False, writeimstat=True, mean_image=None):
    logger.info(self.samp)
    rawimages = self.__getrawlayers(filetype, keep=keeprawimages)

    # apply the extra flattening

    self.meanimage = mean_image if mean_image is not None else meanimage(rawimages, self.samp)

    for image in rawimages:
        image[:] = np.rint(image / self.meanimage.flatfield)
    self.images = rawimages

    for rectangle, image in zip(self.rectangles, self.images):
      rectangle.image = image

    if writeimstat:
      self.imagestats = [
        ImageStats(
          n=rectangle.n,
          mean=np.mean(rectangle.image),
          min=np.min(rectangle.image),
          max=np.max(rectangle.image),
          std=np.std(rectangle.image),
          cx=rectangle.cx,
          cy=rectangle.cy,
        ) for rectangle in self.rectangles
      ]
      writetable(os.path.join(self.dbload, self.samp+"_imstat.csv"), self.imagestats, retry=self.interactive)

    #create the dictionary of compiled GPU FFT objects if possible
    self.gpufftdict = None
    if self.gputhread is not None :
      from reikna.fft import FFT
      #set up an FFT for images of each unique size in the set of overlaps
      self.gpufftdict = {}
      for olap in self.__overlaps :
          cutimages_shapes = tuple(im.shape for im in olap.cutimages)
          assert cutimages_shapes[0] == cutimages_shapes[1]
          if cutimages_shapes[0] not in self.gpufftdict.keys() :
              gpu_im = np.ndarray(cutimages_shapes[0],dtype=np.csingle)
              new_fft = FFT(gpu_im)
              new_fftc = new_fft.compile(self.gputhread)
              self.gpufftdict[cutimages_shapes[0]] = new_fftc

  def updateRectangleImages(self,imgs) :
    """
    Updates the "image" variable in each rectangle based on a dictionary of image layers
    imgs = list of WarpImages to use for update
    """
    for r in self.rectangles :
      thiswarpedimg=[(img.warped_image).get() for img in imgs if img.rawfile_key==r.file.rstrip('.im3')]
      assert len(thiswarpedimg)<2
      if len(thiswarpedimg)==1 :
        np.copyto(r.image,(thiswarpedimg[0]/self.meanimage.flatfield).astype(np.uint16),casting='no')
        #np.copyto(r.image,thiswarpedimg[0],casting='no') #question for Alex: applying meanimage?

  def getOverlapComparisonImagesDict(self) :
    """
    Write out a figure for each overlap showing comparisons between the original and shifted images
    """
    overlap_shift_comparisons = {}
    for o in self.overlaps :
      overlap_shift_comparisons[o.getShiftComparisonImageCodeNameTuple()]=o.getShiftComparisonImages()
    return overlap_shift_comparisons

  def __getGPUthread(self, interactive, force) :
    """
    Tries to create and return a Reikna Thread object to use for running some computations on the GPU
    interactive : if True (and some GPU is available), user will be given the option to choose a device 
    """
    #first try to import Reikna
    try :
      import reikna as rk 
    except ModuleNotFoundError :
      if force: raise
      logger.warning("WARNING: Reikna isn't installed. Please install with 'pip install reikna' to use GPU devices.")
      return None
    #create an API
    #try :
    #    api = rk.cluda.cuda_api()
    #except Exception :
    #  logger.info('CUDA-based GPU API not available; will try to get one based on OpenCL instead.')
    #  try :
    #    api = rk.cluda.ocl_api()
    #  except Exception :
    #    logger.warning('WARNING: Failed to create an OpenCL API; no GPU computation will be available!!')
    #    return None
    try :
      api = rk.cluda.ocl_api()
      #return a thread from the API
      return api.Thread.create(interactive=interactive)
    except Exception :
      if force: raise
      logger.warning('WARNING: Failed to create an OpenCL API; no GPU computation will be available!!')
      return None

  def __getrawlayers(self, filetype, keep=False):
    logger.info(self.samp)
    if filetype=="flatWarpDAPI" :
      ext = f".fw{self.layer:02d}"
    elif filetype=="camWarpDAPI" :
      ext = f".camWarp_layer{self.layer:02d}"
    else :
      raise ValueError(f"requested file type {filetype} not recognized by getrawlayers")
    path = os.path.join(self.root2, self.samp)

    rawimages = np.ndarray(shape=(len(self.rectangles), self.fheight, self.fwidth), dtype=np.uint16)

    if not self.rectangles:
      raise IOError("didn't find any rows in the rectangles table for "+self.samp, 1)

    for i, rectangle in enumerate(self.rectangles):
      #logger.info(f"loading rectangle {i+1}/{len(self.rectangles)}")
      with open(os.path.join(path, rectangle.file.replace(".im3", ext)), "rb") as f:
        #use fortran order, like matlab!
        rawimages[i] = np.memmap(
          f,
          dtype=np.uint16,
          shape=(self.fheight, self.fwidth),
          order="F",
          mode="r"
        )

    if keep:
        self.rawimages = rawimages.copy()
        for rectangle, rawimage in zip(self.rectangles, self.rawimages):
            rectangle.rawimage = rawimage

    return rawimages

  overlaptype = Overlap #can be overridden in subclasses

  @property
  def stitchfilenames(self):
    return (
      os.path.join(self.dbload, self.samp+"_stitch.csv"),
      os.path.join(self.dbload, self.samp+"_affine.csv"),
      os.path.join(self.dbload, self.samp+"_stitch_overlap_covariance.csv"),
    )

  def stitch(self, *, saveresult=True, checkwriting=False, **kwargs):
    result = stitch(overlaps=self.overlaps, rectangles=self.rectangles, **kwargs)

    if saveresult:
      result.applytooverlaps()
      self.writestitchresult(result, check=checkwriting)

    return result

  def writestitchresult(self, result, *, filenames=None, check=False):
    if filenames is None: filenames = self.stitchfilenames
    result.writetable(
      *filenames,
      retry=self.interactive,
      printevery=10000,
      check=check,
    )

  def readstitchresult(self, *, filenames=None, saveresult=True):
    if filenames is None: filenames = self.stitchfilenames
    result = ReadStitchResult(
      *filenames,
      overlaps=self.overlaps,
      rectangles=self.rectangles
    )
    if saveresult: result.applytooverlaps()
    return result

  def subset(self, *, selectrectangles=None, selectoverlaps=None):
    rectanglefilter = rectangleoroverlapfilter(selectrectangles)
    overlapfilter = rectangleoroverlapfilter(selectoverlaps)

    result = AlignmentSet(
      self.root1, self.root2, self.samp,
      interactive=self.interactive,
      selectrectangles=lambda r: self.rectanglefilter(r) and rectanglefilter(r),
      selectoverlaps=lambda o: self.overlapfilter(o) and overlapfilter(o),
    )
    for i, rectangle in enumerate(result.rectangles):
      result.rectangles[i] = [r for r in self.rectangles if r.n == rectangle.n][0]
    result.meanimage = self.meanimage
    result.images = self.images
    for i, overlap in enumerate(result.overlaps):
      result.overlaps[i] = [o for o in self.overlaps if o.n == overlap.n][0]
    return result
