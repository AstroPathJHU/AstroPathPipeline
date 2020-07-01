#!/usr/bin/env python3

import cv2, methodtools, numpy as np, traceback

from ..baseclasses.sample import FlatwSampleBase
from ..prepdb.csvclasses import Batch, Constant
from ..prepdb.overlap import RectangleOverlapCollection
from ..prepdb.rectangle import Rectangle, rectangleoroverlapfilter
from ..utilities import units
from ..utilities.misc import memmapcontext
from ..utilities.tableio import readtable, writetable
from .flatfield import meanimage
from .imagestats import ImageStats
from .overlap import AlignmentResult, AlignmentOverlap
from .stitch import ReadStitchResult, stitch

class AlignmentSet(FlatwSampleBase, RectangleOverlapCollection):
  """
  Main class for aligning a set of images
  """
  def __init__(self, root1, root2, samp, *, interactive=False, selectrectangles=None, selectoverlaps=None, onlyrectanglesinoverlaps=False, useGPU=False, forceGPU=False, imagefilenameadjustment=lambda x: x, **kwargs):
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
    super().__init__(root1, root2, samp, **kwargs)
    self.interactive = interactive

    self.rectanglefilter = rectangleoroverlapfilter(selectrectangles)
    overlapfilter = rectangleoroverlapfilter(selectoverlaps)
    self.overlapfilter = lambda o: overlapfilter(o) and o.p1 in self.rectangleindices and o.p2 in self.rectangleindices

    self.readmetadata(onlyrectanglesinoverlaps=onlyrectanglesinoverlaps)
    self.rawimages=None
    self.__imagefilenameadjustment = imagefilenameadjustment

    self.gpufftdict = None
    self.gputhread=self.__getGPUthread(interactive=interactive, force=forceGPU) if useGPU else None

  def readmetadata(self, *, onlyrectanglesinoverlaps=False):
    """
    Read metadata from csv files
    """
    def intorfloat(string):
      assert isinstance(string, str)
      try: return int(string)
      except ValueError: return float(string)

    width = height = None
    try:
      pscale = self.tiffpscale
      width = self.tiffwidth
      height = self.tiffheight
    except OSError:
      self.logger.warningglobal("couldn't find a component tiff: trusting image size and pscale from constants.csv")
      tmp = readtable(self.dbload/(self.SlideID+"_constants.csv"), Constant, extrakwargs={"pscale": 1})
      pscale = {_.value for _ in tmp if _.name == "pscale"}.pop()
    self.constants     = readtable(self.dbload/(self.SlideID+"_constants.csv"), Constant, extrakwargs={"pscale": pscale})
    self.constantsdict = {constant.name: constant.value for constant in self.constants}

    self.fwidth    = self.constantsdict["fwidth"]
    self.fheight   = self.constantsdict["fheight"]
    self.pscale    = float(self.constantsdict["pscale"])

    if width is not None:
      if (width, height) != (self.fwidth, self.fheight):
        self.logger.warningglobal(f"component tiff has size {width} {height} which is different from {self.fwidth} {self.fheight} (in constants.csv)")
        self.fwidth, self.fheight = width, height
      if self.pscale != pscale:
        if np.isclose(self.pscale, pscale, rtol=1e-6):
          warnfunction = self.logger.warning
        else:
          warnfunction = self.logger.warningglobal
        warnfunction(f"component tiff has pscale {pscale} which is different from {self.pscale} (in constants.csv)")
        self.pscale = pscale

    self.qpscale  = self.constantsdict["qpscale"]
    self.position = np.array([self.constantsdict["xposition"], self.constantsdict["yposition"]])
    self.nclip    = self.constantsdict["nclip"]
    self.layer    = 1

    self.batch = readtable(self.dbload/(self.SlideID+"_batch.csv"), Batch)
    self.scan  = f"Scan{self.batch[0].Scan:d}"

    #self.annotations = readtable(self.dbload/(self.SlideID+"_annotations.csv"), Annotation)
    #self.regions     = readtable(self.dbload/(self.SlideID+"_regions.csv"), Region)
    #self.vertices    = readtable(self.dbload/(self.SlideID+"_vertices.csv"), Vertex)
    #self.imagetable  = readtable(self.dbload/(self.SlideID+"_qptiff.csv"), QPTiffCsv, extrakwargs={"pscale": self.pscale})
    self.__image     = None

    self.__rectangles  = readtable(self.dbload/(self.SlideID+"_rect.csv"), Rectangle, extrakwargs={"pscale": self.pscale})
    self.__rectangles = [r for r in self.rectangles if self.rectanglefilter(r)]
    self.__overlaps  = readtable(self.dbload/(self.SlideID+"_overlap.csv"), self.overlaptype, filter=lambda row: row["p1"] in self.rectangleindices and row["p2"] in self.rectangleindices, extrakwargs={"pscale": self.pscale, "layer": self.layer, "rectangles": self.rectangles, "nclip": self.nclip})
    self.__overlaps = [o for o in self.overlaps if self.overlapfilter(o)]
    if onlyrectanglesinoverlaps:
      oldfilter = self.rectanglefilter
      self.rectanglefilter = lambda r: oldfilter(r) and self.selectoverlaprectangles(r)
      self.__rectangles = [r for r in self.rectangles if self.rectanglefilter(r)]

  @property
  def logmodule(self): return "align"

  @property
  def overlaps(self): return self.__overlaps
  @property
  def rectangles(self): return self.__rectangles

  @methodtools.lru_cache()
  def image(self):
    return cv2.imread(str(self.dbload/(self.SlideID+"_qptiff.jpg")))

  @property
  def aligncsv(self):
    return self.dbload/(self.SlideID+"_align.csv")

  def align(self,*,skip_corners=False,write_result=True,return_on_invalid_result=False,warpwarnings=False,**kwargs):
    #if the raw images haven't already been loaded, load them with the default argument
    #if self.rawimages is None :
    #  self.getDAPI()

    self.logger.info("starting alignment")

    sum_mse = 0.; norm=0.
    done = set()

    for i, overlap in enumerate(self.overlaps, start=1):
      if skip_corners and overlap.tag in [1,3,7,9] :
        continue
      self.logger.info(f"aligning overlap {i}/{len(self.overlaps)}")
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
          if warpwarnings: self.logger.warningglobal(f'Overlap number {i} alignment result {reason}: returning 1e10!!')
          return 1e10
        else :
          if warpwarnings: self.logger.warningglobal(f'Overlap number {i} alignment result {reason}: adding 1e10 to sum_mse!!')
          sum_mse+=1e10

    if write_result :
      self.writealignments()

    self.logger.info("finished align loop for "+self.SlideID)
    return sum_mse/norm

  def writealignments(self, *, filename=None):
    if filename is None: filename = self.aligncsv
    writetable(filename, [o.result for o in self.overlaps if hasattr(o, "result")], retry=self.interactive)

  def readalignments(self, *, filename=None, interactive=True):
    interactive = interactive and self.interactive and filename is None
    if filename is None: filename = self.aligncsv
    self.logger.info("reading alignments from "+str(filename))

    try:
      alignmentresults = {o.n: o for o in readtable(filename, AlignmentResult, extrakwargs={"pscale": self.pscale})}
    except Exception:
      if interactive:
        print()
        traceback.print_exc()
        print()
        answer = ""
        while answer.lower() not in ("y", "n"):
          answer = input(f"readalignments() gave an exception for {self.SlideID}.  Do the alignment?  [Y/N] ")
        if answer.lower() == "y":
          if not hasattr(self, "images"): self.getDAPI()
          self.align()
          return self.readalignments(interactive=False)
      raise

    for o in self.overlaps:
      try:
        o.result = alignmentresults[o.n]
      except KeyError:
        pass
    self.logger.info("done reading alignments for "+self.SlideID)

  def getDAPI(self, filetype="flatWarpDAPI", keeprawimages=False, writeimstat=True, mean_image=None, overwrite=True):
    self.logger.info("getDAPI")
    if overwrite or not hasattr(self, "images"):
      rawimages = self.__getrawlayers(filetype, keep=keeprawimages)

      # apply the extra flattening

      self.meanimage = mean_image if mean_image is not None else meanimage(rawimages, logger=self.logger)

      for image in rawimages:
        image[:] = np.rint(image / self.meanimage.flatfield)
      self.images = rawimages

    if len(self.rectangles) != len(self.images):
      raise ValueError(f"Mismatch in number of rectangles {len(self.rectangles)} and images {len(self.images)}")
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
          pscale=self.pscale,
        ) for rectangle in self.rectangles
      ]
      writetable(self.dbload/(self.SlideID+"_imstat.csv"), self.imagestats, retry=self.interactive)

    #create the dictionary of compiled GPU FFT objects if possible
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

  def updateRectangleImages(self,imgs,usewarpedimages=True) :
    """
    Updates the "image" variable in each rectangle based on a dictionary of image layers
    imgs = list of WarpImages to use for update
    """
    for r in self.rectangles :
      if usewarpedimages :
        thisupdateimg=[(img.warped_image).get() for img in imgs if img.rawfile_key==r.file.rstrip('.im3')]
      else :
        thisupdateimg=[(img.raw_image).get() for img in imgs if img.rawfile_key==r.file.rstrip('.im3')]
      assert len(thisupdateimg)<2
      if len(thisupdateimg)==1 :
        np.copyto(r.image,(thisupdateimg[0]/self.meanimage.flatfield).astype(np.uint16),casting='no')
        #np.copyto(r.image,thisupdateimg[0],casting='no') #applying meanimage?

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
      self.logger.warningglobal("Reikna isn't installed. Please install with 'pip install reikna' to use GPU devices.")
      return None
    #create an API
    #try :
    #    api = rk.cluda.cuda_api()
    #except Exception :
    #  logger.info('CUDA-based GPU API not available, will try to get one based on OpenCL instead.')
    #  try :
    #    api = rk.cluda.ocl_api()
    #  except Exception :
    #    logger.warningglobal('WARNING: Failed to create an OpenCL API, no GPU computation will be available!!')
    #    return None
    try :
      api = rk.cluda.ocl_api()
      #return a thread from the API
      return api.Thread.create(interactive=interactive)
    except Exception :
      if force: raise
      self.logger.warningglobal('Failed to create an OpenCL API, no GPU computation will be available!!')
      return None

  def __getrawlayers(self, filetype, keep=False):
    self.logger.info("getrawlayers")
    if filetype=="flatWarpDAPI" :
      ext = f".fw{self.layer:02d}"
    elif filetype=="camWarpDAPI" :
      ext = f".camWarp_layer{self.layer:02d}"
    else :
      raise ValueError(f"requested file type {filetype} not recognized by getrawlayers")
    path = self.root2/self.SlideID

    rawimages = np.ndarray(shape=(len(self.rectangles), units.pixels(self.fheight, pscale=self.pscale), units.pixels(self.fwidth, pscale=self.pscale)), dtype=np.uint16)

    if not self.rectangles:
      raise IOError("didn't find any rows in the rectangles table for "+self.SlideID, 1)

    for i, rectangle in enumerate(self.rectangles):
      filename = path/self.__imagefilenameadjustment(rectangle.file.replace(".im3", ext))
      self.logger.info(f"loading rectangle {i+1}/{len(self.rectangles)}")
      with open(filename, "rb") as f:
        #use fortran order, like matlab!
        with memmapcontext(
          f,
          dtype=np.uint16,
          shape=(units.pixels(self.fheight, pscale=self.pscale), units.pixels(self.fwidth, pscale=self.pscale)),
          order="F",
          mode="r"
        ) as memmap:
          rawimages[i] = memmap

    if keep:
        self.rawimages = rawimages.copy()
        for rectangle, rawimage in zip(self.rectangles, self.rawimages):
            rectangle.rawimage = rawimage

    return rawimages

  overlaptype = AlignmentOverlap #can be overridden in subclasses

  @property
  def stitchfilenames(self):
    return (
      self.dbload/(self.SlideID+"_affine.csv"),
      self.dbload/(self.SlideID+"_fields.csv"),
      self.dbload/(self.SlideID+"_fieldoverlaps.csv"),
    )

  def stitch(self, *, saveresult=True, checkwriting=False, **kwargs):
    result = stitch(overlaps=self.overlaps, rectangles=self.rectangles, origin=self.position, logger=self.logger, **kwargs)

    if saveresult:
      result.applytooverlaps()
      self.__T = result.T
      self.__fields = result.fields
      self.writestitchresult(result, check=checkwriting)

    return result

  @property
  def T(self):
    try:
      return self.__T
    except AttributeError:
      raise AttributeError("Haven't run stitching, so we don't have the T matrix")

  @property
  def fields(self):
    try:
      return self.__fields
    except AttributeError:
      raise AttributeError("Haven't run stitching, so we don't have the stitched fields")

  def writestitchresult(self, result, *, filenames=None, check=False):
    if filenames is None: filenames = self.stitchfilenames
    result.writetable(
      *filenames,
      retry=self.interactive,
      printevery=10000,
      check=check,
    )

  def readstitchresult(self, *, filenames=None, saveresult=True, interactive=True):
    self.logger.info("reading stitch results")
    interactive = interactive and self.interactive and saveresult and filenames is None
    if filenames is None: filenames = self.stitchfilenames

    try:
      result = ReadStitchResult(
        *filenames,
        overlaps=self.overlaps,
        rectangles=self.rectangles,
        origin=self.position,
      )
    except Exception:
      if interactive:
        print()
        traceback.print_exc()
        print()
        answer = ""
        while answer.lower() not in ("y", "n"):
          answer = input(f"readstitchresult() gave an exception for {self.SlideID}.  Do the stitching?  [Y/N] ")
        if answer.lower() == "y":
          self.stitch()
          return self.readstitchresult(interactive=False)
      raise

    if saveresult:
      result.applytooverlaps()
      self.__T = result.T
      self.__fields = result.fields
    self.logger.info("done reading stitch results")
    return result

  def subset(self, *, selectrectangles=None, selectoverlaps=None):
    rectanglefilter = rectangleoroverlapfilter(selectrectangles)
    overlapfilter = rectangleoroverlapfilter(selectoverlaps)

    result = AlignmentSet(
      self.root1, self.root2, self.SlideID,
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