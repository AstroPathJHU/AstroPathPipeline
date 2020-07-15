#!/usr/bin/env python3

import cv2, methodtools, numpy as np, traceback

from ..baseclasses.sample import ReadRectangles
from ..utilities.tableio import readtable, writetable
from .flatfield import meanimage
from .imagestats import ImageStats
from .overlap import AlignmentResult, AlignmentOverlap
from .stitch import ReadStitchResult, stitch

class AlignmentSet(ReadRectangles):
  """
  Main class for aligning a set of images
  """
  def __init__(self, root1, root2, samp, *, interactive=False, useGPU=False, forceGPU=False, **kwargs):
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

    self.gpufftdict = None
    self.gputhread=self.__getGPUthread(interactive=interactive, force=forceGPU) if useGPU else None

  @property
  def logmodule(self): return "align"

  @methodtools.lru_cache()
  def image(self):
    return cv2.imread(str(self.dbload/(self.SlideID+"_qptiff.jpg")))

  def align(self,*,skip_corners=False,write_result=True,return_on_invalid_result=False,warpwarnings=False,**kwargs):
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
    if filename is None: filename = self.csv("align")
    writetable(filename, [o.result for o in self.overlaps if hasattr(o, "result")], retry=self.interactive)

  def readalignments(self, *, filename=None, interactive=True):
    interactive = interactive and self.interactive and filename is None
    if filename is None: filename = self.csv("align")
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
      images = self.getrawlayers(filetype)

      if keeprawimages:
        self.rawimages = images.copy()
        for rectangle, rawimage in zip(self.rectangles, self.rawimages):
          rectangle.rawimage = rawimage

      # apply the extra flattening

      self.meanimage = mean_image if mean_image is not None else meanimage(images, logger=self.logger)

      for image in images:
        image[:] = np.rint(image / self.meanimage.flatfield)
      self.images = images

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
      self.writecsv("imstat", self.imagestats, retry=self.interactive)

    #create the dictionary of compiled GPU FFT objects if possible
    if self.gputhread is not None :
      from reikna.fft import FFT
      #set up an FFT for images of each unique size in the set of overlaps
      self.gpufftdict = {}
      for olap in self.overlaps :
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
        np.copyto(r.image,thisupdateimg[0],casting='no')

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

  overlaptype = AlignmentOverlap #can be overridden in subclasses

  @property
  def stitchfilenames(self):
    return (
      self.csv("affine"),
      self.csv("fields"),
      self.csv("fieldoverlaps"),
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
        logger=self.logger,
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

  @property
  def layer(self): return 1
