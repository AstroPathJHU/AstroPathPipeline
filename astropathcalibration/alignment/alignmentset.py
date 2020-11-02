#!/usr/bin/env python3

import contextlib, cv2, methodtools, numpy as np, traceback

from ..baseclasses.overlap import RectangleOverlapCollection
from ..baseclasses.sample import FlatwSampleBase, ReadRectanglesOverlapsIm3, ReadRectanglesOverlapsIm3FromXML
from ..utilities import units
from ..utilities.tableio import readtable, writetable
from .imagestats import ImageStats
from .overlap import AlignmentResult, AlignmentOverlap
from .rectangle import AlignmentRectangle, AlignmentRectangleProvideImage
from .stitch import ReadStitchResult, stitch

class AlignmentSetBase(FlatwSampleBase, RectangleOverlapCollection):
  """
  Main class for aligning a set of images
  """
  def __init__(self, root1, root2, samp, *, interactive=False, useGPU=False, forceGPU=False, filetype="flatWarp", use_mean_image=True, **kwargs):
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
    self.__use_mean_image = use_mean_image
    self.interactive = interactive
    super().__init__(root1, root2, samp, filetype=filetype, **kwargs)
    for r in self.rectangles:
      r.setrectanglelist(self.rectangles)

    self.gpufftdict = None
    self.gputhread=self.__getGPUthread(interactive=interactive, force=forceGPU) if useGPU else None

  @property
  def logmodule(self): return "align"

  def inverseoverlapsdictkey(self, overlap):
    return overlap.p2, overlap.p1

  def align(self,*,skip_corners=False,return_on_invalid_result=False,warpwarnings=False,**kwargs):
    self.logger.info("starting alignment")

    sum_mse = 0.
    done = set()

    for i, overlap in enumerate(self.overlaps, start=1):
      if skip_corners and overlap.tag in [1,3,7,9] :
        continue
      self.logger.info(f"aligning overlap {overlap.n} ({i}/{len(self.overlaps)})")
      result = None
      if self.overlapsdictkey(overlap) in done:
        inverseoverlap = self.overlapsdict[self.inverseoverlapsdictkey(overlap)]
        if hasattr(inverseoverlap, "result"):
          result = overlap.getinversealignment(inverseoverlap)
      if result is None:
        result = overlap.align(gputhread=self.gputhread, gpufftdict=self.gpufftdict, **kwargs)
      done.add(self.overlapsdictkey(overlap))

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

    self.logger.info("finished align loop for "+self.SlideID)
    return sum_mse

  def getDAPI(self, keeprawimages=False, mean_image=None):
    self.logger.info("getDAPI")
    with contextlib.ExitStack() as stack:
      for r in self.rectangles:
        stack.enter_context(r.using_image_before_flatfield())
        if keeprawimages:
          for r in self.rectangles:
            self.enter_context(r.using_image_before_flatfield())
      self.images = [self.enter_context(r.using_image()) for r in self.rectangles]

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

  def updateRectangleImages(self,imgs,usewarpedimages=True,correct_with_meanimage=False,recalculate_meanimage=False) :
    """
    Updates the "image" variable in each rectangle based on a dictionary of image layers
    imgs            = list of WarpImages to use for update
    usewarpedimages = if True, warped rather than raw images will be read
    """
    for img in imgs:
      if usewarpedimages :
        thisupdateimg=img.warped_image
      else :
        thisupdateimg=img.raw_image

      if img.rectangle_list_index!=-1 : #if the image comes with its index in the list of rectangles it can be directly updated
        i = img.rectangle_list_index
      else : #otherwise all the rectangles have to be searched
        i = [i for (i, r) in enumerate(self.rectangles) if img.rawfile_key==r.file.rstrip('.im3')]
        assert len(i)==1
        i = i[0]

      r = self.rectangles[i]
      mean_image = None
      if not recalculate_meanimage:
        mean_image = r.meanimage
      newr = AlignmentRectangleProvideImage(rectangle=r, layer=r.layer, mean_image=mean_image, use_mean_image=correct_with_meanimage, image=thisupdateimg, readingfromfile=False)
      self.rectangles[i] = newr

    if correct_with_meanimage :
      for r in self.rectangles:
        r.setrectanglelist(self.rectangles)

    for o in self.overlaps:
      o.updaterectangles(self.rectangles)

  def getOverlapComparisonImagesDict(self) :
    """
    Write out a figure for each overlap showing comparisons between the original and shifted images
    """
    overlap_shift_comparisons = {}
    for o in self.overlaps :
      overlap_shift_comparisons[o.getShiftComparisonDetailTuple()]=o.getShiftComparisonImages()
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

  rectangletype = AlignmentRectangle
  overlaptype = AlignmentOverlap
  alignmentresulttype = AlignmentResult
  @property
  def rectangleextrakwargs(self):
    return {**super().rectangleextrakwargs, "logger": self.logger, "use_mean_image": self.__use_mean_image}

  def stitch(self, saveresult=True, **kwargs):
    result = self.dostitching(**kwargs)
    if saveresult: self.applystitchresult(result)
    return result

  def dostitching(self, **kwargs):
    return stitch(overlaps=self.overlaps, rectangles=self.rectangles, origin=self.position, logger=self.logger, **kwargs)

  def applystitchresult(self, result):
    result.applytooverlaps()
    self.__T = result.T
    self.__fields = result.fields

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

  @property
  def fielddict(self):
    return self.fields.rectangledict

class AlignmentSet(AlignmentSetBase, ReadRectanglesOverlapsIm3):
  @methodtools.lru_cache()
  def image(self):
    return cv2.imread(str(self.dbload/(self.SlideID+"_qptiff.jpg")))

  @property
  def alignmentsfilename(self): return self.csv("align")

  def writealignments(self, *, filename=None):
    if filename is None: filename = self.alignmentsfilename
    writetable(filename, [o.result for o in self.overlaps if hasattr(o, "result")], retry=self.interactive)

  def readalignments(self, *, filename=None, interactive=True):
    interactive = interactive and self.interactive and filename is None
    if filename is None: filename = self.alignmentsfilename
    self.logger.info("reading alignments from "+str(filename))

    try:
      alignmentresults = {o.n: o for o in readtable(filename, self.alignmentresulttype, extrakwargs={"pscale": self.pscale})}
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

  def getDAPI(self, *args, writeimstat=True, **kwargs):
    result = super().getDAPI(*args, **kwargs)

    if writeimstat:
      self.imagestats = []
      for rectangle in self.rectangles:
        with rectangle.using_image() as image:
          self.imagestats.append(
            ImageStats(
              n=rectangle.n,
              mean=np.mean(image),
              min=np.min(image),
              max=np.max(image),
              std=np.std(image),
              cx=rectangle.cx,
              cy=rectangle.cy,
              pscale=self.pscale,
            )
          )
      self.writecsv("imstat", self.imagestats, retry=self.interactive)

    return result

  @property
  def stitchfilenames(self):
    return (
      self.csv("affine"),
      self.csv("fields"),
      self.csv("fieldoverlaps"),
    )

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
      self.applystitchresult(result)

    self.logger.info("done reading stitch results")
    return result

  def stitch(self, *, saveresult=True, checkwriting=False, **kwargs):
    result = super().stitch(saveresult=saveresult, **kwargs)

    if saveresult:
      self.writestitchresult(result, check=checkwriting)

    return result

  def align(self, *args, write_result=True, **kwargs):
    result = super().align(*args, **kwargs)
    if write_result :
      self.writealignments()
    return result

class AlignmentSetFromXML(AlignmentSetBase, ReadRectanglesOverlapsIm3FromXML):
  def __init__(self, *args, nclip, position=None, **kwargs):
    self.__nclip = nclip
    super().__init__(*args, **kwargs)
    if position is None: position = np.array([0, 0])
    self.__position = position
  @property
  def nclip(self): return units.Distance(pixels=self.__nclip, pscale=self.pscale)
  @property
  def position(self): return self.__position
