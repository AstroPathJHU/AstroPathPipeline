#!/usr/bin/env python3

import contextlib, numpy as np, traceback

from ...shared.sample import DbloadSample, ReadRectanglesOverlapsFromXML, ReadRectanglesOverlapsDbloadIm3, ReadRectanglesOverlapsIm3Base, ReadRectanglesOverlapsIm3FromXML, ReadRectanglesOverlapsDbloadComponentTiff, ReadRectanglesOverlapsComponentTiffBase, ReadRectanglesOverlapsComponentTiffFromXML, SampleBase, WorkflowSample
from ...utilities.config import CONST as UNIV_CONST
from ...utilities.misc import get_GPU_thread
from ...utilities.tableio import writetable
from ..prepdb.prepdbsample import PrepDbSample
from .imagestats import ImageStats
from .overlap import AlignmentResult, AlignmentOverlap
from .rectangle import AlignmentRectangle, AlignmentRectangleBase, AlignmentRectangleComponentTiff
from .stitch import ReadStitchResult, stitch

class AlignSampleBase(SampleBase):
  """
  Main class for aligning the HPFs in a slide
  """
  def __init__(self, *args, interactive=False, useGPU=False, forceGPU=False, use_mean_image=True, **kwargs):
    """
    useGPU: use GPU for alignment if available
    forceGPU: use GPU for alignment, and raise an exception if it's not available
    use_mean_image: apply an additional flatfielding to the HPFs, derived from the
                    mean of all HPF images
    """
    self.__use_mean_image = use_mean_image
    self.interactive = interactive
    super().__init__(*args, **kwargs)

    if forceGPU: useGPU = True
    self.gpufftdict = None
    self.gputhread=get_GPU_thread(interactive) if useGPU else None
    self.__images = None

  def initrectangles(self):
    super().initrectangles()
    for r in self.rectangles:
      r.setrectanglelist(self.rectangles)

  @classmethod
  def logmodule(cls): return "align"

  def inverseoverlapsdictkey(self, overlap):
    return overlap.p2, overlap.p1

  def align(self, **kwargs):
    """
    Do the alignment over all HPF overlaps in the sample.
    The individual alignment results can be accessed from the overlaps as
    overlap.result.

    The function returns the weighted average, over all overlaps, of the
    mean squared difference in pixel fluxes.  This can be used as a quality
    check on previous stages of image processing (such as warping and flatfielding)

    Note that a failed alignment is not necessarily bad,
    it just means that there were not enough cells in the overlap area to obtain
    a result.  But you don't want to use that result to calibrate the warping model.

    Keyword arguments are passed to overlap.align()
    """
    #load the images for all HPFs and keep them in memory as long as
    #the AlignSample is active
    self.getDAPI()
    self.logger.info("starting alignment")

    weighted_sum_mse = 0.
    sum_weights = 0.
    done = set()

    for i, overlap in enumerate(self.overlaps, start=1):
      self.logger.debug(f"aligning overlap {overlap.n} ({i}/{len(self.overlaps)})")
      result = None
      #check if the inverse overlap has already been aligned
      #(e.g. if the current overlap is between (1, 2), check the overlap between (2, 1))
      #if so, we don't have to align again
      if self.inverseoverlapsdictkey(overlap) in done:
        inverseoverlap = self.overlapsdict[self.inverseoverlapsdictkey(overlap)]
        if hasattr(inverseoverlap, "result"):
          result = overlap.getinversealignment(inverseoverlap)
      #do the alignment
      if result is None:
        result = overlap.align(gputhread=self.gputhread, gpufftdict=self.gpufftdict, **kwargs)
      done.add(self.overlapsdictkey(overlap))

      #contribution of the mean squared difference after alignment
      #to the weighted sum
      if result is not None and result.exit == 0: 
        w = (overlap.cutimages[0].shape[0]*overlap.cutimages[0].shape[1])
        weighted_sum_mse+=w*result.mse[2]
        sum_weights+=w

    self.logger.info("finished align loop for "+self.SlideID)
    return weighted_sum_mse/sum_weights

  def getDAPI(self, keeprawimages=False, mean_image=None):
    #load the images
    if self.__images is None or keeprawimages:
      #create a context manager for the image loading
      #it is in self.enter_context(), so it will exit when the AlignSample does
      #and the memory will be freed
      if self.__images is None: self.__images = self.enter_context(contextlib.ExitStack())
      with contextlib.ExitStack() as stack:
        #load all the raw images, which are used for computing the flatfield
        for r in self.rectangles:
          stack.enter_context(r.using_image_before_flatfield())
          if keeprawimages:
            self.__images.enter_context(r.using_image_before_flatfield())
        #load all the actual images (calculated by dividing by the mean
        #of the raw images), while the raw images are still in memory
        for r in self.rectangles:
          self.__images.enter_context(r.using_image())

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

  rectangletype = AlignmentRectangleBase
  overlaptype = AlignmentOverlap
  alignmentresulttype = AlignmentResult
  @property
  def rectangleextrakwargs(self):
    return {**super().rectangleextrakwargs, "logger": self.logger, "use_mean_image": self.__use_mean_image}

  def stitch(self, saveresult=True, **kwargs):
    """
    Stitch the alignments together using a spring model
    """
    result = self.dostitching(**kwargs)
    if saveresult: self.applystitchresult(result)
    return result

  def dostitching(self, **kwargs):
    return stitch(overlaps=self.overlaps, rectangles=self.rectangles, origin=self.position, margin=self.margin, logger=self.logger, **kwargs)

  def applystitchresult(self, result):
    result.applytooverlaps()
    self.__T = result.T
    self.__fields = result.fields
    self.__stitchresult = result

  @property
  def T(self):
    """
    The affine matrix from the stitching
    """
    try:
      return self.__T
    except AttributeError:
      raise AttributeError("Haven't run stitching, so we don't have the T matrix")

  @property
  def fields(self):
    """
    The rectangles with additional information about their stitched positions
    """
    try:
      return self.__fields
    except AttributeError:
      raise AttributeError("Haven't run stitching, so we don't have the stitched fields")

  @property
  def fielddict(self):
    """
    A dictionary that allows accessing the fields from their index
    """
    return self.fields.rectangledict

  @property
  def stitchresult(self):
    """
    The stitch result object
    """
    try:
      return self.__stitchresult
    except AttributeError:
      raise AttributeError("Haven't run stitching")

  @stitchresult.setter
  def stitchresult(self, stitchresult):
    self.__stitchresult = stitchresult

  def run(self, *, doalignment=True, dostitching=True):
    if doalignment:
      self.getDAPI()
      self.align()
    else:
      self.readalignments()

    if dostitching:
      self.stitch()

class AlignSampleDbloadBase(AlignSampleBase, DbloadSample, WorkflowSample):
  """
  An alignment set that runs from the dbload folder and can write results
  to the dbload folder
  """
  @property
  def alignmentsfilename(self): return self.csv("align")

  def writealignments(self, *, filename=None):
    """
    Write the alignment results to align.csv
    """
    if filename is None: filename = self.alignmentsfilename
    writetable(filename, [o.result for o in self.overlaps if hasattr(o, "result")], retry=self.interactive, logger=self.logger)

  def readalignments(self, *, filename=None, interactive=True):
    """
    Read the alignment results from align.csv
    """
    interactive = interactive and self.interactive and filename is None
    if filename is None: filename = self.alignmentsfilename
    self.logger.info("reading alignments from "+str(filename))

    try:
      alignmentresults = {o.n: o for o in self.readtable(filename, self.alignmentresulttype)}
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
    """
    Write the stitch results to affine.csv (the affine matrix), fields.csv (the
    stitched positions), and fieldoverlaps.csv (selected components of the covariance
    matrix - sufficient to get the error in the relative position between two fields)
    """
    if filenames is None: filenames = self.stitchfilenames
    result.writetable(
      *filenames,
      retry=self.interactive,
      printevery=10000,
      check=check,
      logger=self.logger,
    )

  def readstitchresult(self, *, filenames=None, saveresult=True, interactive=True):
    """
    Read the stitch results from the stitched csvs
    """
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

  @classmethod
  def getoutputfiles(cls, SlideID, *, dbloadroot, **otherrootkwargs):
    dbload = dbloadroot/SlideID/UNIV_CONST.DBLOAD_DIR_NAME
    return [
      dbload/f"{SlideID}_align.csv",
      dbload/f"{SlideID}_affine.csv",
      dbload/f"{SlideID}_fields.csv",
      dbload/f"{SlideID}_fieldoverlaps.csv",
    ]

  def inputfiles(self, **kwargs):
    return super().inputfiles(**kwargs) + [
      self.csv("constants"),
      self.csv("overlap"),
      self.csv("rect"),
      *(r.imagefile for r in self.rectangles),
    ]

  @classmethod
  def workflowdependencyclasses(cls):
    return [PrepDbSample] + super().workflowdependencyclasses()

class AlignSampleFromXMLBase(AlignSampleBase, ReadRectanglesOverlapsFromXML):
  """
  An alignment set that does not rely on the dbload folder and cannot write the output.
  It is a little slower to initialize than an alignment set that does have dbload.
  """
  def __init__(self, *args, nclip, position=None, **kwargs):
    self.__nclip = nclip
    super().__init__(*args, **kwargs)
    if position is None: position = np.array([0, 0])
    self.__position = position
  @property
  def nclip(self): return self.__nclip*self.onepixel
  @property
  def position(self): return self.__position

class AlignSampleIm3Base(AlignSampleBase, ReadRectanglesOverlapsIm3Base):
  """
  An alignment set that uses im3 images
  """
  rectangletype = AlignmentRectangle
  def __init__(self, *args, filetype="flatWarp", **kwargs):
    super().__init__(*args, filetype=filetype, **kwargs)

class AlignSampleComponentTiffBase(AlignSampleBase, ReadRectanglesOverlapsComponentTiffBase):
  """
  An alignment set that uses component tiffs
  """
  rectangletype = AlignmentRectangleComponentTiff

class AlignSample(AlignSampleIm3Base, ReadRectanglesOverlapsDbloadIm3, AlignSampleDbloadBase):
  #An alignment set that runs on im3 images and can write results to the dbload folder.
  #This is the primary AlignSample class that is used for calibration.
  """
  The alignment step of the pipeline finds the relative shift between adjacent HPFs.
  It then stitches the results together using a spring model.  For more information,
  see README.md and README.pdf in this folder.
  """

class AlignSampleFromXML(AlignSampleIm3Base, ReadRectanglesOverlapsIm3FromXML, AlignSampleFromXMLBase):
  """
  An alignment set that runs on im3 images and does not rely on the dbload folder.
  This class is used for calibrating the warping.
  """

class AlignSampleComponentTiff(AlignSampleComponentTiffBase, ReadRectanglesOverlapsDbloadComponentTiff, AlignSampleDbloadBase):
  """
  An alignment set that runs on component tiff images and can write results to the dbload folder.
  This class is not currently used but is here for completeness.
  """

class AlignSampleComponentTiffFromXML(AlignSampleComponentTiffBase, AlignSampleFromXMLBase, ReadRectanglesOverlapsComponentTiffFromXML):
  """
  An alignment set that runs on component tiff images and does not rely on the dbload folder.
  This class is used for identifying overexposed HPFs.
  """

def main(args=None):
  AlignSample.runfromargumentparser(args)

if __name__ == "__main__":
  main()
