import abc, contextlib, itertools, methodtools, more_itertools, networkx as nx, numpy as np, PIL, skimage.filters, sklearn.linear_model, uncertainties as unc

from ...shared.argumentparser import DbloadArgumentParser, MaskArgumentParser, SelectRectanglesArgumentParser, ZoomFolderArgumentParser
from ...shared.csvclasses import AnnotationInfo, Region, Vertex
from ...shared.polygon import SimplePolygon
from ...shared.qptiff import QPTiff
from ...shared.sample import MaskWorkflowSampleBase, SampleBase, WorkflowSample, XMLPolygonAnnotationReaderSampleWithOutline, ZoomFolderSampleBase
from ...utilities.config import CONST as UNIV_CONST
from ...utilities import units
from ...utilities.dataclasses import MyDataClass
from ...utilities.miscmath import covariance_matrix, floattoint
from ...utilities.optionalimports import cvxpy as cp
from ...utilities.tableio import readtable, writetable
from ...utilities.units.dataclasses import DataClassWithImscale, distancefield
from ..align.alignsample import ReadAffineShiftSample
from ..align.computeshift import computeshift
from ..align.field import Field
from ..align.overlap import AlignmentComparison
from ..annotationinfo.annotationinfo import CopyAnnotationInfoSampleBase
from ..stitchmask.stitchmasksample import AstroPathTissueMaskSample, InformMaskSample, StitchAstroPathTissueMaskSample, StitchInformMaskSample, TissueMaskSampleWithPolygons
from ..zoom.zoomsample import ZoomSample, ZoomSampleBase
from .stitch import AnnoWarpStitchResultDefaultModel, AnnoWarpStitchResultDefaultModelCvxpy

class QPTiffSample(SampleBase, units.ThingWithImscale):
  @property
  def qptiff(self):
    return
  @contextlib.contextmanager
  def using_qptiff(self):
    """
    Context manager for opening the qptiff
    """
    with contextlib.ExitStack() as stack:
      yield stack.enter_context(QPTiff(self.qptifffilename))

  @methodtools.lru_cache()
  @property
  def __imageinfo(self):
    """
    Get the x and y position from the qptiff
    """
    with self.using_qptiff() as fqptiff:
      return {
        "xposition": fqptiff.xposition,
        "yposition": fqptiff.yposition,
      }

  @property
  def xposition(self):
    """
    x position of the qptiff image
    """
    return self.__imageinfo["xposition"]
  @property
  def yposition(self):
    """
    y position of the qptiff image
    """
    return self.__imageinfo["yposition"]

class WSISample(ZoomSampleBase, ZoomFolderSampleBase):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

    self.__nentered = 0
    self.__using_wsi_context = None
    self.__wsi = None

  def __enter__(self):
    result = super().__enter__()
    self.__using_wsi_context = self.enter_context(contextlib.ExitStack())
    return result

  @contextlib.contextmanager
  def using_wsi(self, layer):
    """
    Context manager for opening the wsi
    """
    if self.__nentered == 0:
      #if it's not currently open
      #disable PIL's warning when opening big images
      self.__using_wsi_context.enter_context(self.PILmaximagepixels())
      #open the wsi
      self.__wsi = self.__using_wsi_context.enter_context(PIL.Image.open(self.wsifilename(layer=layer)))
    self.__nentered += 1
    try:
      yield self.__wsi
    finally:
      self.__nentered -= 1
      if self.__nentered == 0:
        #if we don't have any other copies of this context manager going,
        #close the wsi and free the memory
        self.__wsi = None
        self.__using_wsi_context.close()

class AnnoWarpArgumentParserBase(DbloadArgumentParser, SelectRectanglesArgumentParser, ZoomFolderArgumentParser):
  defaulttilepixels = 100

  @classmethod
  def makeargumentparser(cls, _forworkflow=False, **kwargs):
    p = super().makeargumentparser(_forworkflow=_forworkflow, **kwargs)
    p.add_argument("--tilepixels", type=int, default=cls.defaulttilepixels, help=f"size of the tiles to use for alignment (default: {cls.defaulttilepixels})")
    p.add_argument("--round-initial-shift-pixels", type=int, default=1, help="for the initial shift, shift by increments of this many pixels (default: 1)")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--constant-shift-only", action="store_true", help="don't warp the annotations, force a constant shift only")
    if not _forworkflow:
      p.add_argument("--dont-align", action="store_true", help="read the alignments from existing csv files and just stitch")
    return p

  @classmethod
  def initkwargsfromargumentparser(cls, parsed_args_dict):
    kwargs = {
      **super().initkwargsfromargumentparser(parsed_args_dict),
      "tilepixels": parsed_args_dict.pop("tilepixels"),
    }
    return kwargs

  @classmethod
  def runkwargsfromargumentparser(cls, parsed_args_dict):
    kwargs = {
      **super().runkwargsfromargumentparser(parsed_args_dict),
      "readalignments": parsed_args_dict.pop("dont_align", False),
      "roundinitialshiftpixels": parsed_args_dict.pop("round_initial_shift_pixels", 1),
    }
    if parsed_args_dict.pop("constant_shift_only"):
      kwargs["floatedparams"] = "constants"
    return kwargs

  @classmethod
  def argumentparserhelpmessage(cls):
    return AnnoWarpSampleBase.__doc__

class AnnoWarpSampleBase(QPTiffSample, WSISample, WorkflowSample, XMLPolygonAnnotationReaderSampleWithOutline, ReadAffineShiftSample, AnnoWarpArgumentParserBase):
  r"""
  The annowarp module aligns the wsi image created by zoom to the qptiff.
  It rewrites the annotations, which were drawn in qptiff coordinates,
  in im3 coordinates.

  The qptiff is scanned and stitched by the inform software in tiles of
  1400 x 2100 pixels, with an offset of 1000 pixels in the y direction.
  We divide the qptiff and wsi into tiles of 100x100 pixels and align
  them with respect to each other, then fit to a model where \Delta\vec{x}
  is linear in \vec{x} within the tile as well as the index \vec{i} of the
  tile, with a possible constant piece.
  """

  rectangletype = Field

  __bigtilepixels = np.array([1400, 2100])
  __bigtileoffsetpixels = np.array([0, 1000])

  def __init__(self, *args, tilepixels=None, **kwargs):
    """
    tilepixels: we divide the wsi and qptiff into tiles of this size
                in order to align (default: 100)
    """
    super().__init__(*args, **kwargs)
    self.wsilayer = 1
    self.qptifflayer = 1
    if tilepixels is None: tilepixels = self.defaulttilepixels
    self.__tilepixels = tilepixels
    self.__tilesdividenicely = not (np.any(self.__bigtilepixels % self.__tilepixels) or np.any(self.__bigtileoffsetpixels % self.__tilepixels))

    self.__images = None

  @contextlib.contextmanager
  def using_images(self):
    """
    Context manager for opening the wsi and qptiff images
    """
    with self.using_wsi(layer=self.wsilayer) as wsi, self.using_qptiff() as fqptiff, fqptiff.zoomlevels[0].using_image(layer=self.qptifflayer) as qptiff:
      yield wsi, qptiff

  @property
  def tilesize(self):
    """
    The tile size as a Distance
    """
    return units.convertpscale(self.__tilepixels*self.oneappixel, self.apscale, self.imscale)[()]
  @property
  def bigtilesize(self):
    """
    The big tile size (1400, 2100) as a distance
    """
    return units.convertpscale(self.__bigtilepixels*self.oneappixel, self.apscale, self.imscale)
  @property
  def bigtileoffset(self):
    """
    The big tile size (0, 1000) as a distance
    """
    return units.convertpscale(self.__bigtileoffsetpixels*self.oneappixel, self.apscale, self.imscale)

  def getimages(self, *, keep=False):
    """
    Load the wsi and qptiff images and scale them to the same scale

    keep: save the images in memory so that next time you call
          this function it's quicker (default: False)
    """
    if self.__images is not None: return self.__images
    #load the images
    with self.using_images() as (wsi, qptiff):
      #scale them so that they're at the same scale
      qptiff = PIL.Image.fromarray(qptiff)
      wsisize = np.array(wsi.size, dtype=np.uint)
      qptiffsize = np.array(qptiff.size, dtype=np.uint)
      wsisize //= self.ppscale
      qptiffsize = (qptiffsize * self.iqscale).astype(np.uint)
      wsi = wsi.resize(wsisize)
      qptiff = qptiff.resize(qptiffsize)

      #crop them to the same size
      newsize = 0, 0, np.min((wsisize[0], qptiffsize[0])), np.min((wsisize[1], qptiffsize[1]))
      wsi = wsi.crop(newsize)
      qptiff = qptiff.crop(newsize)

      wsi = np.asarray(wsi)
      qptiff = np.asarray(qptiff)

      if keep: self.__images = wsi, qptiff
      return wsi, qptiff

  def align(self, *, debug=False, write_result=False, roundinitialshiftpixels=1):
    """
    Break the wsi and qptiff into tiles and align them
    with respect to each other.  Returns a list of results.

    debug: raise exceptions instead of catching them
           and reporting the individual alignment as
           failed (default: False)
    write_result: write the alignment results to a csv
                  file (default: False)
    """
    wholewsi, wholeqptiff = wsi, qptiff = self.getimages()

    self.logger.info("doing the initial rough alignment")

    onepixel = self.oneimpixel
    imscale = self.imscale
    tilesize = self.tilesize
    bigtilesize = self.bigtilesize
    bigtileoffset = self.bigtileoffset

    #first align the two with respect to each other in case there are
    #big shifts. this makes sure that when we align a wsi tile and a
    #qptiff tile, the two actually contain at least some of the same data.
    #we do this with a signficant zoom (which also smooths the images).

    #we want to keep the tiles of 100x100 pixels, so we round the shift
    #to the nearest 100 pixels in x and y and shift by that much.  The
    #initial shift is included in the final alignment results, so that
    #the results reported are the total relative shift of the wsi and
    #qptiff.

    zoomfactor = 5
    wsizoom = PIL.Image.fromarray(wsi)
    wsizoom = np.asarray(wsizoom.resize(np.array(wsizoom.size)//zoomfactor))
    qptiffzoom = PIL.Image.fromarray(qptiff)
    qptiffzoom = np.asarray(qptiffzoom.resize(np.array(qptiffzoom.size)//zoomfactor))
    firstresult = computeshift((qptiffzoom, wsizoom), usemaxmovementcut=False)

    if roundinitialshiftpixels is None:
      roundtonearest = self.tilesize
    else:
      roundtonearest = roundinitialshiftpixels * onepixel
    roundtonearest = roundtonearest // onepixel * onepixel
    initialdx = floattoint(float(np.rint(firstresult.dx.n * zoomfactor / (roundtonearest/onepixel)) * np.rint(roundtonearest/onepixel)), rtol=1e-4) * onepixel
    initialdy = floattoint(float(np.rint(firstresult.dy.n * zoomfactor / (roundtonearest/onepixel)) * np.rint(roundtonearest/onepixel)), rtol=1e-4) * onepixel

    if initialdx or initialdy:
      self.logger.warningglobal(f"found a relative shift of {firstresult.dx*zoomfactor, firstresult.dy*zoomfactor} pixels between the qptiff and wsi")

    initialdxvec = np.array([initialdx, initialdy])

    #slice and shift the images so that they line up to within 100 pixels
    #we slice both so that they're the same size
    wsix1 = wsiy1 = qptiffx1 = qptiffy1 = 0
    qptiffy2, qptiffx2 = np.array(qptiff.shape) * onepixel
    wsiy2, wsix2 = np.asarray(wsi.shape) * onepixel
    if initialdx > 0:
      wsix1 += initialdx
      qptiffx2 -= initialdx
    else:
      qptiffx1 -= initialdx
      wsix2 += initialdx
    if initialdy > 0:
      wsiy1 += initialdy
      qptiffy2 -= initialdy
    else:
      qptiffy1 -= initialdy
      wsiy2 += initialdy

    wsiinitialslice = slice(
      floattoint(float(wsiy1 / onepixel)),
      floattoint(float(wsiy2 / onepixel)),
    ), slice(
      floattoint(float(wsix1 / onepixel)),
      floattoint(float(wsix2 / onepixel)),
    )
    qptiffinitialslice = slice(
      floattoint(float(qptiffy1 / onepixel)),
      floattoint(float(qptiffy2 / onepixel)),
    ), slice(
      floattoint(float(qptiffx1 / onepixel)),
      floattoint(float(qptiffx2 / onepixel)),
    )

    wsi = wsi[wsiinitialslice]
    qptiff = qptiff[qptiffinitialslice]

    #find the bounding box of the area we need to align
    mx1 = units.convertpscale(min(field.mx1 for field in self.rectangles), self.pscale, imscale, 1)
    mx2 = units.convertpscale(max(field.mx2 for field in self.rectangles), self.pscale, imscale, 1)
    my1 = units.convertpscale(min(field.my1 for field in self.rectangles), self.pscale, imscale, 1)
    my2 = units.convertpscale(max(field.my2 for field in self.rectangles), self.pscale, imscale, 1)

    mx2 = min(mx2, units.Distance(pixels=wsi.shape[1], pscale=imscale) - tilesize)
    my2 = min(my2, units.Distance(pixels=wsi.shape[0], pscale=imscale) - tilesize)

    #find the area we need to align in coordinates of tile index
    n1 = floattoint(float(my1//tilesize))-1
    n2 = floattoint(float(my2//tilesize))+1
    m1 = floattoint(float(mx1//tilesize))-1
    m2 = floattoint(float(mx2//tilesize))+1

    #tweak the y position by -900 for the microsocope glitches
    #sometimes the y position is < 0 but reported by the microscope
    #as 0.  we exclude the fields at negative y from align.
    qshifty = 0
    if self.yposition == 0: qshifty = units.Distance(pixels=900, pscale=imscale)

    results = AnnoWarpAlignmentResults()
    ntiles = (m2+1-m1) * (n2+1-n1)
    self.logger.info("aligning %d tiles of %d x %d pixels", ntiles, self.__tilepixels, self.__tilepixels)
    self.printcuts()
    for n, (ix, iy) in enumerate(itertools.product(np.arange(m1, m2+1), np.arange(n1, n2+1)), start=1):
      if n%100==0 or n==ntiles: self.logger.debug("aligning tile %d/%d", n, ntiles)
      x = floattoint(float(tilesize * (ix-1) // onepixel)) * onepixel
      xmax = floattoint(float(tilesize * ix // onepixel)) * onepixel
      y = floattoint(float(tilesize * (iy-1) // onepixel)) * onepixel
      ymax = floattoint(float(tilesize * iy // onepixel)) * onepixel
      if y+onepixel-qshifty <= 0: continue

      #make sure the tile doesn't span multiple qptiff tiles
      topleft = QPTiffCoordinate(np.array([x+qptiffx1, y+qptiffy1]) + 2*onepixel, bigtilesize=self.bigtilesize, bigtileoffset=self.bigtileoffset, apscale=self.imscale)
      bottomright = QPTiffCoordinate(np.array([xmax+qptiffx1, ymax+qptiffy1]) - 2.01*onepixel, bigtilesize=self.bigtilesize, bigtileoffset=self.bigtileoffset, apscale=self.imscale)
      if not np.all(topleft.bigtileindex == bottomright.bigtileindex):
        continue

      #find the slice of the wsi and qptiff to use
      #note that initialdx and initialdy are not needed here
      #because we already took care of that by slicing the
      #wsi and qptiff
      slc = slice(
        floattoint(float(y / onepixel)),
        floattoint(float(ymax / onepixel)),
      ), slice(
        floattoint(float(x / onepixel)),
        floattoint(float(xmax / onepixel)),
      )
      wsitile = wsi[slc]
      #if this ends up with no pixels inside the wsi, continue
      if not wsitile.size: continue
      #apply cuts to make sure we're in a tissue region that can be aligned
      if not self.passescut(wholewsi, wholeqptiff, wsiinitialslice, qptiffinitialslice, slc): continue
      qptifftile = qptiff[slc]

      alignmentresultkwargs = dict(
        n=n,
        x=x+qptiffx1,
        y=y+qptiffy1,
        pscale=self.pscale,
        apscale=self.apscale,
        tilesize=tilesize,
        bigtilesize=bigtilesize,
        bigtileoffset=bigtileoffset,
        imageshandle=self.getimages,
      )

      #smooth the wsi and qptiff tiles
      wsitile = wsitile - skimage.filters.gaussian(wsitile, sigma=20)
      wsitile = skimage.filters.gaussian(wsitile, sigma=3)
      qptifftile = qptifftile - skimage.filters.gaussian(qptifftile, sigma=20)
      qptifftile = skimage.filters.gaussian(qptifftile, sigma=3)

      #do the alignment
      try:
        shiftresult = computeshift((qptifftile, wsitile), usemaxmovementcut=False)
      except Exception as e:
        if debug: raise
        results.append(
          AnnoWarpAlignmentResult(
            **alignmentresultkwargs,
            dxvec=(
              units.Distance(pixels=unc.ufloat(0, 9999.), pscale=self.imscale),
              units.Distance(pixels=unc.ufloat(0, 9999.), pscale=self.imscale),
            ),
            exit=255,
            exception=e,
          )
        )
      else:
        results.append(
          AnnoWarpAlignmentResult(
            **alignmentresultkwargs,
            #here we apply initialdx and initialdy so that the reported
            #result is the global shift
            dxvec=units.correlated_distances(
              pixels=(shiftresult.dx, shiftresult.dy),
              pscale=imscale,
              power=1,
            ) + initialdxvec,
            exit=shiftresult.exit,
          )
        )
    self.__alignmentresults = results
    if not results:
      raise ValueError("Couldn't align any tiles")
    if write_result:
      self.writealignments()
    return results

  @abc.abstractmethod
  def printcuts(self):
    """print an info message to the logger describing the cuts used"""
  @abc.abstractmethod
  def passescut(self, wholewsi, wholeqptiff, wsiinitialslice, qptiffinitialslice, tileslice):
    """
    return whether the tile described by slc passes the cut

    wholewsi: the whole wsi image (before the initial slice)
    wholeqptiff: the whole qptiff image (before the initial slice)
    wsiinitialslice: the slice for the wsi from the initial alignment
    qptiffinitialslice: the slice for the qptiff from the initial alignment
    tileslice: the slice for this tile
    """

  @property
  def alignmentcsv(self):
    """
    The filename for the csv file where the alignments
    are written
    """
    return self.csv("annowarp")

  def writealignments(self, *, filename=None):
    """
    write the alignments to a csv file
    """
    if filename is None: filename = self.alignmentcsv
    alignmentresults = [result for result in self.__alignmentresults if result]
    writetable(filename, alignmentresults, logger=self.logger)

  def readalignments(self, *, filename=None):
    """
    read the alignments from a csv file
    """
    if filename is None: filename = self.alignmentcsv
    results = self.__alignmentresults = AnnoWarpAlignmentResults(self.readtable(filename, AnnoWarpAlignmentResult, extrakwargs={"tilesize": self.tilesize, "bigtilesize": self.bigtilesize, "bigtileoffset": self.bigtileoffset, "imageshandle": self.getimages}))
    return results

  @classmethod
  def logmodule(self):
    """
    The name of this module for logging purposes
    """
    return "annowarp"

  @classmethod
  def logstartregex(cls):
    new = super().logstartregex()
    old = "runAnnowarp started"
    return rf"(?:{old}|{new})"

  @classmethod
  def logendregex(cls):
    new = super().logendregex()
    old = "annowarp finished"
    return rf"(?:{old}|{new})"

  @staticmethod
  def stitchresultcls(*, model, cvxpy):
    """
    Which stitch result class to use, given a stitching model and whether
    or not to use cvxpy
    """
    return {
      "default": (AnnoWarpStitchResultDefaultModel, AnnoWarpStitchResultDefaultModelCvxpy),
    }[model][cvxpy]

  def stitch(self, *args, cvxpy=False, **kwargs):
    r"""
    Do the stitching

    cvxpy: use cvxpy for the stitching.  this does not give uncertainties
           on the stitching parameters and is mostly useful for debugging
    model: which model to use for stitching.  The only option is "default",
           which breaks the qptiff into tiles and assumes a linear dependence
           of d\vec{x} on \vec{x} within the tile and the tile index \vec{i}
    constraintmus:    means of gaussian constraints on parameters
    constraintsigmas: widths of gaussian constraints on parameters
                      both mus and sigmas have to be None or lists of the same length.  
                      set a particular mu and sigma as None in the list to leave the
                      corresponding parameter unconstrained

    For cvxpy = False only:
    floatedparams: for the default model, options are "all" and "constants"
                   you can also give an list of bools, where True means a parameter
                   is floated and False means it's fixed
    residualpullcutoff: if any tiles have a normalized residual (=residual/error) larger than this,
                        redo the stitching without those tiles (default: 5).
                        can also be None
    """
    return (self.stitch_cvxpy if cvxpy else self.stitch_nocvxpy)(*args, **kwargs)

  def stitch_nocvxpy(self, **kwargs):
    #process the keyword arguments
    #the reason we do it this way rather than in the function definition
    #is so that we can recursively call the function by excluding certain tiles
    allkwargs = kwargs.copy()
    model = kwargs.pop("model", "default")
    constraintmus = kwargs.pop("constraintmus", None)
    constraintsigmas = kwargs.pop("constraintsigmas", None)
    residualpullcutoff = kwargs.pop("residualpullcutoff", 5)
    floatedparams = kwargs.pop("floatedparams", "all")
    _removetiles = kwargs.pop("_removetiles", [])
    _choosetiles = kwargs.pop("_choosetiles", "bigislands")
    if kwargs: raise TypeError(f"Unknown kwargs {kwargs}")

    if constraintmus is constraintsigmas is None:
      self.logger.info("doing the global fit")
    else:
      self.logger.warningglobal("doing the global fit with constraints")
      for mu, sigma in more_itertools.zip_equal(constraintmus, constraintsigmas):
        self.logger.info(f"  {mu} {sigma}")

    #select the tiles to use, recursively using a looser selection if needed
    alignmentresults = AnnoWarpAlignmentResults(_ for _ in self.__alignmentresults if _.n not in _removetiles)
    if _choosetiles == "bigislands":
      alignmentresults = alignmentresults.goodconnectedresults(minislandsize=8)
      if len(alignmentresults) < 15:
        self.logger.warningglobal("didn't find good alignment results in big islands, trying to stitch with smaller islands")
        allkwargs["_choosetiles"] = "smallislands"
        return self.stitch_nocvxpy(**allkwargs)
    elif _choosetiles == "smallislands":
      alignmentresults = alignmentresults.goodconnectedresults(minislandsize=4)
      if len(alignmentresults) < 15:
        self.logger.warningglobal("didn't find good alignment results in small islands, using all good alignment results for stitching")
        allkwargs["_choosetiles"] = "all"
        return self.stitch_nocvxpy(**allkwargs)
    elif _choosetiles == "all":
      alignmentresults = alignmentresults.goodresults
    else:
      raise ValueError(f"Invalid _choosetiles {_choosetiles}")

    #get the A, b, c arrays
    #we are minimizing x^T A x + b^T x + c
    stitchresultcls = self.stitchresultcls(model=model, cvxpy=False)
    A, b, c = stitchresultcls.Abc(alignmentresults, constraintmus, constraintsigmas, floatedparams=floatedparams, logger=self.logger)

    self.logger.info(f"using {len(alignmentresults)} tiles for the fit")
    try:
      #solve the linear equation
      result = units.np.linalg.solve(2*A, -b)
      #get the covariance matrix
      delta2nllfor1sigma = 1
      covariancematrix = units.np.linalg.inv(A) * delta2nllfor1sigma
      result = np.array(units.correlated_distances(distances=result, covariance=covariancematrix))
    except np.linalg.LinAlgError:
      #if the fit fails, try again with a looser selection
      if _choosetiles == "bigislands":
        self.logger.warningglobal("fit failed using big islands, trying to stitch with smaller islands")
        allkwargs["_choosetiles"] = "smallislands"
        return self.stitch_nocvxpy(**allkwargs)
      if _choosetiles == "smallislands":
        self.logger.warningglobal("fit failed using small islands, using all good alignment results for stitching")
        allkwargs["_choosetiles"] = "all"
        return self.stitch_nocvxpy(**allkwargs)
      raise

    #initialize the stitch result object
    stitchresult = stitchresultcls(result, A=A, b=b, c=c, constraintmus=constraintmus, constraintsigmas=constraintsigmas, pscale=self.pscale, apscale=self.apscale)

    #check if there are any outliers
    #if there are, log them, remove them, and recursively rerun
    if residualpullcutoff is not None:
      removemoretiles = []
      infolines = []
      for result in alignmentresults:
        residualsq = np.sum(stitchresult.residual(result, apscale=self.imscale)**2)
        if abs(residualsq.n / residualsq.s) > residualpullcutoff:
          removemoretiles.append(result.n)
          infolines.append(f"{result.n} {residualsq}")
      if removemoretiles:
        self.logger.warningglobal(f"Alignment results {removemoretiles} are outliers (> {residualpullcutoff} sigma residuals), removing them and trying again")
        for l in infolines: self.logger.info(l)
        allkwargs["_removetiles"]=_removetiles+removemoretiles
        return self.stitch(**allkwargs)

    self.__stitchresult = stitchresult
    return self.__stitchresult

  def stitch_cvxpy(self, **kwargs):
    allkwargs = kwargs.copy()
    model = kwargs.pop("model", "default")
    constraintmus = kwargs.pop("constraintmus", None)
    constraintsigmas = kwargs.pop("constraintsigmas", None)
    _removetiles = kwargs.pop("_removetiles", [])
    _choosetiles = kwargs.pop("_choosetiles", "bigislands")
    if kwargs: raise TypeError(f"Unknown kwargs {kwargs}")

    #select the tiles to use, recursively using a looser selection if needed
    alignmentresults = AnnoWarpAlignmentResults(_ for _ in self.__alignmentresults if _.n not in _removetiles)
    if _choosetiles == "bigislands":
      alignmentresults = alignmentresults.goodconnectedresults(minislandsize=8)
      if len(alignmentresults) < 15:
        self.logger.warningglobal("didn't find good alignment results in big islands, trying to stitch with smaller islands")
        allkwargs["_choosetiles"] = "smallislands"
        return self.stitch_nocvxpy(**allkwargs)
    elif _choosetiles == "smallislands":
      alignmentresults = alignmentresults.goodconnectedresults(minislandsize=4)
      if len(alignmentresults) < 15:
        self.logger.warningglobal("didn't find good alignment results in small islands, using all good alignment results for stitching")
        allkwargs["_choosetiles"] = "all"
        return self.stitch_nocvxpy(**allkwargs)
    elif _choosetiles == "all":
      alignmentresults = alignmentresults.goodresults
    else:
      raise ValueError(f"Invalid _choosetiles {_choosetiles}")

    #get the cvxpy variable objects
    stitchresultcls = self.stitchresultcls(model=model, cvxpy=True)
    variables = stitchresultcls.makecvxpyvariables()

    tominimize = 0

    #find the residuals and add their quadratic forms to the problem
    onepixel = self.oneimpixel
    for result in alignmentresults:
      residual = stitchresultcls.cvxpyresidual(result, **variables)
      tominimize += cp.quad_form(residual, units.np.linalg.inv(result.covariance) * onepixel**2)

    #add the constraint quadratic forms to the problem
    tominimize += stitchresultcls.constraintquadforms(variables, constraintmus, constraintsigmas, imscale=self.imscale)

    #do the minimization
    minimize = cp.Minimize(tominimize)
    prob = cp.Problem(minimize)
    prob.solve()

    #create the stitch result object
    self.__stitchresult = stitchresultcls(
      problem=prob,
      pscale=self.pscale,
      apscale=self.apscale,
      constraintmus=constraintmus,
      constraintsigmas=constraintsigmas,
      **variables,
    )

    return self.__stitchresult

  @property
  def stitchcsv(self):
    """
    filename for the stitch csv file
    """
    return self.csv("annowarp-stitch")

  def writestitchresult(self, *, filename=None):
    """
    write the stitch result to file
    """
    if filename is None: filename = self.stitchcsv
    self.__stitchresult.writestitchresult(filename=filename, logger=self.logger)

  @methodtools.lru_cache()
  def __getannotations(self, **kwargs):
    """
    Read the annotations, vertices, and regions from the xml file
    """
    return self.getXMLpolygonannotations(**kwargs)[0]

  @property
  def annotations(self):
    return self.__getannotations()

  @methodtools.lru_cache()
  def __getvertices(self, **kwargs):
    """
    read in the original vertices from the xml file
    """
    vertices = self.getXMLpolygonannotations(**kwargs)[2]
    return [
      QPTiffVertex(
        vertex=v,
        bigtilesize=units.convertpscale(self.bigtilesize, self.imscale, v.annoscale),
        bigtileoffset=units.convertpscale(self.bigtileoffset, self.imscale, v.annoscale),
      ) for v in vertices
    ]

  @property
  def vertices(self):
    """
    get the original vertices
    """
    return self.__getvertices()

  @methodtools.lru_cache()
  def __getwarpedvertices(self):
    """
    Create the new warped vertices
    """
    pscale = self.pscale
    onemicron = self.onemicron
    onepixel = self.onepixel

    annotationsonwsi = any(a.isonwsi for a in self.annotations)
    annotationstoshift = any(a.isonwsi and a.isfromxml for a in self.annotations)
    annotationsonqptiff = any(a.isonqptiff for a in self.annotations)
    annotationstowarp = any(a.isonqptiff and a.isfromxml for a in self.annotations)

    if annotationsonwsi:
      pass

    if annotationstoshift:
      onezoomedinmicron = units.onemicron(pscale=pscale/2)
      myposition = self.affineshift
      for a in self.annotations:
        if a.isonwsi:
          if a.position is None:
            a.position = myposition
          a.shiftannotation = myposition - a.position
          if np.any(a.shiftannotation):
            self.logger.warning(f"shifting annotation {a.name} by {a.shiftannotation / onepixel} pixels")

    if annotationsonqptiff:
      apscale = self.apscale
      oneapmicron = units.onemicron(pscale=apscale)

    if annotationstowarp:
      stitchresult = self.__stitchresult

    result = []
    for v in self.__getvertices():
      if v.isonwsi or v.isfrommask:
        wxvec = v.xvec
        if v.isonwsi:
          wxvec = wxvec / onezoomedinmicron * onemicron
          wxvec += v.annotation.shiftannotation
        wxvec = (wxvec + .000001 * onepixel) // onepixel * onepixel
        result.append(
          WarpedVertex(
            vertex=v,
            wxvec=wxvec,
          )
        )
      elif v.isonqptiff:
        wxvec = v.xvec * 1. #convert to float, if it's int
        if v.isfromxml:
          wxvec += units.nominal_values(stitchresult.dxvec(v, apscale=apscale))
        wxvec = wxvec / oneapmicron * onemicron // onepixel * onepixel
        result.append(
          WarpedQPTiffVertex(
            vertex=v,
            wxvec=wxvec,
            pscale=pscale,
          )
        )
      else:
        assert False, v
    return result

  @property
  def warpedvertices(self):
    """
    Get the new warped vertices in im3 coordinates
    """
    return self.__getwarpedvertices()

  @methodtools.lru_cache()
  def __getregions(self, **kwargs):
    """
    read in the original regions from the xml
    """
    return self.getXMLpolygonannotations(**kwargs)[1]

  @property
  def regions(self):
    """
    read in the original regions from the xml
    """
    return self.__getregions()

  @methodtools.lru_cache()
  @property
  def warpedregions(self):
    """
    Create the new warped regions
    """
    regions = self.regions
    warpedverticesiterator = iter(self.warpedvertices)
    result = []
    for i, region in enumerate(regions, start=1):
      zipfunction = more_itertools.zip_equal if i == len(regions) else zip
      newvertices = []
      polyvertices = [v for v in self.vertices if v.regionid == region.regionid]
      for oldvertex, newvertex in zipfunction(polyvertices, warpedverticesiterator):
        if newvertex.regionid != oldvertex.regionid:
          raise ValueError(f"found inconsistent regionids between regions.csv and vertices.csv: {newvertex.regionid} {oldvertex.regionid}")
        np.testing.assert_array_equal(
          np.round((oldvertex.xvec / oldvertex.oneannopixel).astype(float)),
          np.round((newvertex.xvec / oldvertex.oneannopixel).astype(float)),
        )
        newvertices.append(newvertex.finalvertex)
      result.append(
        Region(
          regionid=region.regionid,
          sampleid=region.sampleid,
          layer=region.layer,
          rid=region.rid,
          isNeg=region.isNeg,
          type=region.type,
          nvert=region.nvert,
          pscale=region.pscale,
          annoscale=region.annoscale,
          poly=SimplePolygon(vertices=newvertices, pscale=region.pscale, annoscale=region.annoscale)
        ),
      )
    return result

  @property
  def annotationinfocsv(self):
    """
    filename for the annotation info csv file
    """
    return self.csv("annotationinfo")

  @property
  def annotationscsv(self):
    """
    filename for the annotations csv file
    """
    return self.csv("annotations")

  def writeannotations(self, *, filename=None):
    """
    Write the annotations to the csv file
    """
    self.logger.info("writing annotations")
    if filename is None: filename = self.annotationscsv
    writetable(filename, self.annotations)

  @property
  def verticescsv(self):
    """
    filename for the vertices csv file
    """
    return self.csv("vertices")

  def writevertices(self, *, filename=None):
    """
    write the warped vertices to the csv file
    """
    self.logger.info("writing vertices")
    if filename is None: filename = self.verticescsv
    writetable(filename, self.warpedvertices, rowclass=WarpedVertex)

  @property
  def regionscsv(self):
    """
    filename for the regions csv file
    """
    return self.csv("regions")

  def writeregions(self, *, filename=None):
    """
    write the warped regions to the csv file
    """
    self.logger.info("writing regions")
    if filename is None: filename = self.regionscsv
    writetable(filename, self.warpedregions)

  def runannowarp(self, *, readalignments=False, roundinitialshiftpixels=1, **kwargs):
    """
    run the full chain

    readalignments: if True, read the alignments from the alignment csv file,
                    otherwise actually do the alignment
    other kwargs are passed to stitch()
    """
    if any(a.isonqptiff for a in self.annotations if a.name != "empty"):
      if not readalignments:
        self.align(roundinitialshiftpixels=roundinitialshiftpixels)
        self.writealignments()
      else:
        self.readalignments()
      self.stitch(**kwargs)
      self.writestitchresult()
    self.writevertices()
    self.writeregions()
    #has to come after writevertices
    self.writeannotations()

  def run(self, *args, **kwargs):
    return self.runannowarp(*args, **kwargs)

  def inputfiles(self, **kwargs):
    return super().inputfiles(**kwargs) + [
      self.qptifffilename,
      self.wsifilename(layer=self.wsilayer),
      self.csv("fields"),
      self.csv("annotationinfo"),
      self.csv("affine"),
    ]

  @classmethod
  def getoutputfiles(cls, SlideID, *, dbloadroot, **kwargs):
    dbload = dbloadroot/SlideID/UNIV_CONST.DBLOAD_DIR_NAME
    result = [
      *super().getoutputfiles(SlideID=SlideID, dbloadroot=dbloadroot, **kwargs),
      dbload/f"{SlideID}_annotations.csv",
      dbload/f"{SlideID}_vertices.csv",
      dbload/f"{SlideID}_regions.csv",
    ]
    scanfolder = kwargs["im3root"]/SlideID/"im3"/f"Scan{kwargs['Scan']}"
    infocsv = dbload/f"{SlideID}_annotationinfo.csv"
    if infocsv.exists():
      infos = readtable(infocsv, AnnotationInfo, extrakwargs={"pscale": 1, "apscale": 1, "scanfolder": scanfolder})
      if any(info.isonqptiff for info in infos):
        result += [
          dbload/f"{SlideID}_annowarp.csv",
          dbload/f"{SlideID}_annowarp-stitch.csv",
        ]
    return result
  @classmethod
  def workflowdependencyclasses(cls, **kwargs):
    return [ZoomSample, CopyAnnotationInfoSampleBase] + super().workflowdependencyclasses(**kwargs)

  @property
  def workflowkwargs(self):
    return {"layers": [1], "tifflayers": None, **super().workflowkwargs}

class AnnoWarpArgumentParserTissueMask(AnnoWarpArgumentParserBase, DbloadArgumentParser, MaskArgumentParser, SelectRectanglesArgumentParser):
  defaultmintissuefraction = 0.2

  @classmethod
  def makeargumentparser(cls, **kwargs):
    p = super().makeargumentparser(**kwargs)
    p.add_argument("--min-tissue-fraction", type=float, default=cls.defaultmintissuefraction, help=f"minimum fraction of pixels in the tile that are considered tissue if it's to be used for alignment (default: {cls.defaultmintissuefraction})")
    return p

  @classmethod
  def initkwargsfromargumentparser(cls, parsed_args_dict):
    kwargs = {
      **super().initkwargsfromargumentparser(parsed_args_dict),
      "mintissuefraction": parsed_args_dict.pop("min_tissue_fraction"),
    }
    return kwargs


class AnnoWarpSampleTissueMask(AnnoWarpSampleBase, TissueMaskSampleWithPolygons, MaskWorkflowSampleBase, AnnoWarpArgumentParserTissueMask):
  """
  Use a tissue mask to determine which tiles to use for alignment

  mintissuefraction: the minimum fraction of tissue pixels in the tile
                     to be used for alignment (default: 0.2)
  """
  def __init__(self, *args, mintissuefraction=None, **kwargs):
    super().__init__(*args, **kwargs)
    if mintissuefraction is None: mintissuefraction = self.defaultmintissuefraction
    self.mintissuefraction = mintissuefraction

  def printcuts(self):
    self.logger.info(f"Cuts: {self.mintissuefraction:.0%} of the HPF is in a tissue region")
  def passescut(self, wholewsi, wholeqptiff, wsiinitialslice, qptiffinitialslice, tileslice):
    with self.using_tissuemask() as mask:
      y1, x1 = wsiinitialslice
      y1 = slice(y1.start*self.ppscale, y1.stop*self.ppscale)
      x1 = slice(x1.start*self.ppscale, x1.stop*self.ppscale)

      y2, x2 = tileslice
      y2 = slice(y2.start*self.ppscale, y2.stop*self.ppscale)
      x2 = slice(x2.start*self.ppscale, x2.stop*self.ppscale)

      maskslice = mask[y1,x1][y2,x2]

      return np.count_nonzero(maskslice) / maskslice.size >= self.mintissuefraction

  def runannowarp(self, *args, **kwargs):
    """
    Load the tissue mask once to avoid duplicate loading
    """
    with self.using_tissuemask():
      return super().runannowarp(*args, **kwargs)

  def align(self, *args, **kwargs):
    """
    Load the tissue mask once to avoid duplicate loading
    (when the full chain is run through runannowarp() this is
    not necessary, but doesn't hurt anything)
    """
    with self.using_tissuemask():
      return super().align(*args, **kwargs)

class AnnoWarpSampleInformTissueMask(AnnoWarpSampleTissueMask, InformMaskSample):
  """
  Use the tissue mask from inform in the component tiff to determine
  which tiles to use for alignment
  """

  @classmethod
  def workflowdependencyclasses(cls, **kwargs):
    return [StitchInformMaskSample] + super().workflowdependencyclasses(**kwargs)
  def printcuts(self, *args, **kwargs):
    super().printcuts(*args, **kwargs)
    self.logger.info("      Using Inform mask to determine tissue regions")

class AnnoWarpSampleAstroPathTissueMask(AnnoWarpSampleTissueMask, AstroPathTissueMaskSample):
  """
  Use the tissue mask from AstroPath to determine
  which tiles to use for alignment
  """

  @classmethod
  def workflowdependencyclasses(cls, **kwargs):
    return [StitchAstroPathTissueMaskSample] + super().workflowdependencyclasses(**kwargs)
  def printcuts(self, *args, **kwargs):
    super().printcuts(*args, **kwargs)
    self.logger.info("      Using AstroPath mask to determine tissue regions")

class QPTiffCoordinateBase(units.ThingWithApscale):
  """
  Base class for any coordinate in the qptiff that works with the big tiles
  You can get the index of the big tile and the location within the big tile
  """
  @property
  @abc.abstractmethod
  def bigtilesize(self):
    """
    (width, height) of the big qptiff tiles
    """
  @property
  @abc.abstractmethod
  def bigtileoffset(self):
    """
    offset of the first qptiff tile
    """
  @property
  @abc.abstractmethod
  def qptiffcoordinate(self):
    """
    coordinate of this object in apscale
    """
  @property
  def bigtileindex(self):
    """
    Index of the big tile this coordinate is in
    """
    return (self.qptiffcoordinate - self.bigtileoffset) // self.bigtilesize
  @property
  def bigtilecorner(self):
    """
    Top left corner of the big tile this coordinate is in
    """
    return self.bigtileindex * self.bigtilesize + self.bigtileoffset
  @property
  def coordinaterelativetobigtile(self):
    """
    Location of this coordinate within the big tile
    """
    return self.qptiffcoordinate - self.bigtilecorner

class QPTiffCoordinate(QPTiffCoordinateBase):
  def __init__(self, coordinate, *, bigtilesize, bigtileoffset, apscale, **kwargs):
    self.__qptiffcoordinate = coordinate
    self.__bigtilesize = bigtilesize
    self.__bigtileoffset = bigtileoffset
    self.__apscale = apscale
    super().__init__(**kwargs)
  @property
  def bigtilesize(self): return self.__bigtilesize
  @property
  def bigtileoffset(self): return self.__bigtileoffset
  @property
  def qptiffcoordinate(self): return self.__qptiffcoordinate
  @property
  def apscale(self): return self.__apscale

class QPTiffCoordinateDataClass(MyDataClass, QPTiffCoordinateBase):
  """
  Base class for a dataclass that wants to use the big tile
  index of a qptiff coordinate.  bigtilesize and bigtileoffset
  are given to the constructor
  """
  def __post_init__(self, *args, bigtilesize, bigtileoffset, **kwargs):
    self.__bigtilesize = bigtilesize
    self.__bigtileoffset = bigtileoffset
    super().__post_init__(*args, **kwargs)
  @property
  def bigtilesize(self): return self.__bigtilesize
  @property
  def bigtileoffset(self): return self.__bigtileoffset

class QPTiffVertex(QPTiffCoordinateDataClass, Vertex):
  """
  A vertex that has qptiff info
  """
  @classmethod
  def transforminitargs(cls, *args, vertex=None, **kwargs):
    vertexkwargs = {"vertex": vertex}
    if isinstance(vertex, QPTiffVertex):
      vertexkwargs.update({
        "bigtilesize": vertex.bigtilesize,
        "bigtileoffset": vertex.bigtileoffset,
      })
    return super().transforminitargs(
      *args,
      **kwargs,
      **vertexkwargs,
    )
  @property
  def qptiffcoordinate(self):
    return self.xvec

class WarpedVertex(Vertex):
  """
  A warped vertex, which includes info about the original
  and warped positions
  """
  wx: units.Distance = distancefield(pixelsormicrons="pixels", dtype=int)
  wy: units.Distance = distancefield(pixelsormicrons="pixels", dtype=int)

  @classmethod
  def transforminitargs(cls, *args, wxvec=None, **kwargs):
    wxveckwargs = {}
    if wxvec is not None:
      wxveckwargs["wx"], wxveckwargs["wy"] = wxvec

    return super().transforminitargs(
      *args,
      **kwargs,
      **wxveckwargs,
    )

  @property
  def wxvec(self):
    """
    The warped position [wx, wy] as a numpy array
    """
    return np.array([self.wx, self.wy])

  originalvertextype = Vertex

  @property
  def originalvertexkwargs(self):
    return {
      "regionid": self.regionid,
      "vid": self.vid,
      "xvec": self.xvec,
      "annoscale": self.annoscale,
      "pscale": self.pscale,
    }

  @property
  def originalvertex(self):
    """
    The original vertex without the warped info
    """
    return self.originalvertextype(**self.originalvertexkwargs)

  @property
  def finalvertex(self):
    """
    The new vertex without the original info
    """
    return Vertex(
      regionid=self.regionid,
      vid=self.vid,
      im3xvec=self.wxvec,
      annoscale=self.annoscale,
      pscale=self.pscale,
    )

class WarpedQPTiffVertex(WarpedVertex, QPTiffVertex):
  originalvertextype = QPTiffVertex
  @property
  def originalvertexkwargs(self):
    return {
      **super().originalvertexkwargs,
      "bigtilesize": self.bigtilesize,
      "bigtileoffset": self.bigtileoffset,
    }

class AnnoWarpAlignmentResult(AlignmentComparison, QPTiffCoordinateBase, DataClassWithImscale):
  """
  A result from the alignment of one tile of the annowarp

  n: the numerical id of the tile, starting from 1
  x, y: the x and y positions of the tile
  dx, dy: the shift in x and y
  covxx, covxy, covyy: the covariance matrix for dx and dy
  exit: the exit code of the alignment (0=success, nonzero=failure, 255=exception)
  """
  n: int
  x: units.Distance = distancefield(pixelsormicrons="pixels", dtype=int, pscalename="imscale")
  y: units.Distance = distancefield(pixelsormicrons="pixels", dtype=int, pscalename="imscale")
  dx: units.Distance = distancefield(pixelsormicrons="pixels", secondfunction="{:.6g}".format, pscalename="imscale")
  dy: units.Distance = distancefield(pixelsormicrons="pixels", secondfunction="{:.6g}".format, pscalename="imscale")
  covxx: units.Distance = distancefield(pixelsormicrons="pixels", power=2, secondfunction="{:.6g}".format, pscalename="imscale")
  covxy: units.Distance = distancefield(pixelsormicrons="pixels", power=2, secondfunction="{:.6g}".format, pscalename="imscale")
  covyy: units.Distance = distancefield(pixelsormicrons="pixels", power=2, secondfunction="{:.6g}".format, pscalename="imscale")
  exit: int

  @classmethod
  def transforminitargs(cls, *args, **kwargs):
    dxvec = kwargs.pop("dxvec", None)
    morekwargs = {}

    if dxvec is not None:
      morekwargs["dx"] = dxvec[0].n
      morekwargs["dy"] = dxvec[1].n
      covariancematrix = covariance_matrix(dxvec)
    else:
      covariancematrix = kwargs.pop("covariance", None)

    if covariancematrix is not None:
      units.np.testing.assert_allclose(covariancematrix[0, 1], covariancematrix[1, 0])
      (morekwargs["covxx"], morekwargs["covxy"]), (morekwargs["covxy"], morekwargs["covyy"]) = covariancematrix

    return super().transforminitargs(*args, **kwargs, **morekwargs)

  def __post_init__(self, tilesize, bigtilesize, bigtileoffset, exception=None, imageshandle=None, *args, **kwargs):
    self.use_gpu = False
    super().__post_init__(*args, **kwargs)
    self.tilesize = tilesize
    self.__bigtilesize = bigtilesize
    self.__bigtileoffset = bigtileoffset
    self.exception = exception
    self.imageshandle = imageshandle

  @property
  def bigtilesize(self): return self.__bigtilesize
  @property
  def bigtileoffset(self): return self.__bigtileoffset

  @property
  def xvec(self):
    """
    [x, y] as a numpy array
    """
    return np.array([self.x, self.y])
  @property
  def covariance(self):
    """
    the covariance matrix as a numpy array
    """
    return np.array([[self.covxx, self.covxy], [self.covxy, self.covyy]])
  @property
  def dxvec(self):
    """
    [dx, dy] as a numpy array with their correlated uncertainties
    """
    return np.array(units.correlated_distances(distances=[self.dx, self.dy], covariance=self.covariance))
  @property
  def dxpscale(self):
    return self.imscale
  @property
  def center(self):
    """
    the center of the tile
    """
    return self.xvec + self.tilesize/2
  qptiffcoordinate = center

  @property
  def unshifted(self):
    """
    the wsi and qptiff images before they are shifted
    """
    wsi, qptiff = self.imageshandle()
    bigshift = abs(floattoint((np.rint(units.nominal_values(self.dxvec / self.tilesize))).astype(float)) + 2) * self.tilesize
    wsitile = wsi[
      floattoint(float(units.pixels(self.y, pscale=self.imscale))):int(float(units.pixels(self.y+self.tilesize+bigshift[1], pscale=self.imscale))),
      floattoint(float(units.pixels(self.x, pscale=self.imscale))):int(float(units.pixels(self.x+self.tilesize+bigshift[0], pscale=self.imscale))),
    ]
    qptifftile = qptiff[
      floattoint(float(units.pixels(self.y, pscale=self.imscale))):int(float(units.pixels(self.y+self.tilesize+bigshift[1], pscale=self.imscale))),
      floattoint(float(units.pixels(self.x, pscale=self.imscale))):int(float(units.pixels(self.x+self.tilesize+bigshift[0], pscale=self.imscale))),
    ]
    return wsitile, qptifftile

  def __bool__(self):
    return not self.exit

class AnnoWarpAlignmentResults(list, units.ThingWithImscale):
  """
  A list of alignment results with some extra methods
  """
  @property
  def goodresults(self):
    """
    All results with exit code == 0
    """
    return type(self)(r for r in self if not r.exit)
  @methodtools.lru_cache()
  @property
  def tilesize(self):
    result, = {_.tilesize for _ in self}
    return result
  @methodtools.lru_cache()
  @property
  def pscale(self):
    result, = {_.pscale for _ in self}
    return result
  @methodtools.lru_cache()
  @property
  def apscale(self):
    result, = {_.apscale for _ in self}
    return result
  @methodtools.lru_cache()
  @property
  def adjacencygraph(self):
    """
    Graph with edges between tiles that are adjacent to each other
    (by edges, not corners)
    """
    g = nx.Graph()
    minxvec = np.min([tile.xvec for tile in self], axis=0)
    dct = {tuple(floattoint(((tile.xvec - minxvec) / self.tilesize).astype(float), atol=.1)): tile for tile in self}

    for (ix, iy), tile in dct.items():
      g.add_node(tile.n, alignmentresult=tile, idx=(ix, iy))
      for otheridx in (ix+1, iy), (ix-1, iy), (ix, iy+1), (ix, iy-1):
        if otheridx in dct:
          g.add_edge(tile.n, dct[otheridx].n)

    return g

  def goodconnectedresults(self, *, minislandsize=8):
    """
    all results with 0 exit code that are in large enough islands
    """
    onepixel = self.oneimpixel
    good = self.goodresults
    g = good.adjacencygraph
    tiledict = {tile.n: tile for tile in self}
    keep = {}
    for island in nx.connected_components(g):
      if len(island) < minislandsize:
        for n in island: keep[n] = False
        continue
      tiles = [tiledict[n] for n in island]
      x = np.array([t.x for t in tiles])
      y = np.array([t.y for t in tiles])
      if len(set(x)) == 1 or len(set(y)) == 1:
        for n in island: keep[n] = False
        continue
      X = np.array([x, y]).T / onepixel
      dx = np.array([t.dx for t in tiles]) / onepixel
      wx = 1/np.array([t.covxx for t in tiles]) * onepixel**2
      dy = np.array([t.dy for t in tiles]) / onepixel
      wy = 1/np.array([t.covyy for t in tiles]) * onepixel**2
      dxfit = sklearn.linear_model.LinearRegression()
      dxfit.fit(X, dx, sample_weight=wx)
      dyfit = sklearn.linear_model.LinearRegression()
      dyfit.fit(X, dy, sample_weight=wy)
      for t in tiles:
        if abs(dxfit.predict(np.array([[t.x, t.y]])/onepixel)*onepixel - t.dx) > 10*onepixel or abs(dyfit.predict(np.array([[t.x, t.y]])/onepixel)*onepixel - t.dy) > 10*onepixel:
          keep[t.n] = False
        else:
          keep[t.n] = True
    return type(self)(_ for _ in good if keep[_.n])

def main(args=None):
  AnnoWarpSampleAstroPathTissueMask.runfromargumentparser(args)

if __name__ == "__main__":
  main()
