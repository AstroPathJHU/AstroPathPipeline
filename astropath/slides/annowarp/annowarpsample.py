import abc, contextlib, cvxpy as cp, itertools, methodtools, more_itertools, networkx as nx, numpy as np, PIL, skimage.filters, sklearn.linear_model, uncertainties as unc

from ...shared.argumentparser import DbloadArgumentParser, MaskArgumentParser, SelectRectanglesArgumentParser, XMLPolygonReaderArgumentParser, ZoomFolderArgumentParser
from ...shared.csvclasses import Region, Vertex
from ...shared.polygon import SimplePolygon
from ...shared.qptiff import QPTiff
from ...shared.sample import MaskWorkflowSampleBase, ReadRectanglesDbloadComponentTiff, SampleBase, WorkflowSample, XMLPolygonReader, ZoomFolderSampleBase
from ...utilities import units
from ...utilities.dataclasses import MyDataClass
from ...utilities.misc import covariance_matrix, floattoint
from ...utilities.tableio import writetable
from ...utilities.units.dataclasses import DataClassWithImscale, distancefield
from ..align.computeshift import computeshift
from ..align.field import FieldReadComponentTiffMultiLayer
from ..align.overlap import AlignmentComparison
from ..zoom.stitchmasksample import AstroPathTissueMaskSample, InformMaskSample, TissueMaskSample, StitchAstroPathTissueMaskSample, StitchInformMaskSample
from ..zoom.zoomsample import ZoomSample, ZoomSampleBase
from .stitch import AnnoWarpStitchResultDefaultModel, AnnoWarpStitchResultDefaultModelCvxpy

class QPTiffSample(SampleBase, units.ThingWithImscale):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

    self.__nentered = 0
    self.__using_qptiff_context = self.enter_context(contextlib.ExitStack())
    self.__qptiff = None

  @contextlib.contextmanager
  def using_qptiff(self):
    """
    Context manager for opening the qptiff
    """
    if self.__nentered == 0:
      #if they're not currently open
      self.__qptiff = self.__using_qptiff_context.enter_context(QPTiff(self.qptifffilename))
    self.__nentered += 1
    try:
      if self.__nentered == 1:
        self.__imageinfo #access it now to make sure it's cached
      yield self.__qptiff
    finally:
      self.__nentered -= 1
      if self.__nentered == 0:
        #if we don't have any other copies of this context manager going,
        #close the qptiff and free the memory
        self.__qptiff = None
        self.__using_qptiff_context.close()

  @methodtools.lru_cache()
  @property
  def __imageinfo(self):
    """
    Get the image info from the wsi and qptiff
    (various scales and the x and y position)
    """
    with self.using_qptiff() as fqptiff:
      zoomlevel = fqptiff.zoomlevels[0]
      apscale = zoomlevel.qpscale
      ipscale = self.pscale / apscale
      ppscale = floattoint(np.round(float(ipscale)))
      iqscale = ipscale / ppscale
      imscales = {apscale * iqscale, self.pscale / ppscale}
      imscale, = imscales
      return {
        "apscale": apscale,
        "ipscale": ipscale,
        "ppscale": ppscale,
        "iqscale": iqscale,
        "imscale": imscale,
        "xposition": fqptiff.xposition,
        "yposition": fqptiff.yposition,
      }

  @property
  def apscale(self):
    """
    The pixels/micron scale of the qptiff image
    """
    return self.__imageinfo["apscale"]
  @property
  def ipscale(self):
    """
    The ratio of pixels/micron scales of the im3 and qptiff images
    """
    return self.__imageinfo["ipscale"]
  @property
  def ppscale(self):
    """
    The ratio of pixels/micron scales of the im3 and qptiff images,
    rounded to an integer
    """
    return self.__imageinfo["ppscale"]
  @property
  def iqscale(self):
    """
    The ratio of ipscale and ppscale, i.e. the remaining non-integer
    part of the ratio of pixels/micron scales of the im3 and qptiff
    images
    """
    return self.__imageinfo["iqscale"]
  @property
  def imscale(self):
    """
    The scale used for alignment: the wsi is scaled by ppscale,
    which is the integer that brings it closest to the qptiff's
    scale, and the qptiff is scaled by whatever 1.00x is needed
    to bring it to the same scale.
    """
    return self.__imageinfo["imscale"]
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

class AnnoWarpArgumentParserBase(DbloadArgumentParser, SelectRectanglesArgumentParser, XMLPolygonReaderArgumentParser, ZoomFolderArgumentParser):
  defaulttilepixels = 100

  @classmethod
  def makeargumentparser(cls, _forworkflow=False, **kwargs):
    p = super().makeargumentparser(_forworkflow=_forworkflow, **kwargs)
    p.add_argument("--tilepixels", type=int, default=cls.defaulttilepixels, help=f"size of the tiles to use for alignment (default: {cls.defaulttilepixels})")
    if not _forworkflow:
      p.add_argument("--dont-align", action="store_true", help="read the alignments from existing csv files and just stitch")
    cls.maskselectionargumentgroup(p)
    return p

  @classmethod
  def maskselectionargumentgroup(cls, argumentparser):
    return argumentparser.add_mutually_exclusive_group(required=True)

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
      "readalignments": parsed_args_dict.pop("dont_align"),
    }
    return kwargs

  @classmethod
  def argumentparserhelpmessage(cls):
    return AnnoWarpSampleBase.__doc__

class AnnoWarpSampleBase(QPTiffSample, ReadRectanglesDbloadComponentTiff, ZoomFolderSampleBase, ZoomSampleBase, WorkflowSample, XMLPolygonReader, AnnoWarpArgumentParserBase):
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

  rectangletype = FieldReadComponentTiffMultiLayer

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
    if np.any(self.__bigtilepixels % self.__tilepixels) or np.any(self.__bigtileoffsetpixels % self.__tilepixels):
      raise ValueError("You should set the tilepixels {self.__tilepixels} so that it divides bigtilepixels {self.__bigtilepixels} and bigtileoffset {self.__bigtileoffsetpixels}")

    self.__nentered = 0
    self.__using_images_context = self.enter_context(contextlib.ExitStack())
    self.__wsi = self.__qptiff = None

    self.__images = None

  @contextlib.contextmanager
  def using_images(self):
    """
    Context manager for opening the wsi and qptiff images
    """
    if self.__nentered == 0:
      #if they're not currently open
      #disable PIL's warning when opening big images
      self.__using_images_context.enter_context(self.PILmaximagepixels())
      #open the images
      self.__wsi = self.__using_images_context.enter_context(PIL.Image.open(self.wsifilename(layer=self.wsilayer)))
      self.__qptiff = self.__using_images_context.enter_context(self.using_qptiff())
    self.__nentered += 1
    try:
      yield self.__wsi, self.__qptiff
    finally:
      self.__nentered -= 1
      if self.__nentered == 0:
        #if we don't have any other copies of this context manager going,
        #close the images and free the memory
        self.__wsi = self.__qptiff = None
        self.__using_images_context.close()

  @property
  def tilesize(self):
    """
    The tile size as a Distance
    """
    return units.Distance(pixels=self.__tilepixels, pscale=self.imscale)
  @property
  def bigtilesize(self):
    """
    The big tile size (1400, 2100) as a distance
    """
    return units.distances(pixels=self.__bigtilepixels, pscale=self.imscale)
  @property
  def bigtileoffset(self):
    """
    The big tile size (0, 1000) as a distance
    """
    return units.distances(pixels=self.__bigtileoffsetpixels, pscale=self.imscale)

  def getimages(self, *, keep=False):
    """
    Load the wsi and qptiff images and scale them to the same scale

    keep: save the images in memory so that next time you call
          this function it's quicker (default: False)
    """
    if self.__images is not None: return self.__images
    with self.using_images() as (wsi, fqptiff):
      #load the images
      zoomlevel = fqptiff.zoomlevels[0]
      qptiff = PIL.Image.fromarray(zoomlevel[self.qptifflayer-1].asarray())

      #scale them so that they're at the same scale
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

  def align(self, *, debug=False, write_result=False):
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

    initialdx = floattoint(np.rint(firstresult.dx.n * zoomfactor / self.__tilepixels) * self.__tilepixels)
    initialdy = floattoint(np.rint(firstresult.dy.n * zoomfactor / self.__tilepixels) * self.__tilepixels)

    if initialdx or initialdy:
      self.logger.warningglobal(f"found a relative shift of {firstresult.dx*zoomfactor, firstresult.dy*zoomfactor} pixels between the qptiff and wsi")

    #slice and shift the images so that they line up to within 100 pixels
    #we slice both so that they're the same size
    wsix1 = wsiy1 = qptiffx1 = qptiffy1 = 0
    qptiffy2, qptiffx2 = qptiff.shape
    wsiy2, wsix2 = wsi.shape
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

    wsiinitialslice = slice(wsiy1, wsiy2), slice(wsix1, wsix2)
    qptiffinitialslice = slice(qptiffy1, qptiffy2), slice(qptiffx1, qptiffx2)

    wsi = wsi[wsiinitialslice]
    qptiff = qptiff[qptiffinitialslice]

    onepixel = self.oneimpixel

    imscale = self.imscale
    tilesize = self.tilesize
    bigtilesize = self.bigtilesize
    bigtileoffset = self.bigtileoffset

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
      x = tilesize * (ix-1)
      y = tilesize * (iy-1)
      if y+onepixel-qshifty <= 0: continue

      #find the slice of the wsi and qptiff to use
      #note that initialdx and initialdy are not needed here
      #because we already took care of that by slicing the
      #wsi and qptiff
      slc = slice(
        floattoint(units.pixels(y, pscale=imscale)),
        floattoint(units.pixels(y+tilesize, pscale=imscale))
      ), slice(
        floattoint(units.pixels(x, pscale=imscale)),
        floattoint(units.pixels(x+tilesize, pscale=imscale)),
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
        imscale=imscale,
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
            dxvec=units.correlated_distances(
              #here we apply initialdx and initialdy so that the reported
              #result is the global shift
              pixels=(shiftresult.dx+initialdx, shiftresult.dy+initialdy),
              pscale=imscale,
              power=1,
            ),
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
    A, b, c = stitchresultcls.Abc(alignmentresults, constraintmus, constraintsigmas, floatedparams=floatedparams)

    try:
      #solve the linear equation
      result = units.np.linalg.solve(2*A, -b)
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

    #get the covariance matrix
    delta2nllfor1sigma = 1
    covariancematrix = units.np.linalg.inv(A) * delta2nllfor1sigma
    result = np.array(units.correlated_distances(distances=result, covariance=covariancematrix))

    #initialize the stitch result object
    stitchresult = stitchresultcls(result, A=A, b=b, c=c, imscale=self.imscale)

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
      imscale=self.imscale,
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
    read in the original vertices from vertices.csv
    """
    vertices = self.getXMLpolygonannotations(**kwargs)[2]
    return [
      QPTiffVertex(
        vertex=v,
        bigtilesize=units.convertpscale(self.bigtilesize, self.imscale, v.apscale),
        bigtileoffset=units.convertpscale(self.bigtileoffset, self.imscale, v.apscale),
      ) for v in vertices
    ]

  @property
  def vertices(self):
    """
    get the original vertices in im3 coordinates
    """
    return self.__getvertices()
  @property
  def apvertices(self):
    """
    get the original vertices in qptiff coordinates
    """
    return self.__getvertices(apscale=self.apscale, pscale=self.apscale)

  @methodtools.lru_cache()
  def __getwarpedvertices(self, *, apscale, pscale):
    """
    Create the new warped vertices
    """
    oneapmicron = units.onemicron(pscale=apscale)
    onemicron = self.onemicron
    onepixel = self.onepixel
    return [
      WarpedVertex(
        vertex=v,
        wxvec=(v.xvec + units.nominal_values(self.__stitchresult.dxvec(v, apscale=apscale))) / oneapmicron * onemicron // onepixel * onepixel,
        pscale=self.pscale,
      ) for v in self.__getvertices(apscale=apscale, pscale=pscale)
    ]

  @property
  def warpedvertices(self):
    """
    Get the new warped vertices in im3 coordinates
    """
    return self.__getwarpedvertices(apscale=self.apscale, pscale=self.pscale)

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
          np.round((oldvertex.xvec / oldvertex.oneappixel).astype(float)),
          np.round((newvertex.xvec / oldvertex.oneappixel).astype(float)),
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
          apscale=region.apscale,
          poly=SimplePolygon(vertices=newvertices, pscale=region.pscale, apscale=region.apscale)
        ),
      )
    return result

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
    writetable(filename, self.warpedvertices)

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

  def runannowarp(self, *, readalignments=False, **kwargs):
    """
    run the full chain

    readalignments: if True, read the alignments from the alignment csv file,
                    otherwise actually do the alignment
    other kwargs are passed to stitch()
    """
    self.writeannotations()
    if not readalignments:
      self.align()
      self.writealignments()
    else:
      self.readalignments()
    self.stitch(**kwargs)
    self.writestitchresult()
    self.writevertices()
    self.writeregions()

  run = runannowarp

  def inputfiles(self, **kwargs):
    return super().inputfiles(**kwargs) + [
      self.qptifffilename,
      self.wsifilename(layer=self.wsilayer),
      self.csv("fields"),
    ]

  @classmethod
  def getoutputfiles(cls, SlideID, *, dbloadroot, **otherrootkwargs):
    dbload = dbloadroot/SlideID/"dbload"
    return [
      dbload/f"{SlideID}_annowarp.csv",
      dbload/f"{SlideID}_annowarp-stitch.csv",
      dbload/f"{SlideID}_vertices.csv",
      dbload/f"{SlideID}_regions.csv",
    ]

  @classmethod
  def workflowdependencies(cls):
    return [ZoomSample] + super().workflowdependencies()

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


class AnnoWarpSampleTissueMask(AnnoWarpSampleBase, TissueMaskSample, MaskWorkflowSampleBase, AnnoWarpArgumentParserTissueMask):
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

  def align(self, *args, **kwargs):
    with self.using_tissuemask():
      return super().align(*args, **kwargs)

class AnnoWarpArgumentParserInformTissueMask(AnnoWarpArgumentParserTissueMask):
  @classmethod
  def maskselectionargumentgroup(cls, argumentparser):
    g = super().maskselectionargumentgroup(argumentparser)
    g.add_argument("--inform-mask", action="store_true", help="use the inform mask found in the component tiff to identify tissue")
    return g
  @classmethod
  def runkwargsfromargumentparser(cls, parsed_args_dict):
    assert parsed_args_dict.pop("inform_mask")
    return super().runkwargsfromargumentparser(parsed_args_dict)

class AnnoWarpSampleInformTissueMask(AnnoWarpSampleTissueMask, InformMaskSample, AnnoWarpArgumentParserInformTissueMask):
  """
  Use the tissue mask from inform in the component tiff to determine
  which tiles to use for alignment
  """

  @classmethod
  def workflowdependencies(cls):
    return [StitchInformMaskSample] + super().workflowdependencies()
  def printcuts(self, *args, **kwargs):
    super().printcuts(*args, **kwargs)
    self.logger.info("      Using Inform mask to determine tissue regions")

class AnnoWarpArgumentParserAstroPathTissueMask(AnnoWarpArgumentParserTissueMask):
  @classmethod
  def maskselectionargumentgroup(cls, argumentparser):
    g = super().maskselectionargumentgroup(argumentparser)
    g.add_argument("--astropath-mask", action="store_true", help="use the AstroPath mask to identify tissue")
    return g

  @classmethod
  def runkwargsfromargumentparser(cls, parsed_args_dict):
    assert parsed_args_dict.pop("astropath_mask")
    return super().runkwargsfromargumentparser(parsed_args_dict)

class AnnoWarpSampleAstroPathTissueMask(AnnoWarpSampleTissueMask, AstroPathTissueMaskSample, AnnoWarpArgumentParserAstroPathTissueMask):
  """
  Use the tissue mask from AstroPath to determine
  which tiles to use for alignment
  """

  @classmethod
  def workflowdependencies(cls):
    return [StitchAstroPathTissueMaskSample] + super().workflowdependencies()
  def printcuts(self, *args, **kwargs):
    super().printcuts(*args, **kwargs)
    self.logger.info("      Using AstroPath mask to determine tissue regions")

class AnnoWarpSampleSelectMask(AnnoWarpSampleInformTissueMask, AnnoWarpSampleAstroPathTissueMask):
  def __init__(self, *args, **kwargs):
    raise TypeError("This class should not be instantiated")
  sampleclass = None
  @classmethod
  def defaultunits(cls):
    result, = {_.defaultunits() for _ in cls.__bases__}
    return result
  @classmethod
  def runfromargumentparser(cls, args=None):
    p = cls.makeargumentparser()
    parsed_args = p.parse_args(args=args)
    if parsed_args.inform_mask:
      return AnnoWarpSampleInformTissueMask.runfromargumentparser(args=args)
    elif parsed_args.astropath_mask:
      return AnnoWarpSampleAstroPathTissueMask.runfromargumentparser(args=args)
    else:
      assert False


class QPTiffCoordinateBase(abc.ABC):
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
    return (self.xvec - self.bigtileoffset) // self.bigtilesize
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

class QPTiffCoordinate(MyDataClass, QPTiffCoordinateBase):
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

class QPTiffVertex(QPTiffCoordinate, Vertex):
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

class WarpedVertex(QPTiffVertex):
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

  @property
  def originalvertex(self):
    """
    The original vertex without the warped info
    """
    return QPTiffVertex(
      regionid=self.regionid,
      vid=self.vid,
      xvec=self.xvec,
      apscale=self.apscale,
      pscale=self.pscale,
      bigtilesize=self.bigtilesize,
      bigtileoffset=self.bigtileoffset,
    )

  @property
  def finalvertex(self):
    """
    The new vertex without the original info
    """
    return Vertex(
      regionid=self.regionid,
      vid=self.vid,
      im3xvec=self.wxvec,
      apscale=self.apscale,
      pscale=self.pscale,
    )

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
    if dxvec is not None:
      kwargs["dx"] = dxvec[0].n
      kwargs["dy"] = dxvec[1].n
      kwargs["covariance"] = covariance_matrix(dxvec)

    covariancematrix = kwargs.pop("covariance", None)
    if covariancematrix is not None:
      units.np.testing.assert_allclose(covariancematrix[0, 1], covariancematrix[1, 0])
      (kwargs["covxx"], kwargs["covxy"]), (kwargs["covxy"], kwargs["covyy"]) = covariancematrix

    return super().transforminitargs(*args, **kwargs)

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
  def center(self):
    """
    the center of the tile
    """
    return self.xvec + self.tilesize/2
  qptiffcoordinate = center
  @property
  def tileindex(self):
    """
    the index of the tile in [x, y]
    """
    return floattoint((self.xvec / self.tilesize).astype(float))

  @property
  def unshifted(self):
    """
    the wsi and qptiff images before they are shifted
    """
    wsi, qptiff = self.imageshandle()
    wsitile = wsi[
      units.pixels(self.y, pscale=self.imscale):units.pixels(self.y+self.tilesize, pscale=self.imscale),
      units.pixels(self.x, pscale=self.imscale):units.pixels(self.x+self.tilesize, pscale=self.imscale),
    ]
    qptifftile = qptiff[
      units.pixels(self.y, pscale=self.imscale):units.pixels(self.y+self.tilesize, pscale=self.imscale),
      units.pixels(self.x, pscale=self.imscale):units.pixels(self.x+self.tilesize, pscale=self.imscale),
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
  def imscale(self):
    result, = {_.imscale for _ in self}
    return result
  @methodtools.lru_cache()
  @property
  def adjacencygraph(self):
    """
    Graph with edges between tiles that are adjacent to each other
    (by edges, not corners)
    """
    g = nx.Graph()
    dct = {tuple(tile.tileindex): tile for tile in self}

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
  AnnoWarpSampleSelectMask.runfromargumentparser(args)

if __name__ == "__main__":
  main()
