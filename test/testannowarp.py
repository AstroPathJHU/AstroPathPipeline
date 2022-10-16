import more_itertools, numpy as np, os, pathlib, re

from astropath.shared.csvclasses import Annotation, Region
from astropath.slides.annowarp.annowarpsample import AnnoWarpAlignmentResult, AnnoWarpSampleAstroPathTissueMask, WarpedQPTiffVertex
from astropath.slides.annowarp.detectbigshift import DetectBigShiftSample
from astropath.slides.annowarp.annowarpcohort import AnnoWarpCohortAstroPathTissueMask
from astropath.slides.annowarp.stitch import AnnoWarpStitchResultEntry
from astropath.utilities import units

from .testbase import assertAlmostEqual, temporarilyreplace, TestBaseCopyInput, TestBaseSaveOutput
from .testwriteannotationinfo import emptyannotationregexfind, emptyannotationregexreplace

thisfolder = pathlib.Path(__file__).parent

class TestAnnoWarp(TestBaseCopyInput, TestBaseSaveOutput):
  @classmethod
  def filestocopy(cls):
    sourceroot = thisfolder/"data"
    destroot = thisfolder/"test_for_jenkins"/"annowarp"
    destrootrename = thisfolder/"test_for_jenkins"/"annowarp"/"renameannotation"
    destrootempty = thisfolder/"test_for_jenkins"/"annowarp"/"emptyannotation"
    for SlideID in "M206",:
      olddbload = sourceroot/SlideID/"dbload"
      newdbload = destroot/SlideID/"dbload"
      newdbloadrename = destrootrename/SlideID/"dbload"
      newdbloadempty = destrootempty/SlideID/"dbload"
      for csv in (
        "affine",
        "constants",
        "fields",
      ):
        yield olddbload/f"{SlideID}_{csv}.csv", newdbload
        yield olddbload/f"{SlideID}_{csv}.csv", newdbloadrename
        yield olddbload/f"{SlideID}_{csv}.csv", newdbloadempty

      olddbload = sourceroot/"reference"/"writeannotationinfo"/SlideID/"dbload"
      olddbloadrename = sourceroot/"reference"/"writeannotationinfo"/"renameannotation"/SlideID/"dbload"
      olddbloadempty = sourceroot/"reference"/"writeannotationinfo"/"emptyannotation"/SlideID/"dbload"
      for csv in (
        "annotationinfo",
      ):
        yield olddbload/f"{SlideID}_{csv}.csv", newdbload
        yield olddbloadrename/f"{SlideID}_{csv}.csv", newdbloadrename
        yield olddbloadempty/f"{SlideID}_{csv}.csv", newdbloadempty

      oldscanfolder = sourceroot/SlideID/"im3"/"Scan1"
      newscanfolder = destroot/SlideID/"im3"/"Scan1"
      newscanfolderempty = destrootempty/SlideID/"im3"/"Scan1"
      yield oldscanfolder/f"{SlideID}_Scan1.annotations.polygons.xml", newscanfolder
      yield oldscanfolder/f"{SlideID}_Scan1.qptiff", newscanfolder
      yield oldscanfolder/f"{SlideID}_Scan1.annotations.polygons.xml", newscanfolder, f"{SlideID}_Scan1.annotations.polygons_2.xml"

      yield oldscanfolder/f"{SlideID}_Scan1.qptiff", newscanfolderempty
      yield oldscanfolder/f"{SlideID}_Scan1.annotations.polygons.xml", newscanfolderempty, None, emptyannotationregexfind, emptyannotationregexreplace

  @classmethod
  def setUpClass(cls):
    from .data.M206.im3.Scan1.assembleqptiff import assembleqptiff
    assembleqptiff()
    from .testzoom import gunzipreference
    gunzipreference("M206")
    from .data.M206.im3.meanimage.image_masking.hackmask import hackmask
    hackmask()
    super().setUpClass()

  @property
  def outputfilenames(self):
    for root in (
      thisfolder/"test_for_jenkins"/"annowarp",
      thisfolder/"test_for_jenkins"/"annowarp"/"renameannotation",
      thisfolder/"test_for_jenkins"/"annowarp"/"emptyannotation",
    ):
      yield root/"logfiles"/"annowarp.log"
      for SlideID in "M206",:
        yield root/SlideID/"dbload"/f"{SlideID}_annotations.csv"
        yield root/SlideID/"dbload"/f"{SlideID}_annowarp.csv"
        yield root/SlideID/"dbload"/f"{SlideID}_annowarp-stitch.csv"
        yield root/SlideID/"dbload"/f"{SlideID}_regions.csv"
        yield root/SlideID/"dbload"/f"{SlideID}_vertices.csv"
        yield root/SlideID/"logfiles"/f"{SlideID}-annowarp.log"

  def compareoutput(self, SlideID, reffolder=None, dbloadroot=None, im3root=None, alignment=True):
    root = thisfolder/"data"
    zoomroot = thisfolder/"data"/"reference"/"zoom"
    logroot = thisfolder/"test_for_jenkins"/"annowarp"
    xmlfolders = []
    if dbloadroot is None: dbloadroot = logroot
    if im3root is None:
      im3root = logroot
    else:
      xmlfolders.append(thisfolder/"data"/SlideID/"im3"/"xml")
    s = AnnoWarpSampleAstroPathTissueMask(root=root, samp=SlideID, zoomroot=zoomroot, dbloadroot=dbloadroot, logroot=logroot, im3root=im3root, uselogfiles=True, tilepixels=100, xmlfolders=xmlfolders)
    if reffolder is None: reffolder = thisfolder/"data"/"reference"/"annowarp"

    alignmentfilename = s.alignmentcsv
    referencealignmentfilename = reffolder/SlideID/"dbload"/s.alignmentcsv.name
    stitchfilename = s.stitchcsv
    referencestitchfilename = reffolder/SlideID/"dbload"/s.stitchcsv.name
    verticesfilename = s.verticescsv
    referenceverticesfilename = reffolder/SlideID/"dbload"/s.verticescsv.name
    annotationsfilename = s.annotationscsv
    referenceannotationsfilename = reffolder/SlideID/"dbload"/s.annotationscsv.name
    regionsfilename = s.regionscsv
    referenceregionsfilename = reffolder/SlideID/"dbload"/s.regionscsv.name

    if alignment:
      rows = s.readtable(alignmentfilename, AnnoWarpAlignmentResult, extrakwargs={"tilesize": s.tilesize, "bigtilesize": s.bigtilesize, "bigtileoffset": s.bigtileoffset}, checkorder=True, checknewlines=True)
      targetrows = s.readtable(referencealignmentfilename, AnnoWarpAlignmentResult, extrakwargs={"tilesize": s.tilesize, "bigtilesize": s.bigtilesize, "bigtileoffset": s.bigtileoffset}, checkorder=True, checknewlines=True)
      for row, target in more_itertools.zip_equal(rows, targetrows):
        assertAlmostEqual(row, target, rtol=1e-5)

      rows = s.readtable(stitchfilename, AnnoWarpStitchResultEntry, checkorder=True, checknewlines=True)
      targetrows = s.readtable(referencestitchfilename, AnnoWarpStitchResultEntry, checkorder=True, checknewlines=True)
      for row, target in more_itertools.zip_equal(rows, targetrows):
        assertAlmostEqual(row, target, rtol=1e-4)
    else:
      self.assertFalse(alignmentfilename.exists())
      self.assertFalse(stitchfilename.exists())

    extrakwargs = {"annotationinfos": s.annotationinfo}
    annotations = rows = s.readtable(annotationsfilename, Annotation, checkorder=True, checknewlines=True, extrakwargs=extrakwargs)
    targetrows = s.readtable(referenceannotationsfilename, Annotation, checkorder=True, checknewlines=True, extrakwargs=extrakwargs)
    for row, target in more_itertools.zip_equal(rows, targetrows):
      assertAlmostEqual(row, target, rtol=1e-4)

    extrakwargs = {"bigtileoffset": s.bigtileoffset, "bigtilesize": s.bigtilesize, "annotations": annotations}
    rows = s.readtable(verticesfilename, WarpedQPTiffVertex, extrakwargs=extrakwargs, checkorder=True, checknewlines=True)
    targetrows = s.readtable(referenceverticesfilename, WarpedQPTiffVertex, extrakwargs=extrakwargs, checkorder=True, checknewlines=True)
    for row, target in more_itertools.zip_equal(rows, targetrows):
      assertAlmostEqual(row, target, rtol=1e-4)

    extrakwargs = {"annotations": annotations}
    rows = s.readtable(regionsfilename, Region, extrakwargs=extrakwargs, checkorder=True, checknewlines=True)
    targetrows = s.readtable(referenceregionsfilename, Region, extrakwargs=extrakwargs, checkorder=True, checknewlines=True)
    for row, target in more_itertools.zip_equal(rows, targetrows):
      assertAlmostEqual(row, target, rtol=1e-4)
      self.assertGreater(row.poly.area, 0)

  def testAlignment(self, SlideID="M206"):
    root = thisfolder/"data"
    zoomroot = thisfolder/"data"/"reference"/"zoom"
    dbloadroot = logroot = im3root = thisfolder/"test_for_jenkins"/"annowarp"
    maskroot = root
    xmlfolder = thisfolder/"data"/SlideID/"im3"/"xml"
    with units.setup_context("safe"):
      s = AnnoWarpSampleAstroPathTissueMask(root=root, samp=SlideID, zoomroot=zoomroot, dbloadroot=dbloadroot, logroot=logroot, im3root=im3root, maskroot=maskroot, uselogfiles=True, tilepixels=100, xmlfolders=[xmlfolder])

      try:
        AnnoWarpSampleAstroPathTissueMask.runfromargumentparser([os.fspath(s.root), SlideID, "--zoomroot", os.fspath(s.zoomroot), "--maskroot", os.fspath(s.maskroot), "--dbloadroot", os.fspath(s.dbloadroot), "--logroot", os.fspath(s.logroot), "--im3root", os.fspath(s.im3root), "--allow-local-edits", "--tilepixels", "100", "--round-initial-shift-pixels", "1", "--xmlfolder", os.fspath(xmlfolder)])

        if not s.runstatus():
          raise ValueError(f"Annowarp on {s.SlideID} {s.runstatus()}")

        self.compareoutput(SlideID)
      except:
        self.saveoutput()
        raise
      else:
        self.removeoutput()

  def testReadingWritingAlignments(self, SlideID="M206"):
    root = thisfolder/"data"
    zoomroot = thisfolder/"data"/"reference"/"zoom"
    dbloadroot = logroot = im3root = thisfolder/"test_for_jenkins"/"annowarp"
    xmlfolder = thisfolder/"data"/SlideID/"im3"/"xml"
    s = AnnoWarpSampleAstroPathTissueMask(root=root, samp=SlideID, zoomroot=zoomroot, dbloadroot=dbloadroot, logroot=logroot, im3root=im3root, tilepixels=100, xmlfolders=[xmlfolder])
    referencefilename = thisfolder/"data"/"reference"/"annowarp"/SlideID/"dbload"/s.alignmentcsv.name
    testfilename = thisfolder/"test_for_jenkins"/"annowarp"/SlideID/"testreadannowarpalignments.csv"
    testfilename.parent.mkdir(parents=True, exist_ok=True)
    s.readalignments(filename=referencefilename)
    s.writealignments(filename=testfilename)
    rows = s.readtable(testfilename, AnnoWarpAlignmentResult, extrakwargs={"tilesize": s.tilesize, "bigtilesize": s.bigtilesize, "bigtileoffset": s.bigtileoffset}, checkorder=True, checknewlines=True)
    targetrows = s.readtable(referencefilename, AnnoWarpAlignmentResult, extrakwargs={"tilesize": s.tilesize, "bigtilesize": s.bigtilesize, "bigtileoffset": s.bigtileoffset}, checkorder=True, checknewlines=True)
    for row, target in more_itertools.zip_equal(rows, targetrows):
      assertAlmostEqual(row, target, rtol=1e-5)
    testfilename.unlink()

  def testStitchCvxpy(self, SlideID="M206"):
    root = thisfolder/"data"
    zoomroot = thisfolder/"data"/"reference"/"zoom"
    dbloadroot = logroot = im3root = thisfolder/"test_for_jenkins"/"annowarp"
    xmlfolder = thisfolder/"data"/SlideID/"im3"/"xml"
    s = AnnoWarpSampleAstroPathTissueMask(root=root, samp=SlideID, zoomroot=zoomroot, dbloadroot=dbloadroot, logroot=logroot, im3root=im3root, tilepixels=100, xmlfolders=[xmlfolder])
    referencefilename = thisfolder/"data"/"reference"/"annowarp"/SlideID/"dbload"/s.alignmentcsv.name
    s.readalignments(filename=referencefilename)
    result1 = s.stitch(residualpullcutoff=None)
    result2 = s.stitch_cvxpy()

    units.np.testing.assert_allclose(result2.coeffrelativetobigtile, units.nominal_values(result1.coeffrelativetobigtile))
    units.np.testing.assert_allclose(result2.bigtileindexcoeff, units.nominal_values(result1.bigtileindexcoeff))
    units.np.testing.assert_allclose(result2.constant, units.nominal_values(result1.constant))

    x = units.nominal_values(result1.flatresult)
    units.np.testing.assert_allclose(
      result2.problem.value,
      x @ result1.A @ x + result1.b @ x + result1.c,
      rtol=0.01,
    )

  def testCohort(self, SlideID="M206", units="safe", moreargs=[], variant=None, **compareoutputkwargs):
    root = thisfolder/"data"
    zoomroot = thisfolder/"data"/"reference"/"zoom"
    dbloadroot = logroot = im3root = thisfolder/"test_for_jenkins"/"annowarp"
    xmlfolder = thisfolder/"data"/SlideID/"im3"/"xml"

    if variant == "rename":
      dbloadroot /= "renameannotation"
      logroot /= "renameannotation"
      variantkwargs = {
        "reffolder": thisfolder/"data"/"reference"/"annowarp"/"renameannotation",
      }
    elif variant == "empty":
      im3root /= "emptyannotation"
      dbloadroot /= "emptyannotation"
      logroot /= "emptyannotation"
      variantkwargs = {
        "reffolder": thisfolder/"data"/"reference"/"annowarp"/"emptyannotation",
      }
    elif variant is None:
      variantkwargs = {}
    else:
      raise ValueError(variant)

    maskroot = root
    args = [os.fspath(root), "--zoomroot", os.fspath(zoomroot), "--logroot", os.fspath(logroot), "--maskroot", os.fspath(maskroot), "--sampleregex", SlideID, "--debug", "--units", units, "--allow-local-edits", "--dbloadroot", os.fspath(dbloadroot), "--im3root", os.fspath(im3root), "--ignore-dependencies", "--rerun-finished", "--tilepixels", "100", "--round-initial-shift-pixels", "1", "--xmlfolder", os.fspath(xmlfolder)] + moreargs
    try:
      AnnoWarpCohortAstroPathTissueMask.runfromargumentparser(args)
      self.compareoutput(SlideID, **compareoutputkwargs, **variantkwargs, dbloadroot=dbloadroot, im3root=im3root)
    except:
      self.saveoutput()
      raise
    else:
      self.removeoutput()

  def testCohortFastUnits(self, SlideID="M206", **kwargs):
    self.testCohort(SlideID=SlideID, units="fast", **kwargs)

  def testWsiAnnotations(self, SlideID="M206", units="fast"):
    dbloadroot = thisfolder/"test_for_jenkins"/"annowarp"
    with open(thisfolder/"data"/"reference"/"writeannotationinfo"/"annotationposition"/SlideID/"dbload"/f"{SlideID}_annotationinfo.csv") as f:
      contents = f.read()
    with temporarilyreplace(dbloadroot/SlideID/"dbload"/f"{SlideID}_annotationinfo.csv", contents):
      self.testCohort(SlideID=SlideID, units=units, reffolder=thisfolder/"data"/"reference"/"annowarp"/"wsiannotations", alignment=False)

  def testConstraint(self, SlideID="M206"):
    root = thisfolder/"data"
    zoomroot = thisfolder/"data"/"reference"/"zoom"
    dbloadroot = logroot = im3root = thisfolder/"test_for_jenkins"/"annowarp"
    xmlfolder = thisfolder/"data"/SlideID/"im3"/"xml"

    s = AnnoWarpSampleAstroPathTissueMask(root=root, samp=SlideID, zoomroot=zoomroot, dbloadroot=dbloadroot, logroot=logroot, im3root=im3root, tilepixels=100, xmlfolders=[xmlfolder])
    referencefilename = thisfolder/"data"/"reference"/"annowarp"/SlideID/"dbload"/s.alignmentcsv.name
    s.readalignments(filename=referencefilename)
    result1 = s.stitch(residualpullcutoff=None)
    constraintmus = np.array([e.value for e in result1.stitchresultnominalentries])
    constraintsigmas = np.array([e.value for e in result1.stitchresultcovarianceentries if re.match(r"cov\((.*), \1\)", e.description)]) ** 0.5
    result2 = s.stitch(constraintmus=constraintmus, constraintsigmas=constraintsigmas, residualpullcutoff=None)
    result3 = s.stitch(constraintmus=constraintmus, constraintsigmas=constraintsigmas, residualpullcutoff=None, floatedparams="constants")

    units.np.testing.assert_allclose(units.nominal_values(result2.coeffrelativetobigtile), units.nominal_values(result1.coeffrelativetobigtile))
    units.np.testing.assert_allclose(units.nominal_values(result2.bigtileindexcoeff), units.nominal_values(result1.bigtileindexcoeff))
    units.np.testing.assert_allclose(units.nominal_values(result2.constant), units.nominal_values(result1.constant))
    units.np.testing.assert_allclose(units.nominal_values(result3.coeffrelativetobigtile), units.nominal_values(result1.coeffrelativetobigtile))
    units.np.testing.assert_allclose(units.nominal_values(result3.bigtileindexcoeff), units.nominal_values(result1.bigtileindexcoeff))
    units.np.testing.assert_allclose(units.nominal_values(result3.constant), units.nominal_values(result1.constant))
    x = units.nominal_values(result2.flatresult)
    units.np.testing.assert_allclose(
      x @ result3.A @ x + result3.b @ x + result3.c,
      x @ result2.A @ x + result2.b @ x + result2.c,
    )

    constraintmus *= 2
    result4 = s.stitch(constraintmus=constraintmus, constraintsigmas=constraintsigmas, residualpullcutoff=None)
    result5 = s.stitch_cvxpy(constraintmus=constraintmus, constraintsigmas=constraintsigmas)

    with np.testing.assert_raises(AssertionError):
      units.np.testing.assert_allclose(units.nominal_values(result2.coeffrelativetobigtile), units.nominal_values(result4.coeffrelativetobigtile))

    units.np.testing.assert_allclose(result5.coeffrelativetobigtile, units.nominal_values(result4.coeffrelativetobigtile))
    units.np.testing.assert_allclose(result5.bigtileindexcoeff, units.nominal_values(result4.bigtileindexcoeff))
    units.np.testing.assert_allclose(result5.constant, units.nominal_values(result4.constant))

    x = units.nominal_values(result4.flatresult)
    units.np.testing.assert_allclose(
      result5.problem.value,
      x @ result4.A @ x + result4.b @ x + result4.c,
      rtol=0.01,
    )

    result6 = s.stitch(constraintmus=constraintmus, constraintsigmas=constraintsigmas, residualpullcutoff=None, floatedparams="constants")
    units.np.testing.assert_allclose(units.nominal_values(result6.coeffrelativetobigtile)[:8], constraintmus[:4].reshape(2, 2))
    units.np.testing.assert_allclose(units.nominal_values(result6.bigtileindexcoeff), constraintmus[4:8].reshape(2, 2))

  def testDetectBigShift(self, SlideID="M21_1"):
    s = DetectBigShiftSample(root=thisfolder/"data", shardedim3root=thisfolder/"data"/"flatw", samp=SlideID, logroot=thisfolder/"test_for_jenkins"/"annowarp", uselogfiles=False, selectrectangles=[1])
    assertAlmostEqual(
      units.convertpscale(s.run(), s.pscale/10, s.pscale),
      np.array({
        "M21_1": [4.276086656088661, 10.14277048484133],
      }[SlideID])*s.onepixel,
      rtol=1e-6
    )

  def testDetectBigShiftFastUnits(self, SlideID="M21_1"):
    with units.setup_context("fast_microns"):
      self.testDetectBigShift(SlideID=SlideID)

  def testRenameAnnotation(self, **kwargs):
    self.testCohort(units="fast_microns", variant="rename", **kwargs)

  def testEmptyAnnotation(self, **kwargs):
    self.testCohort(units="fast_microns", variant="empty", **kwargs)
