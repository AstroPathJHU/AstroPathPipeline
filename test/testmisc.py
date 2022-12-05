import collections, csv, cv2, datetime, hashlib, itertools, logging, more_itertools, numpy as np, os, pathlib, re, time
from astropath.shared.annotationpolygonxmlreader import AllowedAnnotation, checkannotations, writeannotationcsvs, writeannotationinfo
from astropath.shared.contours import findcontoursaspolygons
from astropath.shared.csvclasses import Annotation, Region, Vertex
from astropath.shared.logging import getlogger, printlogger
from astropath.shared.overlap import rectangleoverlaplist_fromcsvs
from astropath.shared.polygon import Polygon, PolygonFromGdal, SimplePolygon
from astropath.shared.rectangle import Rectangle
from astropath.slides.align.alignsample import AlignSample
from astropath.slides.prepdb.prepdbsample import PrepDbSample
from astropath.slides.prepdb.prepdbcohort import PrepDbCohort
from astropath.shared.samplemetadata import APIDDef, MakeSampleDef, SampleDef
from astropath.utilities import units
from astropath.utilities.version.git import thisrepo
from astropath.utilities.tableio import readtable, writetable
from .testbase import assertAlmostEqual, TestBaseCopyInput, TestBaseSaveOutput

thisfolder = pathlib.Path(__file__).parent

class TestMisc(TestBaseCopyInput, TestBaseSaveOutput):
  @property
  def outputfilenames(self):
    yield from [
      thisfolder/"test_for_jenkins"/"misc"/"standaloneannotations"/"M206"/"M206_Scan1.annotationinfo.csv",
      thisfolder/"test_for_jenkins"/"misc"/"standaloneannotations"/"M206"/"M206_annotations.csv",
      thisfolder/"test_for_jenkins"/"misc"/"standaloneannotations"/"M206"/"M206_regions.csv",
      thisfolder/"test_for_jenkins"/"misc"/"standaloneannotations"/"M206"/"M206_vertices.csv",
      thisfolder/"test_for_jenkins"/"misc"/"standaloneannotations"/"M21_1"/"M21_1_Scan1.annotationinfo.csv",
      thisfolder/"test_for_jenkins"/"misc"/"standaloneannotations"/"M21_1"/"M21_1_annotations.csv",
      thisfolder/"test_for_jenkins"/"misc"/"standaloneannotations"/"M21_1"/"M21_1_regions.csv",
      thisfolder/"test_for_jenkins"/"misc"/"standaloneannotations"/"M21_1"/"M21_1_vertices.csv",

      thisfolder/"test_for_jenkins"/"misc"/"makesampledef"/"sampledef.csv",

      thisfolder/"test_for_jenkins"/"misc"/"tableappend"/"sampledef.csv",
      thisfolder/"test_for_jenkins"/"misc"/"tableappend"/"noheader.csv",
    ]

    for folder in ("error_regex", "require_commit", "job_lock"):
      yield from [
        thisfolder/"test_for_jenkins"/"misc"/folder/"logfiles"/"prepdb.log",
        thisfolder/"test_for_jenkins"/"misc"/folder/"M21_1"/"logfiles"/"M21_1-prepdb.log",
        thisfolder/"test_for_jenkins"/"misc"/folder/"M21_1"/"dbload"/"M21_1_qptiff.jpg",
      ]
      for csvfile in ("batch", "constants", "exposures", "overlap", "qptiff", "rect"):
        yield thisfolder/"test_for_jenkins"/"misc"/folder/"M21_1"/"dbload"/f"M21_1_{csvfile}.csv"

  @classmethod
  def filestocopy(cls):
    for SlideID in "M21_1", "M206":
      yield thisfolder/"data"/SlideID/"im3"/"Scan1"/f"{SlideID}_Scan1.annotations.polygons.xml", thisfolder/"test_for_jenkins"/"misc"/"standaloneannotations"/SlideID
    yield thisfolder/"data"/"upkeep_and_progress"/"AstropathAPIDdef_0.csv", thisfolder/"test_for_jenkins"/"misc"/"sampledef_from_apid"/"upkeep_and_progress"
    for SlideID in "M21_1",:
      copyfrom = thisfolder/"data"/SlideID/"inform_data"/"Component_Tiffs"
      copyto = thisfolder/"test_for_jenkins"/"misc"/"missingbatchprocedure"/SlideID/"inform_data"/"Component_Tiffs"
      for filename in copyfrom.glob("*_component_data.tif"):
        yield filename, copyto
        break
      else:
        assert False
      for filename in copyfrom.glob("*_component_data_w_seg.tif"):
        yield filename, copyto/"only_segmented"
        break
      else:
        assert False

  def testRectangleOverlapList(self):
    l = rectangleoverlaplist_fromcsvs(thisfolder/"data"/"M21_1"/"dbload", layer=1)
    islands = l.islands()
    self.assertEqual(len(islands), 2)
    l2 = rectangleoverlaplist_fromcsvs(thisfolder/"data"/"M21_1"/"dbload", selectrectangles=lambda x: x.n in islands[0], layer=1)
    self.assertEqual(l2.islands(), [l.islands()[0]])

    r = l.rectangles[l.rectangledict[1]]
    target = {
      6: l.rectangles[l.rectangledict[2]],
      7: l.rectangles[l.rectangledict[4]],
      8: l.rectangles[l.rectangledict[5]],
      9: l.rectangles[l.rectangledict[6]],
    }
    self.assertEqual(l.neighbors(r), target)

  def testRectangleOverlapListFastUnits(self):
    with units.setup_context("fast"):
      self.testRectangleOverlapList()

  def testPolygonAreas(self, seed=None):
    logger = printlogger("polygonareas")
    p = PolygonFromGdal(pixels="POLYGON ((1 1,2 1,2 2,1 2,1 1))", pscale=5, annoscale=3)
    assertAlmostEqual(p.area, p.onepixel**2, rtol=1e-15)
    assertAlmostEqual(p.perimeter, 4*p.onepixel, rtol=1e-15)
    p = PolygonFromGdal(pixels="POLYGON ((1 1,4 1,4 4,1 4,1 1),(2 2,2 3,3 3,3 2,2 2))", pscale=5, annoscale=3)
    assertAlmostEqual(p.area, 8*p.onepixel**2, rtol=1e-15)
    assertAlmostEqual(p.perimeter, 16*p.onepixel, rtol=1e-15)

    if seed is None:
      try:
        seed = int(hashlib.sha1(os.environ["BUILD_TAG"].encode("utf-8")).hexdigest(), 16)
      except KeyError:
        pass
    rng = np.random.default_rng(seed)
    try:
      areas = 0
      while np.any(areas == 0):
        xysx2 = units.distances(pixels=rng.integers(-10, 11, (2, 100, 2)), pscale=3)
        vertices = [[Vertex(regionid=None, vid=i, x=x, y=y, pscale=5, annoscale=3) for i, (x, y) in enumerate(xys) if x or y] for xys in xysx2]
        p1, p2 = [SimplePolygon(vertices=vv) for vv in vertices]
        areas = np.array([p1.area, p2.area, (p1-p2).area])
      assertAlmostEqual(p1.area-p2.area, (p1-p2).area)
      assertAlmostEqual((p1-p1).area, 0)
      for p in p1, p2, p1-p2:
        assertAlmostEqual(
          p.area,
          p.gdalpolygon().Area() * p.onepixel**2,
          atol=1e-14,
        )
        assertAlmostEqual(
          p.perimeter,
          p.gdalpolygon().Boundary().Length() * p.onepixel,
          atol=1e-14,
        )
        self.maxDiff = None
        self.assertEqual(
          str(p),
          str(p.gdalpolygon(round=True)),
        )
    except:
      logger.error(xysx2)
      raise

  def testPolygonAreasFastUnitsPixels(self):
    with units.setup_context("fast_pixels"):
      self.testPolygonAreas()

  def testPolygonAreasFastUnitsMicrons(self):
    with units.setup_context("fast_microns"):
      self.testPolygonAreas()

  def testPolygonNumpyArray(self):
    logger = printlogger("polygonnumpy")
    polystring = "POLYGON((1.0001 1.0001, 1.0001 8.9999, 8.9999 8.9999, 8.9999 1.0001, 1.0001 1.0001), (4.0001 5.9999, 7.9999 5.9999, 7.9999 4.0001, 4.0001 4.0001))"
    poly = PolygonFromGdal(pixels=polystring, pscale=1, annoscale=3)
    nparray = poly.numpyarray(shape=(10, 10), dtype=np.uint8)
    #doesn't work for arbitrary polygons unless you increase the tolerance, but works for a polygon with right angles
    assertAlmostEqual(poly.area / poly.onepixel**2, np.sum(nparray), rtol=1e-3)

    poly2, = findcontoursaspolygons(nparray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE, pscale=poly.pscale, annoscale=poly.annoscale, logger=logger)
    #does not equal poly1, some gets eaten away

  def testComplicatedPolygon(self):
    polystring1 = "POLYGON((1 1, 1 9, 9 9, 9 1, 1 1))"
    polystring2 = "POLYGON((2 2, 2 8, 8 8, 8 2, 2 2))"
    polystring3 = "POLYGON((3 3, 3 7, 7 7, 7 3, 3 3))"
    poly1 = PolygonFromGdal(pixels=polystring1, pscale=1, annoscale=3)
    poly2 = PolygonFromGdal(pixels=polystring2, pscale=1, annoscale=3)
    poly3 = PolygonFromGdal(pixels=polystring3, pscale=1, annoscale=3)
    inner = Polygon(poly2, [poly3])
    poly = Polygon(poly1, [inner])
    assertAlmostEqual(poly.area, 44*poly.onepixel**2)
    assertAlmostEqual(poly.perimeter, 72*poly.onepixel)

  def testPolygonNumpyArrayFastUnits(self):
    with units.setup_context("fast_microns"):
      self.testPolygonNumpyArray()

  def testStandaloneAnnotations(self, SlideID="M21_1"):
    try:
      folder = thisfolder/"test_for_jenkins"/"misc"/"standaloneannotations"/SlideID
      xmlfile = folder/f"{SlideID}_Scan1.annotations.polygons.xml"
      infofile = folder/f"{SlideID}_Scan1.annotationinfo.csv"

      args1 = [os.fspath(xmlfile), "--infofile", os.fspath(infofile), "--annotations-on-qptiff"]
      info = writeannotationinfo(args1)
      args2 = [os.fspath(infofile)]
      checkannotations(args2)
      args3 = [os.fspath(folder), os.fspath(infofile), "--csvprefix", SlideID]
      writeannotationcsvs(args3)
      extrakwargs = {"annotationinfos": info, "pscale": 1, "apscale": 1}
      for filename, cls in (
        (f"{SlideID}_annotations.csv", Annotation),
        (f"{SlideID}_vertices.csv", Vertex),
        (f"{SlideID}_regions.csv", Region),
      ):
        try:
          rows = readtable(folder/filename, cls, checkorder=True, checknewlines=True, extrakwargs=extrakwargs)
          targetrows = readtable(thisfolder/"data"/"reference"/"misc"/"standaloneannotations"/SlideID/"dbload"/filename, cls, checkorder=True, checknewlines=True, extrakwargs=extrakwargs)
          for i, (row, target) in enumerate(more_itertools.zip_equal(rows, targetrows)):
            assertAlmostEqual(row, target, rtol=1e-5, atol=8e-7)
          if cls == Annotation:
            extrakwargs["annotations"] = rows
            del extrakwargs["annotationinfos"]
        except:
          raise ValueError("Error in "+filename)
    except:
      self.saveoutput()
      raise
    finally:
      self.removeoutput()

  def testStandaloneAnnotationsFastUnits(self, SlideID="M206"):
    with units.setup_context("fast"):
      self.testStandaloneAnnotations(SlideID=SlideID)

  def testPolygonVertices(self):
    polystring = "POLYGON ((1 1,2 1,2 2,1 2,1 1))"
    p = PolygonFromGdal(pixels=polystring, pscale=5, annoscale=3)
    p2 = SimplePolygon(vertices=p.outerpolygon.vertices)
    self.assertEqual(str(p), polystring)
    self.assertEqual(str(p2), polystring)
    assertAlmostEqual(p.outerpolygon.vertices, p2.vertices)
    assertAlmostEqual(p.outerpolygon.vertexarray, p2.vertexarray)

  def testPolygonVerticesFastUnits(self):
    with units.setup_context("fast_microns"):
      self.testPolygonVertices()

  def testSampleDef(self):
    self.maxDiff = None
    s1 = SampleDef(samp="M21_1", root=thisfolder/"data")
    s2 = SampleDef(samp="M21_1", apidfile=thisfolder/"data"/"upkeep_and_progress"/"AstropathAPIDdef_0.csv", SampleID=s1.SampleID)
    s3 = SampleDef(samp="M21_1", apidfile=thisfolder/"data"/"upkeep_and_progress"/"AstropathAPIDdef_0_oldformat.csv", Scan=s1.Scan, SampleID=s1.SampleID)
    s4 = SampleDef(samp="M21_1", apidfile=thisfolder/"data"/"upkeep_and_progress"/"AstropathAPIDdef_0_oldformat.csv", root=thisfolder/"data")
    APID, = {APID for APID in readtable(thisfolder/"data"/"upkeep_and_progress"/"AstropathAPIDdef_0.csv", APIDDef) if APID.SlideID == "M21_1"}
    s5 = SampleDef(samp=APID, SampleID=s1.SampleID)
    APID, = {APID for APID in readtable(thisfolder/"data"/"upkeep_and_progress"/"AstropathAPIDdef_0_oldformat.csv", APIDDef) if APID.SlideID == "M21_1"}
    s6 = SampleDef(samp=APID, Scan=s1.Scan, SampleID=s1.SampleID)
    s7 = SampleDef(samp=APID, root=thisfolder/"data")
    s8 = SampleDef(samp="M21_1", root=thisfolder/"test_for_jenkins"/"misc"/"sampledef_from_apid", Project=0, SampleID=s1.SampleID)
    self.assertEqual(s1, s2)
    self.assertEqual(s1, s3)
    self.assertEqual(s1, s4)
    self.assertEqual(s1, s5)
    self.assertEqual(s1, s6)
    self.assertEqual(s1, s7)
    self.assertEqual(s1, s8)

  def testMakeSampleDef(self):
    self.maxDiff = None
    outfile = thisfolder/"test_for_jenkins"/"misc"/"makesampledef"/"sampledef.csv"
    outfile.parent.mkdir(parents=True, exist_ok=True)
    reference = thisfolder/"data"/"sampledef.csv"
    args = [os.fspath(thisfolder/"data"), "--apidfile", os.fspath(thisfolder/"data"/"upkeep_and_progress"/"AstropathAPIDdef_0.csv"), "--first-sample-id", "1", "--outfile", os.fspath(outfile)]
    MakeSampleDef.runfromargumentparser(args)

    try:
      rows = readtable(outfile, SampleDef, checkorder=True, checknewlines=True)
      targetrows = readtable(reference, SampleDef, checkorder=True, checknewlines=True)

      for row, target in more_itertools.zip_equal(rows, targetrows):
        assertAlmostEqual(row, target)
    except:
      self.saveoutput()
      raise
    else:
      self.removeoutput()

  def testTableAppend(self):
    self.maxDiff = None
    outfile = thisfolder/"test_for_jenkins"/"misc"/"tableappend"/"sampledef.csv"
    noheader = outfile.parent/"noheader.csv"
    outfile.parent.mkdir(parents=True, exist_ok=True)
    reference = thisfolder/"data"/"sampledef.csv"
    sampledefs = readtable(reference, SampleDef, checkorder=True, checknewlines=True)
    firstbatch = sampledefs[:3]
    secondbatch = sampledefs[3:]
    writetable(outfile, firstbatch)
    writetable(outfile, secondbatch, append=True)
    writetable(noheader, firstbatch, header=False)
    with self.assertRaises(ValueError):
      writetable(noheader, secondbatch, append=True)
    onepixel = units.Distance(pixels=1, pscale=1)
    with self.assertRaises(ValueError):
      writetable(noheader, [Vertex(regionid=1, vid=1, x=onepixel, y=onepixel)], append=True, header=False)
    with self.assertRaises(ValueError):
      writetable(noheader, [Rectangle(pscale=1, n=1, x=onepixel, y=onepixel, cx=onepixel, cy=onepixel, w=onepixel, h=onepixel, file='file.im3', t=datetime.datetime.now())], append=True, header=False)
    writetable(noheader, secondbatch, append=True, header=False)

    try:
      for rows in (
        readtable(outfile, SampleDef, checkorder=True, checknewlines=True),
        readtable(noheader, SampleDef, checkorder=True, checknewlines=True, header=False),
      ):
        for row, target in more_itertools.zip_equal(rows, sampledefs):
          assertAlmostEqual(row, target)
    except:
      self.saveoutput()
      raise
    else:
      self.removeoutput()

  def testHPFOffset(self):
    root = thisfolder/"data"
    shardedim3root = thisfolder/"data"/"flatw"
    for SlideID in "M21_1", "YZ71":
      with self.subTest(SlideID=SlideID):
        s = AlignSample(samp=SlideID, root=root, shardedim3root=shardedim3root, uselogfiles=False)
        target = np.array({
          "M21_1": [535.2, 400],
          "YZ71": [744.8, 558.4],
        }[SlideID]) * s.onemicron
        assertAlmostEqual(s.hpfoffset, target)

  def testAnnotationVariations(self):
    annotations = AllowedAnnotation.allowedannotations()
    byname = collections.defaultdict(list)
    bylayer = collections.defaultdict(list)
    bycolor = collections.defaultdict(list)
    for a in annotations:
      byname[a.name].append(a)
      bylayer[a.layer].append(a)
      bycolor[a.color].append(a)

    for k, v in bycolor.items():
      #if there are multiple with the same color, they should be variations
      #of the first one.  e.g. good tissue and good tissue x
      first = v[0]
      for later in v[1:]:
        self.assertRegex(later.name, first.name+" .*")

    for k, v in byname.items():
      self.assertLengthEqual(v, 1)

    for k, v in bylayer.items():
      self.assertLengthEqual(v, 1)

  def testMissingOutputs(self, SlideID="M21_1"):
    dbloadroot = thisfolder/"test_for_jenkins"/"misc"/"missingoutputs"
    args = [os.fspath(thisfolder/"data"), SlideID, "--xmlfolder", os.fspath(thisfolder/"data"/"raw"/SlideID), "--allow-local-edits", "--dbloadroot", os.fspath(dbloadroot), "--logroot", os.fspath(dbloadroot)]
    class DummyPrepDbSample(PrepDbSample):
      def run(self, *args, **kwargs): pass
    args = [os.fspath(thisfolder/"data"), SlideID, "--xmlfolder", os.fspath(thisfolder/"data"/"raw"/SlideID), "--allow-local-edits", "--dbloadroot", os.fspath(dbloadroot)]
    with self.assertRaises(RuntimeError):
      DummyPrepDbSample.runfromargumentparser([*args, "--no-log"])

  def testMissingBatchProcedure(self, SlideID="M21_1"):
    informdataroot = thisfolder/"test_for_jenkins"/"misc"/"missingbatchprocedure"
    s1 = PrepDbSample(thisfolder/"data", SlideID, uselogfiles=False)
    s2 = PrepDbSample(thisfolder/"data", SlideID, informdataroot=informdataroot, uselogfiles=False)
    s3 = PrepDbSample(thisfolder/"data", SlideID, informdataroot=informdataroot/"only_segmented", uselogfiles=False)
    with self.assertRaises(FileNotFoundError):
      s3.nlayersunmixed
    self.assertEqual(s1.nlayersunmixed, s2.nlayersunmixed)

  def testRequireCommit(self):
    testrequirecommit = thisrepo.getcommit("cf271f3a")

    root = thisfolder/"data"
    dbloadroot = logroot = thisfolder/"test_for_jenkins"/"misc"/"require_commit"
    SlideID = "M21_1"
    logfolder = logroot/SlideID/"logfiles"
    logfolder.mkdir(exist_ok=True, parents=True)

    logfile = logfolder/f"{SlideID}-prepdb.log"
    with getlogger(root=logroot, samp=SampleDef(SlideID=SlideID, Project=0, Cohort=0), module="prepdb", reraiseexceptions=False, uselogfiles=True, printthreshold=logging.CRITICAL+1) as logger:
      logger.info("testing")

    with open(logfile, newline="") as f:
      f, f2 = itertools.tee(f)
      startregex = re.compile(PrepDbSample.logstartregex())
      reader = csv.DictReader(f, fieldnames=("Project", "Cohort", "SlideID", "message", "time"), delimiter=";")
      for row in reader:
        match = startregex.match(row["message"])
        istag = not bool(match.group("commit"))
        if match: break
      else:
        assert False
      contents = "".join(f2)

    usecommit = testrequirecommit.parents[0]
    #purposely write an INVALID commit hash (with aaaaa at the end)
    #testing with --require-commit is in testgeomcell.py
    if istag:
      contents = contents.replace(match.group("version"), f"{match.group('version')}.dev0+g{usecommit.shorthash(8)}aaaaa")
    else:
      contents = contents.replace(match.group("commit"), usecommit.shorthash(8)+"aaaaa")

    with open(logfile, "w", newline="") as f:
      f.write(contents)

    s = PrepDbSample(root=root, dbloadroot=dbloadroot, logroot=logroot, samp=SlideID)
    s.dbload.mkdir(parents=True, exist_ok=True)
    for filename in s.outputfiles:
      filename.touch()

    args = [os.fspath(thisfolder/"data"), "--sampleregex", SlideID, "--debug", "--units", "fast", "--xmlfolder", os.fspath(thisfolder/"data"/"raw"/SlideID), "--allow-local-edits", "--ignore-dependencies", "--dbloadroot", os.fspath(dbloadroot), "--logroot", os.fspath(logroot), "--skip-qptiff"]
    PrepDbCohort.runfromargumentparser(args) #this should not run anything
    with open(s.csv("rect")) as f: assert not f.read().strip()
    PrepDbCohort.runfromargumentparser(args + ["--require-commit", str(testrequirecommit.parents[0].parents[0])])
    with open(s.csv("rect")) as f: assert f.read().strip()

    self.removeoutput()

  def testErrorRegex(self):
    root = thisfolder/"data"
    dbloadroot = logroot = thisfolder/"test_for_jenkins"/"misc"/"error_regex"
    SlideID = "M21_1"
    logfolder = logroot/SlideID/"logfiles"
    logfolder.mkdir(exist_ok=True, parents=True)

    with getlogger(root=logroot, samp=SampleDef(SlideID=SlideID, Project=0, Cohort=0), module="prepdb", reraiseexceptions=False, uselogfiles=True, printthreshold=logging.CRITICAL+1):
      raise ValueError("testing error regex matching")

    s = PrepDbSample(root=root, dbloadroot=dbloadroot, logroot=logroot, samp=SlideID)
    s.dbload.mkdir(parents=True, exist_ok=True)
    for filename in s.outputfiles:
      filename.touch()

    args = [os.fspath(thisfolder/"data"), "--sampleregex", SlideID, "--debug", "--units", "fast", "--xmlfolder", os.fspath(thisfolder/"data"/"raw"/SlideID), "--allow-local-edits", "--ignore-dependencies", "--dbloadroot", os.fspath(dbloadroot), "--logroot", os.fspath(logroot), "--skip-qptiff", "--rerun-error", "other error"]
    PrepDbCohort.runfromargumentparser(args) #this should not run anything
    with open(s.csv("rect")) as f: assert not f.read().strip()
    PrepDbCohort.runfromargumentparser(args + ["--rerun-error", "testing error regex matching"])
    with open(s.csv("rect")) as f: assert f.read().strip()

    self.removeoutput()

  def testJobLock(self):
    root = thisfolder/"data"
    dbloadroot = logroot = thisfolder/"test_for_jenkins"/"misc"/"job_lock"
    SlideID = "M21_1"
    logfolder = logroot/SlideID/"logfiles"
    logfolder.mkdir(exist_ok=True, parents=True)

    s = PrepDbSample(root=root, dbloadroot=dbloadroot, logroot=logroot, samp=SlideID)
    s.dbload.mkdir(parents=True, exist_ok=True)
    for filename in s.outputfiles:
      filename.touch()

    with open(s.lockfile, "w") as f: pass

    args = [os.fspath(thisfolder/"data"), "--sampleregex", SlideID, "--debug", "--units", "fast", "--xmlfolder", os.fspath(thisfolder/"data"/"raw"/SlideID), "--allow-local-edits", "--ignore-dependencies", "--dbloadroot", os.fspath(dbloadroot), "--logroot", os.fspath(logroot), "--skip-qptiff", "--rerun-error", "other error"]
    PrepDbCohort.runfromargumentparser(args + ["--job-lock-timeout", "0:0:1"]) #this should not run anything
    with open(s.csv("rect")) as f: assert not f.read().strip()
    time.sleep(1)
    PrepDbCohort.runfromargumentparser(args + ["--job-lock-timeout", "0:0:1"])
    with open(s.csv("rect")) as f: assert f.read().strip()

    self.removeoutput()
