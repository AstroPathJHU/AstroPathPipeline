import contextlib, datetime, job_lock, os, pathlib, re
from ...utilities.config import CONST as UNIV_CONST
from ...shared.cohort import FilterResult, GeomFolderCohort, GlobalDbloadCohort, GlobalDbloadCohortBase, PhenotypeFolderCohort, SampleFilter, SelectRectanglesCohort, WorkflowCohort
from ...shared.csvclasses import MakeClinicalInfo, ControlCore, ControlFlux, ControlSample, GlobalBatch, MergeConfig
from ...shared.samplemetadata import SampleDef
from ...shared.workflowdependency import WorkflowDependency
from .csvscansample import CsvScanBase, CsvScanSample, RunCsvScanBase

class CsvScanGlobalCsv(CsvScanBase, GlobalDbloadCohortBase, WorkflowDependency, contextlib.ExitStack):
  def __init__(self, *args, batchroot, **kwargs):
    super().__init__(*args, **kwargs)
    if batchroot is None: batchroot = self.root
    self.batchroot = pathlib.Path(batchroot)

  def __enter__(self):
    result = super().__enter__()
    self.enter_context(self.logger)
    return result

  @property
  def logger(self): return super().logger

  @property
  def samp(self):
    return SampleDef(Project=self.Project, Cohort=self.Cohort, SlideID=self.SlideID)
  @property
  def SlideID(self):
    return f"project{self.Project}"

  def inputfiles(self, **kwargs):
    return []  #will be checked in run()

  def runcsvscan(self, *, checkcsvs=True, ignorecsvs=[], segmentationalgorithms=None):
    toload = []
    batchcsvs = {
      self.batchroot/"Batch"/f"{csv}_{s.BatchID:02d}.csv"
      for csv in ("MergeConfig", "BatchID")
      for s in self.sampledefs()
    }
    otherbatchcsvs = {
      self.batchroot/"Batch"/f"{csv}_{BatchID:02d}.csv"
      for csv in ("MergeConfig", "BatchID")
      for BatchID in range(1, max(s.BatchID for s in self.sampledefs())+1)
    } - batchcsvs

    CSfoldername = self.root.name
    if not CSfoldername:
      assert len(self.root.parts) == 1
      splitdrive = self.root.drive.split("\\")
      assert len(splitdrive) == 4 and splitdrive[0] == splitdrive[1] == ""
      CSfoldername = splitdrive[3]

    tablename = CSfoldername.replace("Clinical_Specimen", "Clinical_Table_Specimen")
    if tablename == CSfoldername: raise ValueError(f"Expected the folder name {CSfoldername} to have Clinical_Specimen in it")
    clinicalcsvs = {
      csv
      for csv in (self.root/"Clinical").glob(f"{tablename}_*.csv")
      if re.match(f"{tablename}_[0-9]+.csv", csv.name)
    }
    if not clinicalcsvs:
      raise FileNotFoundError(f"Didn't find any clinical csvs in {self.root/'Clinical'}")
    globalcontrolcsvs = [
      self.root/"Ctrl"/f"project{self.Project}_ctrl{ctrl}.csv"
      for ctrl in ("cores", "fluxes", "samples")
    ]
    for i, csv in enumerate(globalcontrolcsvs):
      if not csv.exists():
        capital = csv.with_name(csv.name.replace("project", "Project"))
        if capital.exists():
          globalcontrolcsvs[i] = capital
    ctrlsamplescsv = globalcontrolcsvs[-1]
    try:
      controlcsvs = {
        self.root/sample.SlideID/UNIV_CONST.DBLOAD_DIR_NAME/f"{sample.SlideID}_control.csv"
        for sample in self.readtable(ctrlsamplescsv, ControlSample)
      }
    except FileNotFoundError:
      controlcsvs = set()
    expectcsvs = batchcsvs | clinicalcsvs | set(globalcontrolcsvs)
    queuecsvs = {log.with_name(log.name.replace(".log", "-queue.csv")) for log in (self.root/"logfiles").iterdir()}
    optionalcsvs = otherbatchcsvs | controlcsvs
    unknowncsvs = set()
    for csv in self.globalcsvs:
      if csv == self.csv("loadfiles"): continue
      if csv == self.root/"sampledef.csv": continue
      if csv in queuecsvs: continue
      if csv.parent == self.root/"tmp_inform_data": continue
      if csv.parent == self.root/"upkeep_and_progress": continue

      try:
        expectcsvs.remove(csv)
      except KeyError:
        try:
          optionalcsvs.remove(csv)
        except KeyError:
          if any(regex.match(os.fspath(csv.relative_to(self.root))) for regex in ignorecsvs): continue
          unknowncsvs.add(csv)
          continue

      if csv.parent == self.batchroot/"Batch":
        match = re.match("(.*)_[0-9]+[.]csv", csv.name)
        csvclass, tablename = {
          "BatchID": (GlobalBatch, "Batch"),
          "MergeConfig": (MergeConfig, "MergeConfig")
        }[match.group(1)]
        extrakwargs = {}
        idx = 1
      elif csv.parent == self.root/"Clinical":
        csvclass = MakeClinicalInfo(csv)
        tablename = "Clinical"
        extrakwargs = {}
        idx = 2
      elif csv.parent == self.root/"Ctrl":
        match = re.match(f"[Pp]roject{self.Project}_(.*)[.]csv", csv.name)
        csvclass, tablename = {
          "ctrlcores": (ControlCore, "Ctrlcores"),
          "ctrlfluxes": (ControlFlux, "Ctrlfluxes"),
          "ctrlsamples": (ControlSample, "Ctrlsamples"),
        }[match.group(1)]
        extrakwargs = {}
        idx = 3
      elif csv in controlcsvs:
        continue
      else:
        assert False, csv

      toload.append({"csv": csv, "csvclass": csvclass, "tablename": tablename, "extrakwargs": extrakwargs, "SlideID": f"project{self.Project}-{idx}"})
      toload.sort(key=lambda x: (str(x["csv"]).lower()))

    if expectcsvs or unknowncsvs:
      errors = []
      if expectcsvs:
        errors.append("Some csvs are missing: "+", ".join(str(_) for _ in sorted(expectcsvs)))
      if unknowncsvs:
        errors.append("Unknown csvs: "+", ".join(str(_) for _ in sorted(unknowncsvs)))
      raise ValueError("\n".join(errors))

    loadfiles = [self.processcsv(checkcsv=checkcsvs, **kwargs) for kwargs in toload]

    self.dbload.mkdir(exist_ok=True)
    self.writecsv("loadfiles", loadfiles, header=False)

  def run(self, *args, **kwargs): return self.runcsvscan(*args, **kwargs)

  @property
  def globalcsvs(self):
    sampledefs = self.readtable(self.root/"sampledef.csv", SampleDef)
    slideids = [s.SlideID for s in sampledefs]
    for folder in {self.root}:
      for subfolder in folder.iterdir():
        if subfolder.name.endswith(".csv"): yield subfolder
        if not subfolder.is_dir(): continue
        if subfolder.name in slideids: continue
        yield from subfolder.rglob("*.csv")

  @classmethod
  def defaultunits(cls): return CsvScanSample.defaultunits()
  @classmethod
  def getlogfile(cls, *, logroot, **workflowkwargs):
    return logroot/"logfiles"/f"{cls.logmodule()}.log"
  def joblock(self, corruptfiletimeout=datetime.timedelta(minutes=10), **kwargs):
    lockfiles = [mainlog.with_suffix(".lock") for mainlog in self.mainlogs]
    return job_lock.MultiJobLock(*lockfiles, corruptfiletimeout=corruptfiletimeout, mkdir=True, **kwargs)

  @classmethod
  def usegloballogger(cls): return True
  @classmethod
  def logstartregex(cls): return rf"{cls.logmodule()} v[0-9a-f.devgd+]+$"
  @classmethod
  def logendregex(cls): return rf"end {cls.logmodule()}"

  @classmethod
  def getoutputfiles(cls, *, dbloadroot, Project, **workflowkwargs):
    return [dbloadroot/UNIV_CONST.DBLOAD_DIR_NAME/f"project{Project}_loadfiles.csv"]

  @property
  def workflowkwargs(self):
    return {
      **super().workflowkwargs,
      "SlideID": f"project{self.Project}",
      "Project": self.Project,
    }

  def workflowdependencies(self, **kwargs):
    return [(CsvScanSample, sampledef.SlideID) for sampledef in self.sampledefs() if sampledef]

  usetissue = True
  useTMA = False

class CsvScanCohort(GlobalDbloadCohort, GeomFolderCohort, PhenotypeFolderCohort, SelectRectanglesCohort, WorkflowCohort, RunCsvScanBase):
  sampleclass = CsvScanSample
  __doc__ = sampleclass.__doc__

  def __init__(self, *args, doglobalcsv, **kwargs):
    self.__doglobalcsv = doglobalcsv
    super().__init__(*args, **kwargs)

  @property
  def doglobalcsv(self): return self.__doglobalcsv

  @classmethod
  def makeargumentparser(cls, **kwargs):
    p = super().makeargumentparser(**kwargs)
    p.add_argument("--do-global-csv", action="store_true", help="also make the global csv for the project")
    return p

  @classmethod
  def initkwargsfromargumentparser(cls, parsed_args_dict):
    result = {
      **super().initkwargsfromargumentparser(parsed_args_dict),
      "doglobalcsv": parsed_args_dict.pop("do_global_csv"),
    }
    def correctsegmentations(self, sample, **kwargs):
      segmentationalgorithms = frozenset(self.segmentationalgorithms)
      folder = self.geomroot/sample.SlideID/"geom"
      geomalgorithms = frozenset(subfolder.name for subfolder in folder.glob("*/") if any(subfolder.iterdir()))
      messages = []
      missing = segmentationalgorithms - geomalgorithms
      extras = geomalgorithms - segmentationalgorithms
      if missing: messages.append(f"Sample has missing algorithms in geom: {', '.join(missing)}.")
      if extras: messages.append(f"Sample has extra algorithms in geom: {', '.join(extras)}.")
      if missing or extras:
        return FilterResult(False, " ".join(messages))
      else:
        return FilterResult(True, "Sample has the right segmentation algorithms in geom.")
    if not parsed_args_dict["print_mode"] and not result["skipcells"]:
      result["slideidfilters"].append(SampleFilter(correctsegmentations, None, None))
    return result

  def samplesandsampledefswithfilters(self, **kwargs):
    yield from super().samplesandsampledefswithfilters(**kwargs)
    if self.doglobalcsv:
      globalcsv = self.globalcsv()
      filters = [filter(self, globalcsv, **kwargs) for filter in self.samplefilters]
      yield self.globalcsv(), filters

  @property
  def initiatesamplekwargs(self):
    return {
      **super().initiatesamplekwargs,
      "segmentationalgorithms": self.segmentationalgorithms,
      "skipannotations": self.skipannotations,
      "skipcells": self.skipcells,
    }

  def runsample(self, sample, **kwargs):
    return sample.runcsvscan(**kwargs)

  def run(self, *, checkcsvs=True, print_mode=None, **kwargs):
    super().run(checkcsvs=checkcsvs, print_mode=print_mode, **kwargs)

  @property
  def globalcsvinitkwargs(self):
    return {
      k: v for k, v in self.initiatesamplekwargs.items()
      if {
        "batchroot": True,
        "dbloadroot": True,
        "geomroot": False,
        "im3root": False,
        "informdataroot": False,
        "logroot": True,
        "phenotyperoot": False,
        "reraiseexceptions": True,
        "root": True,
        "selectrectangles": False,
        "uselogfiles": True,
        "xmlfolders": False,
        "moremainlogroots": True,
        "skipstartfinish": True,
        "printthreshold": True,
        "segmentationalgorithms": True,
        "sampledefroot": True,
        "skipannotations": True,
        "skipcells": True,
      }[k]
    }

  def globalcsv(self):
    return CsvScanGlobalCsv(**self.globalcsvinitkwargs)

def main(args=None):
  CsvScanCohort.runfromargumentparser(args)

if __name__ == "__main__":
  main()
